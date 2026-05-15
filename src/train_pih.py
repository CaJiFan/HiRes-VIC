"""
CleanRL-style training script for PiH (Peg-in-Hole) with ManiSkill.

Primary algorithm: SAC (default) — hyperparameters mirror the SB3 SAC used for
the Wipe baselines to keep comparisons fair (batch_size=512, buffer=1M,
tau=0.002, lr=3e-4, gamma parameterised).

Secondary algorithm: PPO (--algorithm PPO) — standard CleanRL PPO for
on-policy ablations.  Activated by flag; all PPO-specific args are parsed
but silently ignored by SAC and vice-versa.

Usage
─────
  # SAC baseline (no geometry)
  python src/train_pih.py --run_name debug --n_envs 8

  # SPD manifold (Riemannian)
  python src/train_pih.py --run_name pih_spd --use_spd --n_envs 8

  # Riemannian + LLM prior
  python src/train_pih.py --run_name pih_spd_llm --use_spd --use_llm_prior \\
      --llm_profile configs/pih_impedance_profile.yaml

  # PPO ablation
  python src/train_pih.py --run_name pih_ppo --algorithm PPO --use_spd --n_envs 8
"""

from __future__ import annotations

import os
import sys
import argparse
import random
import time
import math
from collections import deque
from pathlib import Path

sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), "..")))

from gymnasium import envs
import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
from torch.distributions import Normal
import gymnasium as gym
import wandb

from hires_vic.wrappers.pih_curriculum import InsertionCurriculumWrapper


# ── Argument parsing ──────────────────────────────────────────────────────────

def parse_args():
    p = argparse.ArgumentParser()

    # Shared
    p.add_argument("--run_name",       required=True)
    p.add_argument("--env",            default="PegInsertionSide-v1")
    p.add_argument("--algorithm",      default="SAC", choices=["SAC", "PPO"])
    p.add_argument("--seed",           type=int,   default=0)
    p.add_argument("--total_timesteps",type=int,   default=1_000_000)
    p.add_argument("--n_envs",         type=int,   default=8)
    p.add_argument("--gamma",          type=float, default=0.99,
                   help="Discount factor — critical for PiH (try 0.90–0.99)")
    p.add_argument("--sim_backend",    default="gpu", choices=["gpu", "cpu"])
    p.add_argument("--use_sim2real_obs", action="store_true", default=True,
                   help="Filter obs to only real-robot-compatible signals "
                        "(joint state + EEF FK + wrist F/T). Default: True")

    # Geometry flags
    p.add_argument("--use_spd",   action="store_true", help="Full 3×3 SPD manifold stiffness")
    p.add_argument("--use_lie",   action="store_true", help="SO(3) log-map for EEF orientation in obs")
    p.add_argument("--use_diag",  action="store_true", help="Diagonal SPD ablation (log-scale, no off-diagonal)")
    p.add_argument("--use_fixed", action="store_true", help="Fixed impedance — no VIC, Kp frozen in controller")

    # LLM prior
    p.add_argument("--use_llm_prior",      action="store_true")
    p.add_argument("--llm_backend",        default="ollama", choices=["openai", "ollama"])
    p.add_argument("--llm_model",          default=None)
    p.add_argument("--llm_query_interval", type=int,   default=50)
    p.add_argument("--llm_prior_weight",   type=float, default=0.4)
    p.add_argument("--llm_profile",        default="configs/pih_impedance_profile.yaml",
                   help="Path to YAML impedance profile for this task")

    # SAC hyperparameters
    p.add_argument("--learning_rate",  type=float, default=3e-4)
    p.add_argument("--buffer_size",    type=int,   default=1_000_000)
    p.add_argument("--batch_size",     type=int,   default=512)
    p.add_argument("--tau",            type=float, default=0.002)
    p.add_argument("--learning_starts",type=int,   default=5_000)
    p.add_argument("--train_freq",     type=int,   default=1,
                   help="Update networks every N env steps per worker")
    p.add_argument("--gradient_steps", type=int,   default=-1,
                   help="Gradient steps per update (-1 = n_envs, matching SB3 default)")
    p.add_argument("--autotune_alpha", action="store_true", default=True)
    p.add_argument("--alpha",          type=float, default=0.2)

    # PPO hyperparameters (ignored if algorithm=SAC)
    p.add_argument("--n_steps",    type=int,   default=512,  help="PPO: rollout steps per env")
    p.add_argument("--n_epochs",   type=int,   default=10,   help="PPO: epochs per update")
    p.add_argument("--clip_range", type=float, default=0.1,  help="PPO: clip epsilon")
    p.add_argument("--ent_coef",   type=float, default=0.01, help="PPO: entropy coefficient")
    p.add_argument("--vf_coef",    type=float, default=0.5,  help="PPO: value function coefficient")
    p.add_argument("--max_grad_norm", type=float, default=0.5)

    # Logging
    p.add_argument("--eval_freq",       type=int, default=100_000)
    p.add_argument("--n_eval_episodes", type=int, default=10)
    p.add_argument("--log_freq",        type=int, default=10_000)
    p.add_argument("--record_video",      action="store_true",
                   help="Record one eval episode as wandb video at each eval checkpoint")
    p.add_argument("--max_episode_steps", type=int, default=200,
                   help="Max steps per episode (ManiSkill default is 100; 300 recommended for PiH)")
    p.add_argument("--video_fps",       type=int, default=20)

    args = p.parse_args()
    if args.gradient_steps == -1:
        args.gradient_steps = args.n_envs
    return args


# ── Network architectures ─────────────────────────────────────────────────────
# Matched to SB3 defaults (verified from stable_baselines3/sac/policies.py):
#   SAC: net_arch=[256,256], ReLU, LOG_STD_MIN=-20, LOG_STD_MAX=2, log_std_init=-3
#   PPO: net_arch=dict(pi=[64,64], vf=[64,64]), Tanh, ortho_init=True, log_std_init=0.0

LOG_STD_MAX =  2
LOG_STD_MIN = -20   # SB3 value; CleanRL reference uses -5 — using SB3 for comparability


class SoftQNetwork(nn.Module):
    """SAC critic: (obs+action) → 256 → 256 → 1. Matches SB3 ContinuousCritic."""
    def __init__(self, obs_dim: int, action_dim: int):
        super().__init__()
        self.net = nn.Sequential(
            nn.Linear(obs_dim + action_dim, 256), nn.ReLU(),
            nn.Linear(256, 256),                  nn.ReLU(),
            nn.Linear(256, 1),
        )

    def forward(self, obs, action):
        return self.net(torch.cat([obs, action], dim=-1))


class SACActorNetwork(nn.Module):
    """
    SAC actor: obs → 256 → 256 → (mu_head, log_std_head).
    Matches SB3 Actor exactly:
      - Shared trunk [256, 256] with ReLU (SB3 SAC default)
      - Separate linear heads, no activation on output
      - log_std head bias initialised to log_std_init=-3 (SB3 default)
      - Squashed Gaussian: y = tanh(x), x ~ N(mu, exp(log_std))
    """
    def __init__(self, obs_dim: int, action_dim: int, log_std_init: float = -3.0):
        super().__init__()
        self.shared = nn.Sequential(
            nn.Linear(obs_dim, 256), nn.ReLU(),
            nn.Linear(256, 256),     nn.ReLU(),
        )
        self.mu_head      = nn.Linear(256, action_dim)
        self.log_std_head = nn.Linear(256, action_dim)
        # SB3 initialises log_std bias to log_std_init=-3
        nn.init.constant_(self.log_std_head.bias, log_std_init)

    def forward(self, obs):
        h = self.shared(obs)
        mu = self.mu_head(h)
        log_std = torch.clamp(self.log_std_head(h), LOG_STD_MIN, LOG_STD_MAX)
        return mu, log_std

    def get_action(self, obs):
        mu, log_std = self(obs)
        std = log_std.exp()
        dist = Normal(mu, std)
        x_t = dist.rsample()
        y_t = torch.tanh(x_t)
        log_prob = (dist.log_prob(x_t) - torch.log(1 - y_t.pow(2) + 1e-6)).sum(-1, keepdim=True)
        return y_t, log_prob, torch.tanh(mu)


def _ortho_init(module: nn.Module, gain: float = math.sqrt(2)):
    """Orthogonal init used by SB3 PPO (gain=sqrt(2) hidden, 0.01 policy out, 1.0 value out)."""
    if isinstance(module, nn.Linear):
        nn.init.orthogonal_(module.weight, gain=gain)
        nn.init.constant_(module.bias, 0.0)


class PPOActorNetwork(nn.Module):
    """
    PPO actor: obs → 64 → 64 → action_dim.
    Matches SB3 ActorCriticPolicy with net_arch=dict(pi=[64,64]):
      - Tanh activations (SB3 PPO default — NOT ReLU, distinct from SAC)
      - Orthogonal init: sqrt(2) for hidden layers, 0.01 for policy output layer
      - Shared log_std parameter (init=0.0, i.e. std≈1 at start)
    """
    def __init__(self, obs_dim: int, action_dim: int):
        super().__init__()
        self.net = nn.Sequential(
            nn.Linear(obs_dim, 64), nn.Tanh(),
            nn.Linear(64, 64),      nn.Tanh(),
        )
        self.mu_head = nn.Linear(64, action_dim)
        self.log_std = nn.Parameter(torch.zeros(action_dim))  # log_std_init=0.0

        for layer in self.net:
            _ortho_init(layer)
        _ortho_init(self.mu_head, gain=0.01)

    def get_action_and_log_prob(self, obs, action=None):
        mu   = self.mu_head(self.net(obs))
        std  = self.log_std.exp().expand_as(mu)
        dist = Normal(mu, std)
        if action is None:
            action = dist.sample()
        y = torch.tanh(action)
        log_prob = (dist.log_prob(action) - torch.log(1 - y.pow(2) + 1e-6)).sum(-1)
        return y, log_prob, dist.entropy().sum(-1)


class PPOCriticNetwork(nn.Module):
    """
    PPO value function: obs → 64 → 64 → 1.
    Matches SB3 ActorCriticPolicy with net_arch=dict(vf=[64,64]):
      - Tanh activations, ortho init with gain=1.0 for the output layer.
    """
    def __init__(self, obs_dim: int):
        super().__init__()
        self.net = nn.Sequential(
            nn.Linear(obs_dim, 64), nn.Tanh(),
            nn.Linear(64, 64),      nn.Tanh(),
        )
        self.value_head = nn.Linear(64, 1)
        for layer in self.net:
            _ortho_init(layer)
        _ortho_init(self.value_head, gain=1.0)

    def forward(self, obs):
        return self.value_head(self.net(obs))


# ── Replay buffer (GPU-compatible) ────────────────────────────────────────────

class ReplayBuffer:
    """Simple circular replay buffer that stores tensors on the target device."""

    def __init__(self, capacity: int, obs_dim: int, action_dim: int, device: torch.device):
        self.cap     = capacity
        self.ptr     = 0
        self.size    = 0
        self.device  = device

        self.obs     = torch.zeros((capacity, obs_dim),    device=device)
        self.next_obs= torch.zeros((capacity, obs_dim),    device=device)
        self.actions = torch.zeros((capacity, action_dim), device=device)
        self.rewards = torch.zeros((capacity, 1),          device=device)
        self.dones   = torch.zeros((capacity, 1),          device=device)

    def add(self, obs, next_obs, action, reward, done):
        # Accept numpy arrays or tensors
        def _t(x):
            if not isinstance(x, torch.Tensor):
                x = torch.tensor(x, dtype=torch.float32)
            return x.to(self.device)

        obs, next_obs, action = _t(obs), _t(next_obs), _t(action)
        reward = _t(reward).reshape(-1, 1)
        done   = _t(done).reshape(-1, 1)

        n = obs.shape[0]  # batch from n_envs
        idxs = torch.arange(n) + self.ptr
        idxs = idxs % self.cap

        self.obs[idxs]      = obs
        self.next_obs[idxs] = next_obs
        self.actions[idxs]  = action
        self.rewards[idxs]  = reward
        self.dones[idxs]    = done

        self.ptr  = int((self.ptr + n) % self.cap)
        self.size = min(self.size + n, self.cap)

    def sample(self, batch_size: int):
        idxs = torch.randint(0, self.size, (batch_size,), device=self.device)
        return (
            self.obs[idxs], self.next_obs[idxs],
            self.actions[idxs], self.rewards[idxs], self.dones[idxs],
        )

    def __len__(self):
        return self.size


# ── Environment factory ───────────────────────────────────────────────────────

def make_envs(args, is_eval=False):
    """Create ManiSkill parallel envs wrapped with ManiSkillRiemannianWrapper."""
    import mani_skill.envs  # noqa: F401 — registers ManiSkill envs
    from hires_vic.envs.maniskill_riemannian import ManiSkillRiemannianWrapper

    n = 1 if is_eval else args.n_envs
    # ManiSkill 3 supports GPU-parallel simulation natively via num_envs
    # Use CPU backend for eval to avoid: "GPU PhysX can only be enabled once"
    sim_backend = "cpu" if is_eval else args.sim_backend
    print(f"Creating {'evaluation' if is_eval else 'training'} envs with sim_backend='{sim_backend}'...")
    env = gym.make(
        args.env,
        num_envs=n,
        obs_mode="state",
        sim_backend=sim_backend,
        max_episode_steps=args.max_episode_steps,
        render_mode=None,
    )

    env = InsertionCurriculumWrapper(env, setup_steps=90)

    # PiH task metrics: insertion depth from env info (populated by ManiSkill)
    def pih_task_metrics_fn(env, info):
        metrics = {}
        try:
            raw = env.unwrapped.get_obs()
            extra = raw.get("extra", {}) if isinstance(raw, dict) else {}
            if "insertion_depth" in extra:
                metrics["physics/insertion_depth"] = float(extra["insertion_depth"].mean())
        except Exception:
            pass
        if info.get("success") is not None:
            success = info["success"]
            # Handle batched success (array/tensor) by taking mean
            if hasattr(success, "mean"):
                success = success.float().mean()
            metrics["physics/pih_success"] = float(success)
        return metrics

    env = ManiSkillRiemannianWrapper(
        env,
        use_spd=args.use_spd,
        use_lie_group=args.use_lie,
        use_diag=args.use_diag,
        use_fixed=args.use_fixed,
        is_eval=is_eval,
        use_llm_prior=args.use_llm_prior,
        llm_backend=args.llm_backend,
        llm_model=args.llm_model,
        llm_query_interval=args.llm_query_interval,
        llm_prior_weight=args.llm_prior_weight,
        llm_profile_path=args.llm_profile if args.use_llm_prior else None,
        use_sim2real_obs=args.use_sim2real_obs,
        task_metrics_fn=pih_task_metrics_fn,
    )

    # RecordEpisodeStatistics doesn't support batched envs; only apply for single-env
    if n == 1:
        env = gym.wrappers.RecordEpisodeStatistics(env)
    return env


# ── WandB metric logging helper ───────────────────────────────────────────────

def log_episode_metrics(infos, global_step: int, prefix: str = "train"):
    """Extract episode-end metrics from info dicts and log to wandb."""
    metric_keys = [
        "physics/kp_trans_x_avg", "physics/kp_trans_y_avg", "physics/kp_trans_z_avg",
        "physics/kp_rot_x_avg",   "physics/kp_rot_y_avg",   "physics/kp_rot_z_avg",
        "physics/contact_step_ratio",
        "smoothness/avg_cond_num", "smoothness/avg_riemannian_jerk",
        "smoothness/avg_euclidean_jerk", "smoothness/avg_coupling_magnitude",
        "smoothness/avg_force",    "smoothness/std_force",
        "physics/insertion_depth", "physics/pih_success",
        "llm/impedance_mode",
    ]
    log_dict = {}
    # Handle both list-of-dicts (SB3) and dict-of-arrays (ManiSkill)
    info_list = infos if isinstance(infos, (list, tuple)) else [infos]
    for info in info_list:
        for key in metric_keys:
            if key in info:
                if isinstance(info[key], str):
                    # print(f"Logging string metric '{key}': {info[key]}")
                    continue
                log_dict[f"{prefix}/{key}"] = float(
                    info[key].mean() if hasattr(info[key], "mean") else info[key]
                )
    if log_dict:
        wandb.log(log_dict, step=global_step)


# ── SAC training loop ─────────────────────────────────────────────────────────

def train_sac(args, device: torch.device, video_env=None):
    envs      = make_envs(args)
    eval_envs = make_envs(args, is_eval=True)

    obs_dim    = int(np.prod(envs.observation_space.shape))
    action_dim = int(np.prod(envs.action_space.shape))

    actor    = SACActorNetwork(obs_dim, action_dim).to(device)
    qf1      = SoftQNetwork(obs_dim, action_dim).to(device)
    qf2      = SoftQNetwork(obs_dim, action_dim).to(device)
    qf1_tgt  = SoftQNetwork(obs_dim, action_dim).to(device)
    qf2_tgt  = SoftQNetwork(obs_dim, action_dim).to(device)
    qf1_tgt.load_state_dict(qf1.state_dict())
    qf2_tgt.load_state_dict(qf2.state_dict())

    q_opt = optim.Adam(list(qf1.parameters()) + list(qf2.parameters()), lr=args.learning_rate)
    a_opt = optim.Adam(actor.parameters(), lr=args.learning_rate)

    # Automatic entropy tuning (mirrors SB3 target_entropy="auto")
    if args.autotune_alpha:
        target_entropy = -action_dim  # = -|A|
        log_alpha = torch.zeros(1, requires_grad=True, device=device)
        alpha = log_alpha.exp().item()
        alpha_opt = optim.Adam([log_alpha], lr=args.learning_rate)
    else:
        alpha = args.alpha

    rb = ReplayBuffer(args.buffer_size, obs_dim, action_dim, device)

    obs, _ = envs.reset(seed=args.seed)
    obs = _to_tensor(obs, device)

    global_step = 0
    ep_returns  = deque(maxlen=100)
    ep_lengths  = deque(maxlen=100)
    start_time  = time.time()

    while global_step < args.total_timesteps:
        # ── Collect experience ────────────────────────────────────────────────
        if global_step < args.learning_starts:
            # actions = np.array(
            #     [envs.action_space.sample() for _ in range(args.n_envs)]
            # )
            actions = torch.rand((args.n_envs, action_dim), dtype=torch.float32, device=device) * 2.0 - 1.0
        else:
            with torch.no_grad():
                actions, _, _ = actor.get_action(obs)
            # actions = actions.cpu().numpy()

        next_obs, rewards, terminated, truncated, infos = envs.step(actions)
        if global_step == args.learning_starts:
            print("\n" + "="*55)
            print("🚀 ZERO-COPY GPU PHYSICS VERIFICATION 🚀")
            print(f"1. Action Tensor Device:    {actions.device}")
            print(f"2. Raw Obs Output Type:     {type(next_obs)}")
            if isinstance(next_obs, torch.Tensor):
                print(f"3. Raw Obs Output Device:   {next_obs.device}")
            if isinstance(rewards, torch.Tensor):
                print(f"4. Reward Tensor Device:    {rewards.device}")
            print("="*55 + "\n")
        dones = torch.logical_or(terminated, truncated).float()
        # next_obs_t = _to_tensor(next_obs, device)

        rb.add(obs, next_obs, actions, rewards, dones)
        obs = next_obs
        # obs = next_obs
        global_step += args.n_envs

        # Episode stats
        if "episode" in infos:
            ep_info = infos["episode"]
            _r = _to_numpy(ep_info["r"])
            _l = _to_numpy(ep_info["l"])
            ep_returns.extend(_r.tolist())
            ep_lengths.extend(_l.tolist())

        log_episode_metrics(infos, global_step)

        # ── Update networks ───────────────────────────────────────────────────
        if global_step < args.learning_starts or global_step % args.train_freq != 0:
            continue

        for _ in range(args.gradient_steps):
            obs_b, next_obs_b, act_b, rew_b, done_b = rb.sample(args.batch_size)

            # Critic update
            with torch.no_grad():
                next_a, next_log_pi, _ = actor.get_action(next_obs_b)
                q1_next = qf1_tgt(next_obs_b, next_a)
                q2_next = qf2_tgt(next_obs_b, next_a)
                min_q_next = torch.min(q1_next, q2_next) - alpha * next_log_pi
                target_q = rew_b + (1 - done_b) * args.gamma * min_q_next

            qf1_loss = F.mse_loss(qf1(obs_b, act_b), target_q)
            qf2_loss = F.mse_loss(qf2(obs_b, act_b), target_q)
            qf_loss  = qf1_loss + qf2_loss

            q_opt.zero_grad()
            qf_loss.backward()
            q_opt.step()

            # Actor update
            pi, log_pi, _ = actor.get_action(obs_b)
            min_q_pi = torch.min(qf1(obs_b, pi), qf2(obs_b, pi))
            actor_loss = (alpha * log_pi - min_q_pi).mean()

            a_opt.zero_grad()
            actor_loss.backward()
            a_opt.step()

            # Alpha update
            if args.autotune_alpha:
                with torch.no_grad():
                    _, log_pi, _ = actor.get_action(obs_b)
                alpha_loss = (-log_alpha.exp() * (log_pi + target_entropy)).mean()
                alpha_opt.zero_grad()
                alpha_loss.backward()
                alpha_opt.step()
                alpha = log_alpha.exp().item()

            # Soft target update (τ = args.tau)
            for p, tp in zip(qf1.parameters(), qf1_tgt.parameters()):
                tp.data.copy_(args.tau * p.data + (1 - args.tau) * tp.data)
            for p, tp in zip(qf2.parameters(), qf2_tgt.parameters()):
                tp.data.copy_(args.tau * p.data + (1 - args.tau) * tp.data)

        # ── Periodic logging ──────────────────────────────────────────────────
        if global_step % args.log_freq == 0 and ep_returns:
            sps = int(global_step / (time.time() - start_time))
            wandb.log({
                "train/mean_episode_return": np.mean(ep_returns),
                "train/mean_episode_length": np.mean(ep_lengths),
                "train/actor_loss":  actor_loss.item(),
                "train/critic_loss": qf_loss.item(),
                "train/alpha":       alpha,
                "charts/SPS":        sps,
            }, step=global_step)

        # ── Evaluation ───────────────────────────────────────────────────────
        if global_step % args.eval_freq < args.n_envs:
            _run_eval(actor, eval_envs, device, global_step, args.n_eval_episodes)
            if video_env is not None:
                record_video_probe(actor, video_env, device, global_step, fps=args.video_fps)

    # Final save
    save_dir = Path("outputs/models")
    save_dir.mkdir(parents=True, exist_ok=True)
    torch.save(actor.state_dict(), save_dir / f"{args.run_name}_actor_final.pt")
    envs.close()
    eval_envs.close()


# ── PPO training loop ─────────────────────────────────────────────────────────

def train_ppo(args, device: torch.device, video_env=None):
    envs      = make_envs(args)
    eval_envs = make_envs(args, is_eval=True)

    obs_dim    = int(np.prod(envs.observation_space.shape))
    action_dim = int(np.prod(envs.action_space.shape))

    # SB3 PPO uses separate actor/critic networks — no shared trunk (default)
    actor  = PPOActorNetwork(obs_dim, action_dim).to(device)
    critic = PPOCriticNetwork(obs_dim).to(device)
    opt    = optim.Adam(
        list(actor.parameters()) + list(critic.parameters()),
        lr=args.learning_rate, eps=1e-5,
    )

    # Rollout storage
    n_steps  = args.n_steps
    n_envs   = args.n_envs
    rollout_obs     = torch.zeros((n_steps, n_envs, obs_dim),    device=device)
    rollout_actions = torch.zeros((n_steps, n_envs, action_dim), device=device)
    rollout_logprobs= torch.zeros((n_steps, n_envs),             device=device)
    rollout_rewards = torch.zeros((n_steps, n_envs),             device=device)
    rollout_dones   = torch.zeros((n_steps, n_envs),             device=device)
    rollout_values  = torch.zeros((n_steps, n_envs),             device=device)

    obs, _ = envs.reset(seed=args.seed)
    obs = _to_tensor(obs, device)
    done = torch.zeros(n_envs, device=device)

    global_step   = 0
    ep_returns    = deque(maxlen=100)
    ep_lengths    = deque(maxlen=100)
    start_time    = time.time()
    num_updates   = args.total_timesteps // (n_steps * n_envs)

    for update in range(1, num_updates + 1):
        # LR annealing (SB3 PPO default)
        frac = 1.0 - (update - 1) / num_updates
        opt.param_groups[0]["lr"] = frac * args.learning_rate

        # ── Collect rollout ───────────────────────────────────────────────────
        for step in range(n_steps):
            global_step += n_envs
            rollout_obs[step]   = obs
            rollout_dones[step] = done

            with torch.no_grad():
                action, logprob, _ = actor.get_action_and_log_prob(obs)
                value = critic(obs)
                rollout_values[step] = value.squeeze(-1)

            rollout_actions[step]  = action
            rollout_logprobs[step] = logprob

            next_obs, reward, terminated, truncated, infos = envs.step(action.cpu().numpy())
            done = _to_tensor(
                np.logical_or(_to_numpy(terminated), _to_numpy(truncated)).astype(np.float32),
                device,
            )
            rollout_rewards[step] = _to_tensor(reward, device).flatten()
            obs = _to_tensor(next_obs, device)

            if "episode" in infos:
                ep_info = infos["episode"]
                ep_returns.extend(_to_numpy(ep_info["r"]).tolist())
                ep_lengths.extend(_to_numpy(ep_info["l"]).tolist())
            log_episode_metrics(infos, global_step)

        # ── GAE advantages ────────────────────────────────────────────────────
        with torch.no_grad():
            next_value = critic(obs).squeeze(-1)

        advantages = torch.zeros_like(rollout_rewards, device=device)
        last_gae   = 0.0
        lam        = 0.95
        for t in reversed(range(n_steps)):
            nxt_val  = next_value if t == n_steps - 1 else rollout_values[t + 1]
            nxt_done = done       if t == n_steps - 1 else rollout_dones[t + 1]
            delta    = rollout_rewards[t] + args.gamma * nxt_val * (1 - nxt_done) - rollout_values[t]
            last_gae = delta + args.gamma * lam * (1 - nxt_done) * last_gae
            advantages[t] = last_gae
        returns = advantages + rollout_values

        # ── Policy update ─────────────────────────────────────────────────────
        b_obs      = rollout_obs.reshape(-1, obs_dim)
        b_actions  = rollout_actions.reshape(-1, action_dim)
        b_logprobs = rollout_logprobs.reshape(-1)
        b_adv      = advantages.reshape(-1)
        b_ret      = returns.reshape(-1)
        b_adv      = (b_adv - b_adv.mean()) / (b_adv.std() + 1e-8)

        idxs = np.arange(n_steps * n_envs)
        for _ in range(args.n_epochs):
            np.random.shuffle(idxs)
            for start in range(0, len(idxs), args.batch_size):
                mb   = idxs[start: start + args.batch_size]
                mb_t = torch.tensor(mb, device=device)

                _, new_logprob, entropy = actor.get_action_and_log_prob(
                    b_obs[mb_t], b_actions[mb_t]
                )
                new_value = critic(b_obs[mb_t])
                ratio  = (new_logprob - b_logprobs[mb_t]).exp()
                mb_adv = b_adv[mb_t]

                pg_loss1 = -mb_adv * ratio
                pg_loss2 = -mb_adv * torch.clamp(ratio, 1 - args.clip_range, 1 + args.clip_range)
                pg_loss  = torch.max(pg_loss1, pg_loss2).mean()

                vf_loss  = F.mse_loss(new_value.squeeze(-1), b_ret[mb_t])
                ent_loss = entropy.mean()

                loss = pg_loss + args.vf_coef * vf_loss - args.ent_coef * ent_loss

                opt.zero_grad()
                loss.backward()
                nn.utils.clip_grad_norm_(
                    list(actor.parameters()) + list(critic.parameters()),
                    args.max_grad_norm,
                )
                opt.step()

        if update % max(1, (args.log_freq // (n_steps * n_envs))) == 0 and ep_returns:
            sps = int(global_step / (time.time() - start_time))
            wandb.log({
                "train/mean_episode_return": np.mean(ep_returns),
                "train/mean_episode_length": np.mean(ep_lengths),
                "train/policy_loss": pg_loss.item(),
                "train/value_loss":  vf_loss.item(),
                "charts/SPS":        sps,
            }, step=global_step)

        if global_step % args.eval_freq < n_steps * n_envs:
            _run_eval((actor, critic), eval_envs, device, global_step,
                      args.n_eval_episodes, is_ppo=True)
            if video_env is not None:
                record_video_probe((actor, critic), video_env, device, global_step,
                                   is_ppo=True, fps=args.video_fps)

    save_dir = Path("outputs/models")
    save_dir.mkdir(parents=True, exist_ok=True)
    torch.save(actor.state_dict(),  save_dir / f"{args.run_name}_ppo_actor_final.pt")
    torch.save(critic.state_dict(), save_dir / f"{args.run_name}_ppo_critic_final.pt")
    envs.close()
    eval_envs.close()


# ── Video probes ──────────────────────────────────────────────────────────────

def make_video_env(args):
    """Single CPU env with render_mode='rgb_array' for video probes."""
    import mani_skill.envs  # noqa: F401
    from hires_vic.envs.maniskill_riemannian import ManiSkillRiemannianWrapper

    env = gym.make(
        args.env,
        num_envs=1,
        obs_mode="state",
        sim_backend="cpu",
        render_mode="rgb_array",
        max_episode_steps=args.max_episode_steps,
    )
    env = ManiSkillRiemannianWrapper(
        env,
        use_spd=args.use_spd,
        use_lie_group=args.use_lie,
        use_diag=args.use_diag,
        use_fixed=args.use_fixed,
        is_eval=False,
        use_llm_prior=False,
        use_sim2real_obs=args.use_sim2real_obs,
        task_metrics_fn=None,
    )
    return env


def record_video_probe(policy, video_env, device, global_step, is_ppo=False, fps=20):
    """Run one episode in video_env, collect RGB frames, upload to wandb."""
    if is_ppo:
        ppo_actor, _ = policy

    obs, _ = video_env.reset()
    obs = _to_tensor(obs, device)
    frames = []

    while True:
        frame = video_env.render()
        if isinstance(frame, torch.Tensor):
            frame = frame.cpu().numpy()
        frame = np.asarray(frame, dtype=np.uint8)
        if frame.ndim == 4:   # (1, H, W, C) → (H, W, C)
            frame = frame[0]
        frames.append(frame)

        with torch.no_grad():
            if is_ppo:
                action, _, _ = ppo_actor.get_action_and_log_prob(obs)
            else:
                action, _, _ = policy.get_action(obs)

        # obs, _, terminated, truncated, _ = video_env.step(action.cpu().numpy())
        obs, _, terminated, truncated, _ = video_env.step(action.cpu())
        obs = _to_tensor(obs, device)
        if _to_numpy(terminated).any() or _to_numpy(truncated).any():
            break

    if frames:
        video_array = np.stack(frames, axis=0)  # (T, H, W, C)
        video_array = np.transpose(video_array, (0, 3, 1, 2))
        wandb.log(
            {"eval/video": wandb.Video(video_array, fps=fps, format="mp4")},
            step=global_step,
        )


# ── Evaluation ────────────────────────────────────────────────────────────────

def _run_eval(policy, eval_envs, device, global_step, n_episodes, is_ppo=False):
    """
    policy: SACActorNetwork for SAC, or (PPOActorNetwork, PPOCriticNetwork) tuple for PPO.
    """
    if is_ppo:
        ppo_actor, _ = policy
    obs, _ = eval_envs.reset()
    obs = _to_tensor(obs, device)

    ep_returns, ep_successes = [], []
    kp_accs = []

    for _ in range(n_episodes):
        ep_ret = 0.0
        while True:
            with torch.no_grad():
                if is_ppo:
                    action, _, _ = ppo_actor.get_action_and_log_prob(obs)
                else:
                    action, _, _ = policy.get_action(obs)
            # obs, rew, terminated, truncated, info = eval_envs.step(action.cpu().numpy())
            obs, rew, terminated, truncated, info = eval_envs.step(action.cpu())
            obs = _to_tensor(obs, device)
            ep_ret += float(_to_numpy(rew).mean())
            if _to_numpy(terminated).any() or _to_numpy(truncated).any():
                ep_returns.append(ep_ret)
                if "success" in info:
                    ep_successes.append(float(_to_numpy(info["success"]).mean()))
                if hasattr(eval_envs, "kp_history") and eval_envs.kp_history:
                    kp_accs.append(np.mean(eval_envs.kp_history, axis=0))
                obs, _ = eval_envs.reset()
                obs = _to_tensor(obs, device)
                break

    log = {
        "eval/mean_return":    np.mean(ep_returns) if ep_returns else 0.0,
        "eval/success_rate":   np.mean(ep_successes) if ep_successes else 0.0,
    }
    if kp_accs:
        kp_mean = np.mean(kp_accs, axis=0)
        for i, ax in enumerate(["x", "y", "z"]):
            log[f"eval/kp_trans_{ax}_avg"] = kp_mean[i]
            log[f"eval/kp_rot_{ax}_avg"]   = kp_mean[i + 3]
    wandb.log(log, step=global_step)
    print(f"[eval @{global_step}] return={log['eval/mean_return']:.2f} "
          f"success={log['eval/success_rate']:.2%}")


# ── Utilities ─────────────────────────────────────────────────────────────────

def _to_tensor(x, device):
    if isinstance(x, torch.Tensor):
        return x.float().to(device)
    return torch.tensor(np.asarray(x), dtype=torch.float32, device=device)


def _to_numpy(x):
    if isinstance(x, torch.Tensor):
        return x.cpu().numpy()
    return np.asarray(x)


# ── Entry point ───────────────────────────────────────────────────────────────

def main():
    args = parse_args()

     # Enable GPU PhysX once before creating any ManiSkill environments
    if args.sim_backend == "gpu":
        try:
            import sapien.physx
            sapien.physx.enable_gpu()
            print("GPU PhysX enabled for ManiSkill simulation.")
        except RuntimeError:
            print("Warning: GPU PhysX unavailable; falling back to CPU simulation.")
            pass  # Already enabled, or GPU unavailable

    random.seed(args.seed)
    np.random.seed(args.seed)
    torch.manual_seed(args.seed)

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Device: {device} | Algorithm: {args.algorithm} | Envs: {args.n_envs}")

    alg_tag    = args.algorithm.upper()
    geom_tag   = ("SPD" if args.use_spd else "DIAG" if args.use_diag else "FIXED" if args.use_fixed else "VIC")
    run_name   = f"{alg_tag}_{args.env}_{geom_tag}_{args.run_name}_seed{args.seed}"

    wandb.init(
        project="HiRes-VIC",
        name=run_name,
        config=vars(args),
        sync_tensorboard=False,
    )

    if args.algorithm.upper() == "PPO":
        video_env = make_video_env(args) if args.record_video else None
        train_ppo(args, device, video_env=video_env)
    else:
        video_env = make_video_env(args) if args.record_video else None
        train_sac(args, device, video_env=video_env)
    if args.record_video and video_env is not None:
        video_env.close()

    wandb.finish()
    print("Training complete.")


if __name__ == "__main__":
    main()
