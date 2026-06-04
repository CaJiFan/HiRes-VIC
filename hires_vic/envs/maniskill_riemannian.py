"""
ManiSkill Riemannian Variable Impedance Controller.

Implements SPD (Sym+(3)) stiffness learning for ManiSkill environments as a
gym.Wrapper — analogous to RiemannianController for Robosuite but without deep
controller inheritance.

Riemannian control approximation
─────────────────────────────────
For a delta-pose controller (arm_pd_ee_delta_pose), the native PD law is:
    F = diag(Kp_base) * delta_pos   (isotropic, element-wise)

We achieve anisotropic Riemannian VIC by pre-multiplying the delta:
    delta_pos_eff = (Kp_matrix / λ_max(Kp_matrix)) @ delta_pos_raw

so that the native controller effectively computes:
    F ∝ Kp_matrix @ delta_pos_raw

where Kp_matrix ∈ Sym+(3) is learned on the SPD manifold via spd_grl_map.
λ_max normalisation keeps the effective delta within the same scale as the raw
delta, ensuring the RL agent's scale assumptions remain valid.

Full Riemannian VIC (F = Kp @ error directly in task space) would require a
custom ManiSkill ArticulationController. This wrapper is the dev-stage
approximation; full controller integration is planned for the camera-ready.

Sim2Real observation space
──────────────────────────
When `use_sim2real_obs=True` (default), the wrapper overrides the environment's
native observation with a filtered vector containing only signals that are
available on a real Franka robot:

  arm_qpos    (7)  ← Franka joint encoders
  arm_qvel    (7)  ← Franka joint velocity (from encoders)
  eef_pos     (3)  ← FK from joint encoders
  eef_ori  (3|4)  ← SO(3) log-map (use_lie=True) or raw quaternion [w,x,y,z]
  gripper_qpos (2) ← Gripper encoder (finger widths)
  tcp_force   (3)  ← Franka built-in wrist F/T sensor (force)
  tcp_torque  (3)  ← Franka built-in wrist F/T sensor (torque)

Total: 28D (with SO(3)) or 29D (with quaternion).

Privileged task state (peg/hole positions) is EXCLUDED — the policy must infer
task progress from F/T patterns and EEF motion alone, exactly as it would on
the physical robot.

Action layouts (RL policy output → wrapper input)
──────────────────────────────────────────────────
SPD mode (use_spd=True):
  [m11,m22,m33,m23,m13,m12, r1,r2,r3, dx,dy,dz, drx,dry,drz, gripper]
  [----  6D Mandel (SPD) ----][3D rot][-- 3D pos --][-- 3D ori --][  1 ]
  = native_dim + 9  (e.g. 7 + 9 = 16 for Panda + gripper)

Diagonal mode (use_diag=True):
  [kpx,kpy,kpz, rx,ry,rz, dx,dy,dz, drx,dry,drz, gripper]
  [-- 6D diag stiffness  ][-- 3D pos --][-- 3D ori --][  1 ]
  = native_dim + 6

Baseline VIC (use_spd=False, use_diag=False, use_fixed=False) — DEFAULT:
  [kpx,kpy,kpz, rx,ry,rz, dx,dy,dz, drx,dry,drz, gripper]
  Linear Kp mapping to [min_kp, max_kp] — matches Robosuite OSC variable_kp convention.
  = native_dim + 6

Fixed (use_fixed=True):
  Passthrough — wrapper is a no-op on the action. Kp frozen in controller config.
"""

from __future__ import annotations

import math
import numpy as np
import scipy.linalg as spla
import torch
import gymnasium as gym
from gymnasium import spaces

from hires_vic.geometry.riemannian import spd_grl_map
from hires_vic.geometry.lie_groups import so3_log_map


class ManiSkillRiemannianWrapper(gym.Wrapper):
    """
    Wraps any ManiSkill (or generic gymnasium) env to add:
      - SPD manifold action parameterisation (full 3×3 or diagonal)
      - Riemannian direction-dependent delta_pos scaling
      - Physics / geometry metric logging (cond_num, Riemannian jerk, coupling)
      - Sim2Real observation filtering (F/T + proprioception only, no privileged state)
      - PiH task metric logging (insertion depth, alignment error)
      - LLM impedance planner integration (optional)

    Parameters
    ----------
    env              : ManiSkill gymnasium environment (already gym-wrapped)
    use_spd          : Full 3×3 SPD manifold (Mandel basis, 6 extra action dims)
    use_lie_group    : SO(3) log-map for EEF orientation in observations
    use_diag         : Diagonal SPD (log-space scalar per axis, 6 extra dims)
    use_fixed        : Fixed impedance — pure passthrough, Kp frozen in controller config
    use_sim2real_obs : Filter obs to only real-robot-compatible signals (default True)
    is_eval          : If True, accumulate kp history for evaluation logging
    use_llm_prior    : Enable LLM impedance planner blending
    llm_backend      : "openai" | "ollama"
    llm_model        : Model name override (None = backend default)
    llm_query_interval : Steps between LLM queries
    llm_prior_weight   : Blend weight w (action = (1-w)*RL + w*prior)
    llm_profile_path   : Path to YAML impedance profile for this task
    task_metrics_fn    : Optional callable(env, info) -> dict for custom metrics
    """

    # Franka robot DOF breakdown (PegInsertionSide-v1 default)
    _ARM_JOINTS   = 7
    _GRIPPER_DOFS = 2  # 2 finger joints

    def __init__(
        self,
        env,
        use_spd=False,
        use_lie_group=False,
        use_diag=False,
        use_fixed=False,
        use_sim2real_obs=True,
        is_eval=False,
        use_llm_prior=False,
        llm_backend="openai",
        llm_model=None,
        llm_query_interval=50,
        llm_prior_weight=0.4,
        llm_profile_path=None,
        task_metrics_fn=None,
    ):
        super().__init__(env)

        self.use_spd          = use_spd
        self.use_lie_group    = use_lie_group
        self.use_diag         = use_diag
        self.use_fixed        = use_fixed
        self.use_sim2real_obs = use_sim2real_obs
        self.is_eval          = is_eval
        self.task_metrics_fn  = task_metrics_fn
        # Baseline VIC: no geometry flags and not fixed — linear diagonal Kp, matches Robosuite default
        self._use_variable_kp = not (use_spd or use_diag or use_fixed)

        self.dt = 1 / 20  # 20 Hz control
        self.min_kp, self.max_kp = 1.0, 300.0

        # ── Action space ─────────────────────────────────────────────────────
        # Handle both single env (shape=(8,)) and vectorized envs (shape=(n, 8))
        native_dim = env.action_space.shape[-1]
        if self.use_spd:
            self._extra_dims  = 9  # 6 Mandel SPD + 3 rot
            self._mandel_slice = slice(0, 6)
            self._rot_slice    = slice(6, 9)
            self._pose_slice   = slice(9, 9 + native_dim)
        elif self.use_diag:
            self._extra_dims  = 6  # 3 trans_kp + 3 rot_kp (log-scale diagonal SPD ablation)
            self._trans_slice  = slice(0, 3)
            self._rot_slice    = slice(3, 6)
            self._pose_slice   = slice(6, 6 + native_dim)
        elif self._use_variable_kp:
            self._extra_dims  = 6  # 3 trans_kp + 3 rot_kp (linear-scale baseline VIC)
            self._trans_slice  = slice(0, 3)
            self._rot_slice    = slice(3, 6)
            self._pose_slice   = slice(6, 6 + native_dim)
        else:  # use_fixed
            self._extra_dims  = 0
            self._pose_slice   = slice(0, native_dim)

        total_action_dim = native_dim + self._extra_dims
        self.action_space = spaces.Box(
            low=-1.0, high=1.0, shape=(total_action_dim,), dtype=np.float32
        )

        # ── Observation space ────────────────────────────────────────────────
        if self.use_sim2real_obs:
            # Only real-robot-compatible signals (no privileged task state)
            # arm_qpos(7) + arm_qvel(7) + eef_pos(3) + eef_ori(3|4) + gripper(2) + force(3) + torque(3)
            ori_dim = 3 if use_lie_group else 4
            self._sim2real_obs_dim = (
                self._ARM_JOINTS      # arm joint positions
                + self._ARM_JOINTS    # arm joint velocities
                + 3                   # eef position
                + ori_dim             # eef orientation (SO3 log or quat)
                + self._GRIPPER_DOFS  # gripper finger positions
                + 3                   # tcp force (wrist F/T)
                + 3                   # tcp torque (wrist F/T)
            )
            self.observation_space = spaces.Box(
                low=-np.inf, high=np.inf,
                shape=(self._sim2real_obs_dim,), dtype=np.float32
            )
        else:
            self.observation_space = env.observation_space

        # ── Metric histories (reset each episode) ────────────────────────────
        self._reset_histories()

        # ── LLM planner ──────────────────────────────────────────────────────
        self.use_llm_prior = use_llm_prior
        self.llm_planner   = None
        if use_llm_prior:
            from hires_vic.llm.impedance_planner import LLMImpedancePlanner
            self.llm_planner = LLMImpedancePlanner(
                backend=llm_backend,
                model=llm_model,
                query_every_n_steps=llm_query_interval,
                prior_weight=llm_prior_weight,
                profile_path=llm_profile_path,
                use_spd_manifold=self.use_spd
            )
        # self._current_llm_mode = "approach"
        self._current_llm_mode = "align"

        print(
            f"ManiSkillRiemannianWrapper | SPD={use_spd} Diag={use_diag} "
            f"VIC={self._use_variable_kp} Fixed={use_fixed} "
            f"LieGroup={use_lie_group} Sim2Real={use_sim2real_obs} "
            f"| action {native_dim}D→{total_action_dim}D "
            f"| obs {self.observation_space.shape[0]}D"
        )

        if self.is_eval:
            self.kp_history = []

    # ── Sim2Real observation extraction ──────────────────────────────────────

    @staticmethod
    def _quat_wxyz_to_so3_log(q: np.ndarray) -> np.ndarray:
        """
        Quaternion [w, x, y, z] (SAPIEN3 convention) → so(3) log-map vector.
        Input shape: (..., 4). Output shape: (..., 3).
        """
        single = q.ndim == 1
        if single:
            q = q[np.newaxis]

        # q_t = torch.tensor(q, dtype=torch.float32)
        # w, x, y, z = q_t[:, 0], q_t[:, 1], q_t[:, 2], q_t[:, 3]
        # B = q_t.shape[0]
        w, x, y, z = q[:, 0], q[:, 1], q[:, 2], q[:, 3]
        B = q.shape[0]

        # Build rotation matrix from quaternion (standard formula)
        R = torch.stack([
            1 - 2*(y*y + z*z),  2*(x*y - z*w),      2*(x*z + y*w),
            2*(x*y + z*w),      1 - 2*(x*x + z*z),  2*(y*z - x*w),
            2*(x*z - y*w),      2*(y*z + x*w),      1 - 2*(x*x + y*y),
        ], dim=1).reshape(B, 3, 3)

        # omega = so3_log_map(R).detach().numpy()  # (B, 3)
        omega = so3_log_map(R)
        return omega[0] if single else omega

    def _extract_sim2real_obs(self) -> np.ndarray:
        """
        Reads only real-robot-compatible state from ManiSkill sim:
          arm_qpos (7), arm_qvel (7), eef_pos (3), eef_ori (3|4),
          gripper_qpos (2), tcp_force (3), tcp_torque (3).

        Works with both CPU and GPU (CUDA tensor) sim backends.
        Returns shape (n_envs, obs_dim) for vectorised envs or (obs_dim,) for single.
        """
        env = self.env.unwrapped

        # def _np(t):
        #     if isinstance(t, torch.Tensor):
        #         return t.float().cpu().numpy()
        #     return np.asarray(t, dtype=np.float32)

        try:
            # Joint state from Franka encoders
            qpos_full = env.agent.robot.get_qpos()   # (..., 9)
            qvel_full = env.agent.robot.get_qvel()   # (..., 9)

            arm_qpos     = qpos_full[..., :self._ARM_JOINTS]    # (..., 7)
            gripper_qpos = qpos_full[..., self._ARM_JOINTS:]    # (..., 2)
            arm_qvel     = qvel_full[..., :self._ARM_JOINTS]    # (..., 7)

            # EEF pose via FK (identical result to joint encoder FK on real robot)
            tcp_p = env.agent.tcp.pose.p   # (..., 3)
            tcp_q = env.agent.tcp.pose.q   # (..., 4) [w, x, y, z]

            if self.use_lie_group:
                eef_ori = self._quat_wxyz_to_so3_log(tcp_q)  # (..., 3)
            else:
                eef_ori = tcp_q                               # (..., 4)

            # Wrist F/T sensor (Franka FRI / FT sensor via SAPIEN contact forces)
            try:
                tcp_force = env.agent.tcp.get_net_contact_forces()  # (..., 3)
            except Exception:
                shape = arm_qpos.shape[:-1] + (3,)
                # tcp_force = np.zeros(shape, dtype=np.float32)
                tcp_force = torch.zeros(shape, dtype=torch.float32, device=arm_qpos.device)

            # Torque: attempt to get from contact sensor; fallback to zeros
            try:
                # ManiSkill 3 exposes net contact torques on some versions
                tcp_torque = env.agent.tcp.get_net_contact_torques()
            except Exception:
                # tcp_torque = np.zeros_like(tcp_force)
                tcp_torque = torch.zeros_like(tcp_force)

        except Exception as e:
            # Robust fallback if env API differs — returns safe zero obs
            print(f"[ManiSkillRiemannianWrapper] Obs extraction failed: {e}. "
                  "Returning zero obs.")
            n = getattr(env, "num_envs", 1)
            ori_dim = 3 if self.use_lie_group else 4
            return np.zeros((n, self._sim2real_obs_dim), dtype=np.float32)

        parts = [arm_qpos, arm_qvel, tcp_p, eef_ori, gripper_qpos,
                 tcp_force, tcp_torque]
        # obs = np.concatenate(parts, axis=-1).astype(np.float32)
        obs = torch.cat(parts, dim=-1).float()

        # Squeeze batch dim for single-env case (n_envs=1 → (obs_dim,))
        # if obs.ndim == 2 and obs.shape[0] == 1 and not hasattr(env, "num_envs"):
        #     obs = obs[0]

        return obs

    # ── Private helpers ───────────────────────────────────────────────────────

    def _reset_histories(self):
        self._prev_Kp              = np.eye(3)
        self._episode_steps        = 0
        self._episode_contact_steps= 0
        self._ep_kp                = np.zeros(6)
        self.cond_num_history      = []
        self.euclidean_jerk_history= []
        self.riemannian_jerk_history=[]
        self.coupling_history      = []
        self.force_history         = []

    def _spd_from_action(self, action_flat):
        """Parse action → (Kp_matrix 3×3, kp_rot 3D, native_action, physical_kp_vals).
        Handles both unbatched (action_dim,) and batched (n_envs, action_dim) actions."""
        batched = action_flat.ndim > 1
        if batched:
            action_first = action_flat[0]  # Extract first env for impedance extraction
        else:
            action_first = action_flat

        if self.use_spd:
            # mandel = action_first[self._mandel_slice].copy()
            mandel = action_first[self._mandel_slice].clone()
            log_min, log_max = math.log(self.min_kp), math.log(self.max_kp)
            mandel[0:3] = log_min + 0.5 * (mandel[0:3] + 1.0) * (log_max - log_min)
            mandel[3:6] = mandel[3:6] * 0.2  # off-diagonal scale

            kp_matrix = spd_grl_map(mandel.unsqueeze(0)).squeeze(0)

            kp_rot = self.min_kp + 0.5 * (action_first[self._rot_slice] + 1.0) * (self.max_kp - self.min_kp)
            native = action_flat[:, self._pose_slice] if batched else action_first[self._pose_slice]
            # physical_kp_vals = np.concatenate([np.diag(kp_matrix), kp_rot])
            physical_kp_vals = torch.cat([torch.diag(kp_matrix), kp_rot], dim=0)

        elif self.use_diag:
            # Diagonal SPD ablation: log-scale mapping (same exponential map as SPD, no off-diagonal)
            kp_raw = action_first[self._trans_slice].clone()
            log_min, log_max = math.log(self.min_kp), math.log(self.max_kp)
            kp_diag = torch.exp(log_min + 0.5 * (kp_raw + 1.0) * (log_max - log_min))
            kp_matrix = torch.diag(kp_diag)
            kp_rot = self.min_kp + 0.5 * (action_first[self._rot_slice] + 1.0) * (self.max_kp - self.min_kp)
            native = action_flat[:, self._pose_slice] if batched else action_first[self._pose_slice]
            physical_kp_vals = torch.cat([kp_diag, kp_rot], dim=0)

        elif self._use_variable_kp:
            # Baseline VIC: linear mapping — mirrors Robosuite OSC variable_kp convention
            # kp_raw = action_first[self._trans_slice].copy()
            kp_raw = action_first[self._trans_slice].clone()
            kp_diag = self.min_kp + 0.5 * (kp_raw + 1.0) * (self.max_kp - self.min_kp)
            kp_matrix = torch.diag(kp_diag)
            kp_rot = self.min_kp + 0.5 * (action_first[self._rot_slice] + 1.0) * (self.max_kp - self.min_kp)
            native = action_flat[:, self._pose_slice] if batched else action_first[self._pose_slice]
            # physical_kp_vals = np.concatenate([kp_diag, kp_rot])
            physical_kp_vals = torch.cat([kp_diag, kp_rot], dim=0)

        else:
            # Fixed: Kp frozen, pure passthrough — log constant Kp for comparison metrics
            kp_matrix = torch.eye(3) * self.min_kp
            kp_rot    = torch.ones(3) * self.min_kp
            native = action_flat[:, self._pose_slice] if batched else action_first[self._pose_slice]
            physical_kp_vals = torch.ones(6) * self.min_kp

        return kp_matrix, kp_rot, native, physical_kp_vals

    def _apply_riemannian_scaling(self, kp_matrix, native_action):
        """
        Pre-multiply delta_pos (first 3 dims of native action) by λ_max-normalised
        Kp_matrix.  Orientation dims [3:6] and gripper are passed through unchanged.
        Handles both unbatched (action_dim,) and batched (n_envs, action_dim) actions.
        """
        if not (self.use_spd or self.use_diag or self._use_variable_kp):
            return native_action

        # Only apply scaling if we have at least 3 position dimensions
        n_pos_dims = 3 if (native_action.ndim == 1) else native_action.shape[-1]
        if n_pos_dims < 3:
            return native_action

        eigvals  = torch.linalg.eigvalsh(kp_matrix)
        lam_max  = torch.clamp(eigvals.max(), min=1e-6)
        Kp_norm  = kp_matrix / lam_max

        out = native_action.clone()
        if native_action.ndim == 1:
            out[:3] = Kp_norm @ native_action[:3]
        else:  # batched: (n_envs, action_dim)
            out[:, :3] = (Kp_norm @ native_action[:, :3].T).T
        return out

    def _compute_spd_metrics(self, kp_matrix):
        if isinstance(kp_matrix, torch.Tensor):
            kp_matrix = kp_matrix.detach().cpu().numpy()

        epsilon  = 1e-6
        safe_kp  = kp_matrix + np.eye(3) * epsilon
        safe_prev= self._prev_Kp + np.eye(3) * epsilon

        self.cond_num_history.append(np.linalg.cond(safe_kp))
        self.euclidean_jerk_history.append(np.linalg.norm(safe_kp - safe_prev, ord="fro"))

        try:
            eigs      = spla.eigvals(safe_kp, safe_prev)
            real_eigs = np.clip(np.real(eigs), 1e-8, np.inf)
            riem_jerk = np.sqrt(np.sum(np.log(real_eigs) ** 2))
        except Exception:
            riem_jerk = 0.0
        self.riemannian_jerk_history.append(riem_jerk)

        off_diag = ~np.eye(3, dtype=bool)
        self.coupling_history.append(np.linalg.norm(safe_kp[off_diag]))
        self._prev_Kp = safe_kp.copy()

    def _get_contact_force(self):
        try:
            tcp_force = self.env.unwrapped.agent.tcp.get_net_contact_forces()
            if isinstance(tcp_force, torch.Tensor):
                # tcp_force = tcp_force.cpu().numpy()
                return tcp_force.float().mean(dim=0).norm().item()
            return float(np.linalg.norm(np.asarray(tcp_force).mean(axis=0)))
        except Exception:
            return 0.0

    def _build_llm_obs(self):
        """Build a minimal obs dict for the LLM planner from available sim state."""
        try:
            env  = self.env.unwrapped
            def _np(t):
                return t.float().cpu().numpy() if isinstance(t, torch.Tensor) else np.asarray(t)

            eef_pos     = _np(env.agent.tcp.pose.p).mean(axis=0) if hasattr(env, "num_envs") else _np(env.agent.tcp.pose.p)
            tcp_force   = self._get_contact_force()
            contact     = tcp_force > 0.5

            obs = {"robot0_eef_pos": eef_pos, "robot0_contact": contact}

            # Try to get PiH-specific extra state if available (privileged — only used by LLM)
            try:
                raw = env.get_obs()
                extra = raw.get("extra", {}) if isinstance(raw, dict) else {}
                if "peg_head_pos_wrt_goal" in extra:
                    obs["peg_head_pos"] = _np(extra["peg_head_pos_wrt_goal"]).mean(axis=0)
                if "insertion_depth" in extra:
                    obs["insertion_depth"] = float(_np(extra["insertion_depth"]).mean())
            except Exception:
                pass

            return obs
        except Exception:
            return {}

    # ── Gymnasium API ─────────────────────────────────────────────────────────

    def reset(self, seed=None, options=None):
        _, info = self.env.reset(seed=seed, options=options)

        self._reset_histories()
        if self.is_eval:
            self.kp_history = []
        if self.llm_planner is not None:
            self.llm_planner.reset()

        obs = self._extract_sim2real_obs() if self.use_sim2real_obs else _

        return obs, info

    def step(self, action):
        # ── LLM prior blending ───────────────────────────────────────────────
        if self.llm_planner is not None and self._extra_dims > 0:
            suggestion = self.llm_planner.query(self._build_llm_obs())
            w = suggestion.confidence
            n = len(suggestion.action_prior)
            action = action.clone()
            action_prior_t = torch.as_tensor(suggestion.action_prior, device=action.device, dtype=action.dtype)
            action[:, :n] = (1 - w) * action[:, :n] + w * action_prior_t
            self._current_llm_mode = suggestion.mode

        # ── Parse SPD action ─────────────────────────────────────────────────
        kp_matrix, kp_rot, native_action, physical_kp_vals = self._spd_from_action(action)

        # ── Riemannian delta_pos scaling ─────────────────────────────────────
        native_scaled = self._apply_riemannian_scaling(kp_matrix, native_action)

        native_scaled[..., -1] = -1.0

        # ── Physics step ─────────────────────────────────────────────────────
        _, reward, terminated, truncated, info = self.env.step(native_scaled)
        self._episode_steps += 1

        # ── Filtered observation ─────────────────────────────────────────────
        obs = self._extract_sim2real_obs() if self.use_sim2real_obs else _

        # ── SPD metrics ──────────────────────────────────────────────────────
        self._compute_spd_metrics(kp_matrix)

        force = self._get_contact_force()
        self.force_history.append(force)
        info["step/contact_force"] = force
        if force > 0.5:
            self._episode_contact_steps += 1

        # self._ep_kp += physical_kp_vals
        self._ep_kp += physical_kp_vals.detach().cpu().numpy()
        if self.is_eval:
            # self.kp_history.append(physical_kp_vals.copy())
            self.kp_history.append(physical_kp_vals.detach().cpu().numpy())

        if self.llm_planner is not None:
            info["llm/impedance_mode"]   = self._current_llm_mode
            info["llm/prior_confidence"] = self.llm_planner.prior_weight

        # ── Episode-end aggregation ──────────────────────────────────────────
        # Handle both batched and unbatched dones (may be np.ndarray or torch.Tensor)
        if isinstance(terminated, torch.Tensor):
            terminated_any = terminated.any().item()
        else:
            terminated_any = terminated.any() if isinstance(terminated, np.ndarray) else terminated

        if isinstance(truncated, torch.Tensor):
            truncated_any = truncated.any().item()
        else:
            truncated_any = truncated.any() if isinstance(truncated, np.ndarray) else truncated

        if terminated_any or truncated_any:
            n = max(self._episode_steps, 1)
            kp_avg = self._ep_kp / n

            info["physics/kp_trans_x_avg"] = kp_avg[0]
            info["physics/kp_trans_y_avg"] = kp_avg[1]
            info["physics/kp_trans_z_avg"] = kp_avg[2]
            info["physics/kp_rot_x_avg"]   = kp_avg[3]
            info["physics/kp_rot_y_avg"]   = kp_avg[4]
            info["physics/kp_rot_z_avg"]   = kp_avg[5]
            info["physics/contact_step_ratio"] = self._episode_contact_steps / n

            def _smean(lst): return float(np.mean(lst)) if lst else 0.0
            def _smax(lst):  return float(np.max(lst))  if lst else 0.0

            info["smoothness/avg_cond_num"]          = _smean(self.cond_num_history)
            info["smoothness/max_cond_num"]          = _smax(self.cond_num_history)
            info["smoothness/avg_euclidean_jerk"]    = _smean(self.euclidean_jerk_history)
            info["smoothness/avg_riemannian_jerk"]   = _smean(self.riemannian_jerk_history)
            info["smoothness/avg_coupling_magnitude"]= _smean(self.coupling_history)
            info["smoothness/avg_force"]             = _smean(self.force_history)
            info["smoothness/std_force"]             = float(np.std(self.force_history)) if self.force_history else 0.0

            if self.llm_planner is not None:
                llm_stats = self.llm_planner.get_episode_stats()
                info.update(llm_stats)

            if self.task_metrics_fn is not None:
                extra = self.task_metrics_fn(self.env, info)
                if extra:
                    info.update(extra)

        return obs, reward, terminated, truncated, info
