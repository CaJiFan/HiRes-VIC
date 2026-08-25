import torch
import math
import random
import gymnasium as gym
import numpy as np
import scipy.linalg as spla
from gymnasium import spaces
from hires_vic.geometry.riemannian import spd_grl_map
from hires_vic.geometry.lie_groups import so3_log_map
from robosuite.utils import transform_utils as T
import wandb

class GeometricWrapper(gym.Wrapper):
    def __init__(
        self,
        env,
        use_spd_manifold=False,
        use_lie_group=False,
        use_diag_manifold=False,
        use_fixed=False,
        is_eval=False,
        stiffness_penalty=0.0,
        success_bonus=0.0,
        force_penalty=0.0,
        terminate_on_unsafe=False,
        use_ema=False,
        use_llm_prior=False,
        llm_backend="ollama",
        llm_model="llama3.2",
        llm_query_interval=50,
        llm_prior_weight=0.4,
        llm_profile_path=None,
        llm_anneal_steps=0,
        llm_anneal_floor=0.05,
        llm_anneal_schedule="linear",
        task_type=None,
        task_metrics_fn=None,
        use_vision=False,
        early_terminate_on_success=False, # If False, runs until horizon so open-door/peg-insert rewards accumulate
        add_prior_obs=False,          # If True, append [prior_actions, w] to obs for ALL configs
        use_quality_reward=False,     # Use checkpoint-gated quality reward (arXiv:2502.12599 adaptation)
        use_sequential_waypoints=True,# Enforce sequential 0->1->2->3->4 waypoint guidance (matches arXiv:2502.12599)
        quality_f_target=15.0,        # Target normal force in Newtons for the Gaussian reward
        quality_sigma=15.0,           # Std-dev (N) of the force quality Gaussian
        quality_r_checkpoint=0.08,    # Checkpoint radius (m): how close to a marker to earn quality reward
        quality_w_con=1.5,            # Weight for checkpoint-gated contact reward
        quality_w_force=2.0,          # Weight for force quality Gaussian reward
        quality_w_guide=1.5,          # Weight for nearest-marker guidance reward (larger scale than checkpoint)
        quality_guide_scale=0.35,     # Length scale (m) for r_guide — must be >> r_checkpoint so gradient exists from hover height
    ):
        super().__init__(env)

        self.gym_env = self.env
        
        self.use_spd_manifold = use_spd_manifold
        self.use_lie_group = use_lie_group
        self.use_diag_manifold = use_diag_manifold
        self.use_fixed = use_fixed
        self.stiffness_penalty = stiffness_penalty
        self.success_bonus = success_bonus
        self.early_terminate_on_success = early_terminate_on_success
        self.force_penalty = force_penalty
        self.terminate_on_unsafe = terminate_on_unsafe
        self.use_ema = use_ema
        self.is_eval = is_eval
        self.add_prior_obs = add_prior_obs  # Whether to append [prior, w] to obs
        self.use_vision = use_vision

        # --- QUALITY REWARD (arXiv:2502.12599 adaptation) ---
        self.use_quality_reward = use_quality_reward
        self.use_sequential_waypoints = use_sequential_waypoints
        self.quality_f_target = quality_f_target
        self.quality_sigma = quality_sigma
        self.quality_r_checkpoint = quality_r_checkpoint
        self.quality_w_con = quality_w_con
        self.quality_w_force = quality_w_force
        self.quality_w_guide = quality_w_guide
        self.quality_guide_scale = quality_guide_scale

        self.dt = 1/20 # Assuming 20 Hz control frequency
        self.prev_Kp = None
        self.prev_ang_vel = None

        self.cond_num_history = []
        self.euclidean_jerk_history = []
        self.riemannian_jerk_history = []
        self.coupling_history = []
        
        # Universal Stiffness EMA Filter
        self._ema_kp_matrix = None
        self._ema_kp_rot = None
        self.ema_alpha = 0.15
        self.ang_accel_history = []
        self.force_history = []
        # SPD explosion diagnostics
        self._spd_pre_clamp_eigmax_history = []
        self._spd_clamp_violations = 0
        self._spd_coupling_max_history = []

        if self.is_eval:
            self.kp_history = []
        
        # Seed deferred from make_env() — consumed on the first reset() call.
        # This avoids calling env.reset(seed=...) explicitly in make_env, which
        # would trigger a second full episode-setup sequence (e.g. TeleportWrapper
        # scripted steps) on top of the one SubprocVecEnv already runs at init.
        self._pending_seed: int | None = None

        # Internal counters for logging
        self.episode_force_sum = 0.0
        self.episode_steps = 0
        self.episode_count = 0
        self.episode_contact_steps = 0
        self.violation_count = 0

        self.ep_kp_trans_x = 0.0
        self.ep_kp_trans_y = 0.0
        self.ep_kp_trans_z = 0.0
        self.ep_kp_rot_x = 0.0
        self.ep_kp_rot_y = 0.0
        self.ep_kp_rot_z = 0.0

        self.min_kp, self.max_kp  = (1, 300)

        self.task_type = task_type
        self.task_metrics_fn = task_metrics_fn

        self.use_llm_prior = use_llm_prior
        self.llm_planner = None
        if self.use_llm_prior:
            from hires_vic.llm.impedance_planner import LLMImpedancePlanner
            print(f"🤖 Initializing LLM Impedance Planner with backend: {llm_backend} and model: {llm_model}")
            print(f"   Vision enabled: {use_vision}")
            self.llm_planner = LLMImpedancePlanner(
                backend=llm_backend,
                model=llm_model,
                query_every_n_steps=llm_query_interval,
                prior_weight=llm_prior_weight,
                profile_path=llm_profile_path,
                use_spd_manifold=self.use_spd_manifold,
                use_vision=use_vision,
                image_size=(224, 224),
                anneal_steps=llm_anneal_steps,
                anneal_floor=llm_anneal_floor,
                anneal_schedule=llm_anneal_schedule,
            )

            self._current_llm_mode = 'align'
        
        # Camera buffers for VLM
        self.last_wrist_image = None
        self.last_view_image = None

        self._last_obs_dict = {}
        # Peg tip Z position cached each episode in reset() for the P-controller.
        # (peg position is not in the observation dict, so we read it from the sim once.)
        self._peg_top_z = None
        self._consecutive_stuck_steps = 0
        self._last_nut_z = None

        print(f"""
            🔧 Robosuite Wrapper Initialized
            | SPD: {self.use_spd_manifold}
            | Lie Group: {self.use_lie_group}
            | Diag Manifold: {self.use_diag_manifold}
            | Fixed: {self.use_fixed}
        """)


        action_dim = 6 if self.use_fixed else 12
        action_dim += 3 if self.use_spd_manifold else 0
        # action_dim = 9 if self.use_spd_manifold else 6

        # prior_dim: how many action dims belong to the stiffness parameterization.
        #   - SPD manifold (Mandel basis): 6 params → 3×3 SPD matrix
        #   - Diagonal / baseline:         3 params → diagonal kp_trans
        # This is also the number of dims the LLM prior blends (action[:prior_dim]).
        self.prior_dim = 9 if self.use_spd_manifold else 6

        # action_dim: what the RL policy actually outputs.
        if self.task_type == 'nutassembly':
            # We want to learn stiffness (prior_dim) + [dx, dy, dz, d_roll, d_pitch, d_yaw] wiggle.
            # (Note: dz will be overwritten by P-controller, but agent outputs the full 6D kin array).
            # Gripper is forced closed.
            action_dim = self.prior_dim + 6
        
        # We only add a separate gripper dimension if it's not wipe and not nutassembly
        gripper_dim = 0 if ('wipe' in self.task_type or 'nutassembly' in self.task_type) else 1
        if gripper_dim > 0 and self.task_type != 'nutassembly':
            action_dim += gripper_dim


        self.action_space = spaces.Box(
            low=-1.0, high=1.0, shape=(action_dim,), dtype=np.float32
        )
        print(f"▶️ Action space: task_type='{self.task_type}' → shape={self.action_space.shape} "
              f"(prior_dim={self.prior_dim})")

        # --- TRUE RESIDUAL RL: STATE TRACKERS ---
        # extra_obs_dim: how many extra dims to append to the flat obs.
        # Controlled by add_prior_obs (default False).
        # When True, [prior_actions (prior_dim), confidence_w (1)] are appended
        # for ALL configs (LLM and non-LLM alike) to keep obs spaces consistent.
        # Non-LLM runs will always see zeros in these dims.
        self.extra_obs_dim = (self.prior_dim + 1) if self.add_prior_obs else 0
        
        self.current_prior = np.zeros(self.prior_dim, dtype=np.float32)
        self.current_w = 0.0

        # Observation space expansion
        obs = self.env.reset()
        flat_obs = self._flatten_obs(obs)
        
        expanded_obs_shape = (flat_obs.shape[0] + self.extra_obs_dim,)

        print('▶️ Base Observation space shape: ', flat_obs.shape)
        print('▶️ Expanded Observation space shape: ', expanded_obs_shape,
              '(prior obs appended)' if self.add_prior_obs else '(no prior obs)')

        self.observation_space = spaces.Box(
            low=-np.inf, high=np.inf, shape=expanded_obs_shape, dtype=np.float32
        )

    def _get_active_waypoint_rel_pos(self):
        """
        Helper to compute the 3D relative vector from gripper EEF to the active target marker.
        Works in both sequential and nearest mode.
        """
        try:
            raw_env = self.env.unwrapped
            raw_obs = self._last_obs_dict
            if not raw_obs:
                raw_obs = raw_env._get_observations()
                
            eef_pos = np.array(raw_obs.get('robot0_eef_pos', [0.0, 0.0, 0.0]), dtype=float)
            wiped_markers = getattr(raw_env, 'wiped_markers', [])
            all_markers = getattr(raw_env, 'model', None)
            if all_markers is not None:
                all_markers = getattr(all_markers.mujoco_arena, 'markers', [])
            else:
                all_markers = []

            target_pos = None
            if getattr(self, 'use_sequential_waypoints', True):
                # Sequential mode: sort unwiped markers spatially along Y-axis
                unwiped = []
                for marker in all_markers:
                    if marker in wiped_markers:
                        continue
                    bid = raw_env.sim.model.body_name2id(marker.root_body)
                    pos = np.array(raw_env.sim.data.body_xpos[bid], dtype=float)
                    unwiped.append((marker, pos))
                if len(unwiped) > 0:
                    unwiped.sort(key=lambda item: item[1][1])
                    target_pos = unwiped[0][1]
            else:
                # Nearest mode: find closest unwiped marker
                min_dist = np.inf
                for marker in all_markers:
                    if marker in wiped_markers:
                        continue
                    bid = raw_env.sim.model.body_name2id(marker.root_body)
                    pos = np.array(raw_env.sim.data.body_xpos[bid], dtype=float)
                    dist = np.linalg.norm(eef_pos - pos)
                    if dist < min_dist:
                        min_dist = dist
                        target_pos = pos

            if target_pos is not None:
                return target_pos - eef_pos
        except Exception:
            pass

        return np.zeros(3, dtype=np.float32)

    def _flatten_obs(self, obs):
        """
        Flattens the Robosuite dictionary into a single vector for the RL agent.
        Selects only the useful keys (proprioception + object state).
        Applies Lie Group Logarithmic Map to orientation if using GRL.
        """
        raw_obs = self.env.unwrapped._get_observations()
        # Cache the full raw observation dict for planner/inspection
        self._last_obs_dict = raw_obs

        # Make a shallow copy and defensively remove large camera/image entries
        raw_obs_c = raw_obs.copy()
        raw_obs_c.pop('robot0_proprio-state', None)
        raw_obs_c.pop('object-state', None)
        all_keys = list(raw_obs_c.keys())

        # if self.is_eval:
            # print("🔍 Raw observation keys:\n", all_keys)
        # else:
            # print("🔍 Raw observation keys (Training):\n", all_keys)

        # Exclude common camera/image keys that would massively increase the flat obs size
        skip_substrings = ('image', 'camera', 'rgb', 'frontview', 'agentview', 'render', 'depth')
        filtered_keys = [k for k in all_keys if not any(sub in k.lower() for sub in skip_substrings)]

        new_obs = []
        for key in filtered_keys:
            try:
                val = raw_obs_c.get(key)
                if val is None:
                    continue

                if 'quat' in key and self.use_lie_group:
                    quat = np.asarray(val, dtype=np.float32)
                    # Normalize quaternion to avoid numeric issues
                    norm = np.linalg.norm(quat)
                    if norm < 1e-8:
                        quat = np.array([1.0, 0.0, 0.0, 0.0], dtype=np.float32)
                    else:
                        quat = quat / norm
                    rot_mat = T.quat2mat(quat)
                    rot_tensor = torch.tensor(rot_mat, dtype=torch.float32).unsqueeze(0)
                    omega = so3_log_map(rot_tensor).squeeze(0).detach().numpy()
                    new_obs.append(omega)
                else:
                    arr = np.asarray(val).flatten()
                    # Extra safety: skip extremely large arrays
                    if arr.size > 200000:
                        continue
                    new_obs.append(arr)
            except Exception:
                # Best-effort: ignore keys that fail to flatten
                continue

        # Fallbacks: try to construct something usable if filtered produced nothing
        if not new_obs:
            try:
                parts = []
                for k, v in raw_obs.items():
                    if any(sub in k.lower() for sub in skip_substrings):
                        continue
                    parts.append(np.asarray(v).flatten())
                if parts:
                    return np.concatenate(parts).astype(np.float32)
            except Exception:
                pass

            # Final fallback: if `obs` param looks like a numpy array, return its flattened form
            try:
                param = obs
                if isinstance(param, (list, tuple)):
                    param = param[0]
                return np.asarray(param).flatten().astype(np.float32)
            except Exception:
                return np.zeros((0,), dtype=np.float32)

        if self.task_type == 'wipe':
            rel_pos = self._get_active_waypoint_rel_pos()
            if self._last_obs_dict is not None and isinstance(self._last_obs_dict, dict):
                self._last_obs_dict['gripper_to_active_waypoint'] = rel_pos
            if getattr(self, 'use_sequential_waypoints', False):
                new_obs.append(rel_pos)

        return np.concatenate(new_obs).astype(np.float32)

    def _compute_quality_reward(self) -> float:
        """
        Checkpoint-gated quality reward adapted from arXiv:2502.12599.

        Components:
          r_con_q   : contact reward gated by being within quality_r_checkpoint of nearest unwiped marker
          r_force_q : Gaussian force quality reward exp(-(F_n - F_target)^2 / 2*sigma^2), also gated
          r_guide   : smooth nearest-marker guidance reward (always on, no gating needed)

        Returns:
            float: total quality reward for this step
        """
        try:
            raw_env = self.env.unwrapped
            raw_obs = self._last_obs_dict

            # ── 1. EEF position ──────────────────────────────────────────
            eef_pos = np.array(raw_obs.get('robot0_eef_pos', [0, 0, 0]), dtype=float)

            # ── 2. Determine target waypoint (Sequential vs. Nearest) ────
            wiped_markers = getattr(raw_env, 'wiped_markers', [])
            all_markers = getattr(raw_env, 'model', None)
            if all_markers is not None:
                all_markers = getattr(all_markers.mujoco_arena, 'markers', [])
            else:
                all_markers = []

            min_dist = np.inf
            if getattr(self, 'use_sequential_waypoints', True):
                # Spatial Sequential Mode (arXiv:2502.12599):
                # Sort unwiped markers spatially along Y-axis (left-to-right: -Y to +Y).
                # This guarantees a smooth, monotonic left-to-right sweeping trajectory across the table!
                unwiped = []
                for marker in all_markers:
                    if marker in wiped_markers:
                        continue
                    try:
                        bid = raw_env.sim.model.body_name2id(marker.root_body)
                        pos = np.array(raw_env.sim.data.body_xpos[bid], dtype=float)
                        unwiped.append((marker, pos))
                    except Exception:
                        continue

                if len(unwiped) > 0:
                    unwiped.sort(key=lambda item: item[1][1])  # Sort by Y-coordinate
                    active_marker, active_pos = unwiped[0]
                    min_dist = np.linalg.norm(eef_pos - active_pos)
            else:
                # Nearest Mode: find closest unwiped marker among all remaining
                for marker in all_markers:
                    if marker in wiped_markers:
                        continue
                    try:
                        body_id = raw_env.sim.model.body_name2id(marker.root_body)
                        marker_pos = np.array(raw_env.sim.data.body_xpos[body_id], dtype=float)
                        dist = np.linalg.norm(eef_pos - marker_pos)
                        if dist < min_dist:
                            min_dist = dist
                    except Exception:
                        continue

            # If all markers are wiped, task is complete
            if min_dist == np.inf:
                return 10.0  # Completion bonus

            # ── 3. Smooth Checkpoint Indicator (Gaussian weighting) ───────
            # Using a smooth Gaussian weighting (sigma_c = 0.15m) instead of a hard step threshold
            # eliminates the reward cliff and provides a continuous, convex gradient toward
            # unwiped markers across the entire table.
            sigma_c = 0.15
            I_checkpoint = float(np.exp(- (min_dist ** 2) / (2.0 * (sigma_c ** 2))))

            # ── 4. Contact state ─────────────────────────────────────────
            has_contact = float(bool(raw_obs.get('robot0_contact', False)))

            # ── 5. Normal force (projected onto table normal) ────────────
            # Table normal for 45-deg tilt around Y: n = [sin(45), 0, cos(45)]
            tilt_rad = np.radians(45.0)
            table_normal = np.array([np.sin(tilt_rad), 0.0, np.cos(tilt_rad)])
            try:
                robot = raw_env.robots[0]
                arm_key = robot.arms[0] if hasattr(robot, 'arms') else 'right'
                ft = np.array(robot.recent_ee_forcetorques[arm_key].current[:3], dtype=float)
                F_normal = abs(np.dot(ft, table_normal))  # scalar normal force magnitude
            except Exception:
                F_normal = 0.0

            # ── 6. Reward components ─────────────────────────────────────
            # a) Contact reward, weighted by smooth checkpoint indicator
            r_con_q = self.quality_w_con * has_contact * I_checkpoint

            # b) Force quality: Bounded Gaussian centered at F_target=15N, sigma=15N.
            #    Forces from 0 to 15N earn full reward (1.0). Forces from 15N to 60N decay smoothly
            #    without instantly collapsing to zero, providing a continuous gradient.
            f_excess = max(0.0, F_normal - self.quality_f_target)
            sigma_f = getattr(self, 'quality_sigma', 15.0)
            if has_contact > 0:
                r_force_q = self.quality_w_force * I_checkpoint * np.exp(-(f_excess ** 2) / (2.0 * (sigma_f ** 2)))
            else:
                r_force_q = 0.0

            # c) Smooth guidance toward nearest unwiped marker (always on)
            r_guide = self.quality_w_guide * (1.0 - np.tanh(min_dist / self.quality_guide_scale))

            return float(r_con_q + r_force_q + r_guide)

        except Exception as e:
            # Fail silently — never crash the training loop over reward computation
            return 0.0

    def reset(self, seed=None, options=None):
        # Robosuite's native reset doesn't take a seed directly in the call usually,
        # but we can set numpy's seed if needed.
        # Also consume any deferred seed stored by make_env() at construction time
        # (avoids a second env.reset() call that would re-run TeleportWrapper setup).
        effective_seed = seed if seed is not None else self._pending_seed
        self._pending_seed = None  # consume once
        if effective_seed is not None:
            np.random.seed(effective_seed)
            random.seed(effective_seed)

        # IMPORTANT: call self.env.reset() — NOT self.gym_env.reset().
        # For NutAssembly, self.env is TeleportWrapper, whose reset() runs all
        # scripted hover+teleport setup steps on the underlying GymWrapper before
        # returning. Calling self.gym_env.reset() directly would bypass TeleportWrapper
        # and immediately reset the environment, undoing all setup work.
        # For other envs (Door, Wipe), self.env IS self.gym_env, so no difference.
        self.env.reset()

        # Reset EMA filter state
        self._ema_kp_matrix = None
        self._ema_kp_rot = None

        # Cache peg tip Z for the P-controller dz calculation.
        # 'peg_head_pos' does not exist in the obs dict; we read the peg body
        # position from MuJoCo directly once per episode.
        if self.task_type == 'nutassembly':
            try:
                sim = self.gym_env.unwrapped.sim
                peg_key = 'peg1' if 'square' in type(self.gym_env.unwrapped).__name__.lower() else 'peg2'
                peg_id = sim.model.body_name2id(peg_key)
                self._peg_top_z = float(sim.data.body_xpos[peg_id][2])
            except Exception as e:
                print(f'[GeometricWrapper] Could not cache peg Z: {e}')
                self._peg_top_z = None

        flat_obs = self._flatten_obs(None)

        if self.llm_planner is not None:
            self.llm_planner.reset()
            # ✅ FIX: Initialize prior from planner's default suggestion
            self.current_prior = self.llm_planner._last.action_prior.copy()[:self.prior_dim]
            self.current_w = self.llm_planner._last.confidence
        else:
            self.current_prior = np.zeros(self.prior_dim, dtype=np.float32)
            self.current_w = 0.0
        
        if self.add_prior_obs:
            extra_state = np.concatenate([self.current_prior, [self.current_w]], dtype=np.float32)
            flat_obs = np.concatenate([flat_obs, extra_state], dtype=np.float32)

        self.prev_Kp = np.eye(3) # Default starting stiffness
        self.prev_ang_vel = np.zeros(3)
        self.cond_num_history = []
        self.euclidean_jerk_history = []
        self.riemannian_jerk_history = []
        self.coupling_history = []
        self.ang_accel_history = []
        self.force_history = []
        self._spd_pre_clamp_eigmax_history = []
        self._spd_clamp_violations = 0
        self._spd_coupling_max_history = []

        self.episode_force_sum = 0.0
        self.episode_steps = 0
        self.episode_count = 0
        self.episode_contact_steps = 0
        self.violation_count = 0
        self.force_exceedances = 0  # Steps where contact force > FORCE_THRESHOLD

        self.ep_kp_trans_x = 0.0
        self.ep_kp_trans_y = 0.0
        self.ep_kp_trans_z = 0.0
        self.ep_kp_rot_x = 0.0
        self.ep_kp_rot_y = 0.0
        self.ep_kp_rot_z = 0.0

        if self.is_eval:
            self.kp_history.clear()
        
        # Clear image buffers
        self.last_wrist_image = None
        self.last_view_image = None
        
        return flat_obs, {}

    def _compute_stiffness(self, action):
        """
        Stage 1: Resolves the 3x3 Cartesian stiffness matrix and 3D rotational stiffness vector.
        Supports:
          - use_fixed: constant controller stiffness
          - use_spd_manifold: Riemannian Log-Euclidean Residual (LLM) or full 6D Mandel basis (pure RL)
          - Non-SPD (BASELINE & DIAG): Euclidean Residual Policy Learning (LLM) or 6D diagonal (pure RL)
        """
        if self.use_fixed:
            robot = self.gym_env.unwrapped.robots[0]
            fixed_kp = robot.composite_controller.part_controller_config['right']['kp']
            return np.eye(3, dtype=np.float32) * fixed_kp, np.ones(3, dtype=np.float32) * fixed_kp

        if self.use_spd_manifold:
            if self.llm_planner is not None:
                # ── Log-Euclidean Riemannian Residual Mapping (Arsigny et al. 2006, Davchev et al. 2022) ──
                # Geodesic boundary-scaled tangent residual: prevents exponential overshoot beyond [min_kp, max_kp]
                prior_trans = self.min_kp + 0.5 * (self.current_prior[:3] + 1.0) * (self.max_kp - self.min_kp)
                prior_rot   = self.min_kp + 0.5 * (self.current_prior[6:9] + 1.0) * (self.max_kp - self.min_kp)

                ln_min = np.log(self.min_kp)
                ln_max = np.log(self.max_kp)
                ln_prior = np.log(np.clip(prior_trans, self.min_kp, self.max_kp))

                S_prior = np.zeros(6, dtype=np.float32)
                S_prior[0:3] = ln_prior
                S_prior[3:6] = 0.0  # Uncoupled nominal prior

                # Tangent space residual from RL policy: geodesic interpolation toward manifold boundary
                delta_S = np.zeros(6, dtype=np.float32)
                for i in range(3):
                    if action[i] >= 0:
                        delta_S[i] = action[i] * (1.0 - self.current_w) * (ln_max - ln_prior[i])
                    else:
                        delta_S[i] = action[i] * (1.0 - self.current_w) * (ln_prior[i] - ln_min)

                # Bounded off-diagonal coupling matching pure RL SPD exploration
                delta_S[3:6] = action[3:6] * (1.0 - self.current_w) * 0.2

                S_total = S_prior + delta_S
                m_tensor_rl = torch.tensor(S_total, dtype=torch.float32).unsqueeze(0)
                kp_matrix = np.real(spd_grl_map(m_tensor_rl).squeeze(0).detach().numpy())

                delta_k_rot = action[6:9] * (1.0 - self.current_w) * 0.5 * (self.max_kp - self.min_kp)
                kp_rot = np.clip(prior_rot + delta_k_rot, self.min_kp, self.max_kp)
            else:
                # Baseline SPD: Full 6D Mandel basis learning (pure RL).
                # Linear decoding for diagonals to prevent Limp Initialization.
                target_physical = self.min_kp + 0.5 * (action[:3] + 1.0) * (self.max_kp - self.min_kp)

                m_params_rl = np.zeros(6, dtype=np.float32)
                m_params_rl[0:3] = np.log(target_physical)
                m_params_rl[3:6] = action[3:6] * 0.2  # Full 6D off-diagonal exploration

                m_tensor_rl = torch.tensor(m_params_rl, dtype=torch.float32).unsqueeze(0)
                kp_matrix = np.real(spd_grl_map(m_tensor_rl).squeeze(0).detach().numpy())
                kp_rot = self.min_kp + 0.5 * (action[6:9] + 1.0) * (self.max_kp - self.min_kp)

            # Spectral clamping and symmetrization
            _eigvals, _eigvecs = np.linalg.eigh(kp_matrix)
            _pre_clamp_max = float(np.max(_eigvals))
            self._spd_pre_clamp_eigmax_history.append(_pre_clamp_max)
            if _pre_clamp_max > self.max_kp or float(np.min(_eigvals)) < self.min_kp:
                self._spd_clamp_violations += 1
            _eigvals = np.clip(_eigvals, self.min_kp, self.max_kp)
            kp_matrix = _eigvecs @ np.diag(_eigvals) @ _eigvecs.T
            kp_matrix = 0.5 * (kp_matrix + kp_matrix.T)
            return kp_matrix, kp_rot

        else:
            # ── Non-SPD (BASELINE & DIAG): 6D Diagonal Stiffness ──
            if self.llm_planner is not None:
                prior_trans = self.min_kp + 0.5 * (self.current_prior[:3] + 1.0) * (self.max_kp - self.min_kp)
                prior_rot   = self.min_kp + 0.5 * (self.current_prior[3:6] + 1.0) * (self.max_kp - self.min_kp)

                delta_trans = action[:3] * (1.0 - self.current_w) * 0.5 * (self.max_kp - self.min_kp)
                delta_rot   = action[3:6] * (1.0 - self.current_w) * 0.5 * (self.max_kp - self.min_kp)

                kp_trans = np.clip(prior_trans + delta_trans, self.min_kp, self.max_kp)
                kp_rot   = np.clip(prior_rot + delta_rot, self.min_kp, self.max_kp)
            else:
                kp_trans = self.min_kp + 0.5 * (action[:3] + 1.0) * (self.max_kp - self.min_kp)
                kp_rot   = self.min_kp + 0.5 * (action[3:6] + 1.0) * (self.max_kp - self.min_kp)

            return np.diag(kp_trans), kp_rot

    def _assemble_robosuite_action(self, action, kp_matrix_3x3, kp_rot_scaled):
        """
        Stage 2: Assembles the robosuite action vector containing:
          - Stiffness parameters (9 elements flattened for SPD, or 6 elements for Baseline)
          - Kinematic commands (P-controller descent for NutAssembly, or RL delta pos+ori for Door/Wipe)
          - Gripper command
        """
        if self.use_fixed:
            low, high = self.action_space.low, self.action_space.high
            return low + 0.5 * (action + 1.0) * (high - low)

        stiffness_part = kp_matrix_3x3.flatten() if self.use_spd_manifold else np.diag(kp_matrix_3x3)

        if self.task_type == 'nutassembly':
            kin_raw = action[-6:]
            max_trans_wiggle = 0.002 # 2 mm per step
            max_rot_wiggle = 0.05    # ~2.8 degrees per step

            dx = float(kin_raw[0] * max_trans_wiggle)
            dy = float(kin_raw[1] * max_trans_wiggle)

            d_roll  = float(kin_raw[3] * max_rot_wiggle)
            d_pitch = float(kin_raw[4] * max_rot_wiggle)
            d_yaw   = float(kin_raw[5] * max_rot_wiggle)

            _P_GAIN = 1.5
            _FALLBACK_DZ = -0.010
            try:
                nut_pos = self._last_obs_dict.get('SquareNut_pos',
                          self._last_obs_dict.get('RoundNut_pos', None))
                if nut_pos is not None and self._peg_top_z is not None:
                    dz_above = float(nut_pos[2]) - self._peg_top_z
                    dz = float(-_P_GAIN * max(dz_above, 0.0))
                    dz = np.clip(dz, -0.04, 0.0)
                else:
                    dz = _FALLBACK_DZ
            except Exception:
                dz = _FALLBACK_DZ

            trajectory_command = np.array([dx, dy, dz, d_roll, d_pitch, d_yaw], dtype=np.float32)
            gripper_command = np.array([1.0], dtype=np.float32)

            return np.concatenate([
                stiffness_part,
                kp_rot_scaled,
                trajectory_command,
                gripper_command,
            ])
        else:
            # Door / Wipe: agent controls pos+ori (+gripper if applicable) via action[prior_dim:]
            kin_gripper_part = action[self.prior_dim:]
            return np.concatenate([
                stiffness_part,
                kp_rot_scaled,
                kin_gripper_part,
            ])

    def step(self, rl_action):
        action = rl_action.copy()

        # Stage 1: Resolve physical stiffness matrix (3x3) and rotational vector (3,)
        kp_matrix_3x3, kp_rot_scaled = self._compute_stiffness(action)
        physical_kp_vals = np.concatenate([np.diag(kp_matrix_3x3), kp_rot_scaled])

        # Stage 2: Assemble robosuite action (stiffness + kinematics + gripper)
        robosuite_action = self._assemble_robosuite_action(action, kp_matrix_3x3, kp_rot_scaled)

        # ── UNIVERSAL STIFFNESS EMA FILTER ──────────────────────────────────────────
        # Applies a low-pass filter to the final physical stiffness matrix and 
        # rotational vector. By placing it here, it works universally for BASELINE, 
        # DIAG, and SPD configurations, ensuring a fair scientific comparison.
        if self.use_ema:
            if self._ema_kp_matrix is None:
                self._ema_kp_matrix = kp_matrix_3x3
                self._ema_kp_rot = physical_kp_vals[3:6]
            else:
                self._ema_kp_matrix = self.ema_alpha * kp_matrix_3x3 + (1.0 - self.ema_alpha) * self._ema_kp_matrix
                self._ema_kp_rot = self.ema_alpha * physical_kp_vals[3:6] + (1.0 - self.ema_alpha) * self._ema_kp_rot
            
            # Overwrite with smoothed values
            kp_matrix_3x3 = self._ema_kp_matrix
            
            # Update physical_kp_vals for logging/metrics
            physical_kp_vals[:3] = np.diag(self._ema_kp_matrix)
            physical_kp_vals[3:6] = self._ema_kp_rot
            
            # Repack the robosuite_action with the smoothed values
            if self.use_spd_manifold:
                robosuite_action[:9] = self._ema_kp_matrix.flatten()
                robosuite_action[9:12] = self._ema_kp_rot
            else:
                robosuite_action[:3] = np.diag(self._ema_kp_matrix)
                robosuite_action[3:6] = self._ema_kp_rot


        # print('physical_kp_vals: ', physical_kp_vals)
        # check if the env uses a gripper and if so set it to always closed (1.0). The criteria is that if the name is not a wiping
        # env, then we assume it has a gripper that needs to be closed. Allow temporary suppression (e.g., scripted primitives)
        # suppress = getattr(self, 'suppress_forced_gripper', False)
        # print(suppress)
        # if 'wipe' not in self.env.unwrapped.__class__.__name__.lower() and not suppress:
        #     # kp_rot_scaled = np.array([300.0, 300.0, 300.0], dtype=np.float32)
        #     # trajectory_command = np.array([0.0, 0.0, -0.05, 0.0, 0.0, 0.0], dtype=np.float32)
        #     # gripper_command = np.array([1.0], dtype=np.float32)
            
        #     # robosuite_action = np.concatenate([
        #     #     kp_matrix.flatten(),    # 9 elements: 3x3 matrix flattened
        #     #     kp_rot_scaled,          # Scaled rotational stiffness
        #     #     trajectory_command,     # Override pos+ori with a fixed downward trajectory for testing
        #     #     gripper_command         # Override gripper command to always closed for testing
        #     # ])
        #     robosuite_action[-1] = 1.0
            
        # print('Action after processing:', robosuite_action)
        step_return = self.gym_env.step(robosuite_action)
        self.episode_steps += 1
       
        obs, reward, terminated, truncated, info = step_return

        flat_obs = self._flatten_obs(obs)

        # Anti-Jam Penalty Logic
        if self.task_type == 'nutassembly' and not terminated:
            try:
                nut_pos = self._last_obs_dict.get('SquareNut_pos',
                          self._last_obs_dict.get('RoundNut_pos', None))
                if nut_pos is not None:
                    current_nut_z = float(nut_pos[2])
                    if self._last_nut_z is not None:
                        dz_actual = current_nut_z - self._last_nut_z
                        # The agent is "stuck" if it didn't make downward progress (> 0.5mm)
                        # We don't use abs() because moving UP shouldn't reset the stuck counter!
                        if dz_actual > -0.0005:
                            self._consecutive_stuck_steps += 1
                        else:
                            self._consecutive_stuck_steps = 0
                        
                        if self._consecutive_stuck_steps > 10:
                            reward -= 0.5  # Heavy penalty for resting lazy on the peg
                    
                    self._last_nut_z = current_nut_z
            except Exception as e:
                pass

        # Early termination on success with horizon-compensating bonus:
        # Prevents post-completion drift (where the policy flails around and accidentally re-closes the door)
        # while awarding the remaining horizon steps so early success is strictly superior to lingering.
        if not terminated:
            try:
                if bool(self.env.unwrapped._check_success()):
                    terminated = True
                    raw_env = self.env.unwrapped
                    horizon = getattr(raw_env, 'horizon', 100)
                    remaining_steps = max(0, horizon - self.episode_steps)
                    # +1.0 for current success step + 1.0 per remaining step
                    reward += 1.0 + float(remaining_steps) * 1.0
            except AttributeError:
                pass

        flat_obs = self._flatten_obs(obs)

        # --- EXTRACT CAMERA IMAGES FOR VLM ---
        try:
            # Look for end-effector and frontview camera images in the raw obs
            raw_env = getattr(self.env, 'unwrapped', self.env)
            if hasattr(raw_env, '_get_observations'):
                raw_obs_full = raw_env._get_observations()
                # Extract images by common naming patterns
                for key in raw_obs_full.keys():
                    # End-effector camera: "wrist", "eye_in_hand", "robot0_eye_in_hand", etc.
                    if any(x in key.lower() for x in ('wrist', 'eye_in_hand', 'eef')) and 'image' in key.lower():
                        img = raw_obs_full[key]
                        # print(img.shape, key)
                        if isinstance(img, np.ndarray):
                            self.last_wrist_image = img.copy()
                    # Scene camera: "frontview", "agentview", "birdview", etc.
                    if any(x in key.lower() for x in ('frontview', 'agentview', 'birdview')) and 'image' in key.lower():
                        img = raw_obs_full[key]
                        # print(img.shape, key)
                        if isinstance(img, np.ndarray):
                            self.last_view_image = img.copy()
        except Exception as e:
            pass  # Silently fail if images not available

        # --- TRUE RESIDUAL RL: UPDATE LLM & APPEND STATE ---
        if self.llm_planner is not None:
            suggestion = self.llm_planner.query(
                self._last_obs_dict,
                wrist_image=self.last_wrist_image,
                view_image=self.last_view_image
            )
            self.current_prior = suggestion.action_prior[:self.prior_dim]
            # print('current prior:', self.current_prior)
            self.current_w = suggestion.confidence
            self._current_llm_mode = suggestion.mode

        if self.add_prior_obs:
            extra_state = np.concatenate([self.current_prior, [self.current_w]], dtype=np.float32)
            flat_obs = np.concatenate([flat_obs, extra_state], dtype=np.float32)
        # print('flat obs', flat_obs.shape)
        epsilon = 1e-6
        safe_kp = kp_matrix_3x3 + np.eye(3) * epsilon
        safe_prev_kp = self.prev_Kp + np.eye(3) * epsilon
        # Condition Number (SPD Stability)
        # Measures how close the matrix is to collapsing/becoming singular
        cond_num = np.linalg.cond(safe_kp)
        self.cond_num_history.append(cond_num)
        
        # Action Jerk (Parameter Smoothness)
        # Frobenius norm of the step-to-step change in stiffness
        euclidean_action_jerk = np.linalg.norm(safe_kp - safe_prev_kp, ord='fro')
        self.euclidean_jerk_history.append(euclidean_action_jerk)
        # self.prev_Kp = kp_matrix_3x3.copy()

        # 3. TRUE Riemannian Action Jerk (AIRM)
        try:
            # Calculate the eigenvalues of (Prev_Kp^-1 * Current_Kp)
            # We use scipy's generalized eigenvalue solver for numerical stability
            eigenvalues = spla.eigvals(safe_kp, safe_prev_kp)
            # Ensure eigenvalues are strictly positive real numbers to avoid math domain errors
            real_eigs = np.clip(np.real(eigenvalues), 1e-8, np.inf)
            riemannian_jerk = np.sqrt(np.sum(np.log(real_eigs)**2))
        except Exception as e:
            print(f"⚠️ Riemannian jerk calculation failed: {e}")
            riemannian_jerk = 0.0 # Failsafe

        self.riemannian_jerk_history.append(riemannian_jerk)

        off_diag_mask = ~np.eye(3, dtype=bool)
        coupling_magnitude = np.linalg.norm(safe_kp[off_diag_mask])
        self.coupling_history.append(coupling_magnitude)
        # Track per-step max |off-diagonal| element separately for peak detection
        self._spd_coupling_max_history.append(float(np.max(np.abs(safe_kp[off_diag_mask]))))

        self.prev_Kp = safe_kp.copy()

        # Peak Angular Acceleration (Lie Group Stability)
        # Extract current angular velocity from Robosuite's proprioception
        # Adjust 'robot0_eef_vel_ang' if your observation key is slightly different
        robot = self.env.unwrapped.robots[0]
        current_ang_vel = robot._hand_ang_vel["right"]
        ang_accel = np.linalg.norm((current_ang_vel - self.prev_ang_vel) / self.dt)
        self.ang_accel_history.append(ang_accel)
        self.prev_ang_vel = current_ang_vel.copy()

        # ── QUALITY REWARD (arXiv:2502.12599 adaptation) ─────────────────────
        if self.use_quality_reward:
            quality_reward = self._compute_quality_reward()
            # Scale quality reward by the environment's reward scale and normalization factor to maintain consistent scale
            try:
                raw_env = self.env.unwrapped
                if hasattr(raw_env, 'reward_scale') and hasattr(raw_env, 'reward_normalization_factor'):
                    scale_factor = raw_env.reward_scale * raw_env.reward_normalization_factor
                    quality_reward *= scale_factor
            except Exception:
                pass
            reward += quality_reward
            # Expose for debugging/logging
            if not hasattr(self, '_last_quality_reward'):
                self._last_quality_reward = 0.0
            self._last_quality_reward = quality_reward

        # Apply the scale-invariant Riemannian Jerk Penalty universally
        # (For BASELINE's diagonal matrices, this naturally reduces to the log-ratio penalty, 
        # which perfectly matches the scale without blowing up like Euclidean jerk would).
        reward = reward - (self.stiffness_penalty * riemannian_jerk)

        self.log_info(physical_kp_vals, step_return)

        return flat_obs, reward, terminated, truncated, info
    
    def log_info(self, physical_kp_vals, step_return):
        obs, reward, terminated, truncated, info = step_return

        if self.llm_planner is not None:
            info["llm/impedance_mode"] = self._current_llm_mode
            # Log the effective (schedule-adjusted) prior weight, not just the nominal
            info["llm/prior_confidence"] = self.llm_planner._compute_effective_w()

        raw_success = self.env.unwrapped._check_success()
        info["is_success"] = bool(raw_success)
        info["success"] = bool(raw_success)

        if self.is_eval:
            self.kp_history.append(physical_kp_vals.copy())

        self.log_contact_forces()
        self.check_joint_violations()
        # Expose per-step force for callback to correlate with LLM mode
        if self.force_history:
            info["step/contact_force"] = self.force_history[-1]

        self.ep_kp_trans_x += physical_kp_vals[0]
        self.ep_kp_trans_y += physical_kp_vals[1]
        self.ep_kp_trans_z += physical_kp_vals[2]
        self.ep_kp_rot_x += physical_kp_vals[3]
        self.ep_kp_rot_y += physical_kp_vals[4]
        self.ep_kp_rot_z += physical_kp_vals[5]

         # Log Episode Averages (Only when episode ends)
        if terminated or truncated:
            self.episode_count += 1

            # Flush LLM per-episode stats into info so the callback can log them
            if self.llm_planner is not None:
                llm_stats = self.llm_planner.get_episode_stats()
                info.update(llm_stats)

            info["physics/joint_violation_count"] = self.violation_count
            info["physics/kp_trans_x_avg"] = self.ep_kp_trans_x / max(1, self.episode_steps)
            info["physics/kp_trans_y_avg"] = self.ep_kp_trans_y / max(1, self.episode_steps)
            info["physics/kp_trans_z_avg"] = self.ep_kp_trans_z / max(1, self.episode_steps)
            info["physics/kp_rot_x_avg"] = self.ep_kp_rot_x / max(1, self.episode_steps)
            info["physics/kp_rot_y_avg"] = self.ep_kp_rot_y / max(1, self.episode_steps)
            info["physics/kp_rot_z_avg"] = self.ep_kp_rot_z / max(1, self.episode_steps)
            
            
            # Attach to info dict for SB3 to catch
            info['smoothness/avg_cond_num'] = np.mean(self.cond_num_history) if self.cond_num_history else 0
            info['smoothness/max_cond_num'] = np.max(self.cond_num_history) if self.cond_num_history else 0
            info['smoothness/avg_euclidean_jerk'] = np.mean(self.euclidean_jerk_history) if self.euclidean_jerk_history else 0
            info['smoothness/avg_riemannian_jerk'] = np.mean(self.riemannian_jerk_history) if self.riemannian_jerk_history else 0
            info['smoothness/avg_coupling_magnitude'] = np.mean(self.coupling_history) if self.coupling_history else 0
            info['smoothness/max_ang_accel'] = np.max(self.ang_accel_history) if self.ang_accel_history else 0
            info['smoothness/std_force'] = np.std(self.force_history) if self.force_history else 0
            info["smoothness/avg_force"] = np.mean(self.force_history) if self.force_history else 0
            # SPD explosion diagnostics (pre-clamp) — non-zero values confirm eigenvalue explosions
            info['smoothness/spd_pre_clamp_eigmax_avg'] = np.mean(self._spd_pre_clamp_eigmax_history) if self._spd_pre_clamp_eigmax_history else 0
            info['smoothness/spd_pre_clamp_eigmax_peak'] = np.max(self._spd_pre_clamp_eigmax_history) if self._spd_pre_clamp_eigmax_history else 0
            info['smoothness/spd_clamp_violations'] = self._spd_clamp_violations
            info['smoothness/spd_max_offdiag_peak'] = np.max(self._spd_coupling_max_history) if self._spd_coupling_max_history else 0
            # Safety: excessive force and contact engagement
            info["safety/force_exceedance_count"] = self.force_exceedances
            info["safety/force_exceedance_rate"] = self.force_exceedances / max(1, self.episode_steps)
            info["safety/peak_force"] = max(self.force_history) if self.force_history else 0.0
            info["physics/door_contact_rate"] = (
                sum(1 for f in self.force_history if f > 2.0) / max(1, len(self.force_history))
            )

            if self.is_eval and len(self.kp_history) > 0:
                history_array = np.array(self.kp_history)
                eval_kp_avgs = np.mean(history_array, axis=0)

                wandb.log({
                    "eval/kp_trans_x_avg": eval_kp_avgs[0],
                    "eval/kp_trans_y_avg": eval_kp_avgs[1],
                    "eval/kp_trans_z_avg": eval_kp_avgs[2],
                    "eval/kp_rot_x_avg": eval_kp_avgs[3],
                    "eval/kp_rot_y_avg": eval_kp_avgs[4],
                    "eval/kp_rot_z_avg": eval_kp_avgs[5],
                })
            
            # Task-specific metrics: prefer explicit metrics function. If none provided,
            # perform a safe, best-effort detection for Wipe-like environments.
            if self.task_metrics_fn is not None:
                extra_metrics = self.task_metrics_fn(self.env, info)
                if extra_metrics:
                    info.update(extra_metrics)
            else:
                try:
                    raw_env = self.env.unwrapped
                except Exception:
                    raw_env = getattr(self.env, 'env', self.env)

                if hasattr(raw_env, 'num_markers') and hasattr(raw_env, 'wiped_markers'):
                    total_markers = getattr(raw_env, 'num_markers', 0)
                    wiped_markers = getattr(raw_env, 'wiped_markers', [])
                    percent_wiped = len(wiped_markers) / total_markers if total_markers > 0 else 0.0
                    info["physics/raw_wipe_percentage"] = percent_wiped

                    raw_obs = raw_env._get_observations()
                    contact = bool(raw_obs["robot0_contact"])
                    self.episode_contact_steps += int(contact)
                    info["physics/contact_step_ratio"] = self.episode_contact_steps / max(1, self.episode_steps)

                    if self.is_eval and hasattr(self, 'kp_history') and len(self.kp_history) > 0:
                        print(f"Eval | Total Markers: {total_markers} | Wiped Markers: {len(wiped_markers)} | % Wiped: {percent_wiped:.2%}")
                        try:
                            wandb.log({"eval/raw_wipe_percentage": percent_wiped})
                        except Exception:
                            pass
            
    def check_joint_violations(self):
        robot = self.env.unwrapped.robots[0]

        # Check Safety (Joint Limits)
        try:
            if robot.check_q_limits():
                self.violation_count += 1
        except AttributeError:
            # Failsafe just in case
            pass

    def log_contact_forces(self):
        FORCE_THRESHOLD = 50.0  # N — aligned with Wipe env's pressure_threshold_max (60N)
        robot = self.env.unwrapped.robots[0]
        try:
            ee_force = max([
                np.linalg.norm(np.array(robot.recent_ee_forcetorques[arm].current[:3]))
                for arm in robot.arms
            ])
        except Exception as e:
            ee_force = 0.0

        self.force_history.append(ee_force)
        if ee_force > FORCE_THRESHOLD:
            self.force_exceedances += 1
        # self.episode_force_sum += ee_force