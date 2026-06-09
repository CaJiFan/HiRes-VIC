import torch
import math
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
        force_penalty=0.0,
        terminate_on_unsafe=False,
        use_llm_prior=False,
        llm_backend="ollama",
        llm_model="llama3.2",
        llm_query_interval=50,
        llm_prior_weight=0.4,
        llm_profile_path=None,
        task_type=None,
        task_metrics_fn=None,
        use_vision=False,
    ):
        super().__init__(env)

        self.gym_env = self.env.env
        self.teleport_wrapper = self.env
        self.use_spd_manifold = use_spd_manifold
        self.use_lie_group = use_lie_group
        self.use_diag_manifold = use_diag_manifold
        self.use_fixed = use_fixed
        self.stiffness_penalty = stiffness_penalty
        self.force_penalty = force_penalty
        self.terminate_on_unsafe = terminate_on_unsafe
        self.is_eval = is_eval

        self.dt = 1/20 # Assuming 20 Hz control frequency
        self.prev_Kp = None
        self.prev_ang_vel = None

        self.cond_num_history = []
        self.euclidean_jerk_history = []
        self.riemannian_jerk_history = []
        self.coupling_history = []
        self.ang_accel_history = []
        self.force_history = []

        if self.is_eval:
            self.kp_history = []
        
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
                image_size=(224, 224)
            )

            self._current_llm_mode = 'align'
        
        # Camera buffers for VLM
        self.last_wrist_image = None
        self.last_view_image = None

        self._last_obs_dict = {}

        print(f"""
            🔧 Robosuite Wrapper Initialized
            | SPD: {self.use_spd_manifold}
            | Lie Group: {self.use_lie_group}
            | Diag Manifold: {self.use_diag_manifold}
            | Fixed: {self.use_fixed}
        """)


        # action_dim = 6 if self.use_fixed else 12
        # action_dim += 3 if self.use_spd_manifold else 0
        # action_dim = 9 if self.use_spd_manifold else 6

        self.prior_dim = 6 if self.use_spd_manifold else 3
        action_dim = self.prior_dim
        
        # gripper_dim = self.env.env.action_dim - (18 if self.use_spd_manifold else 6 if self.use_fixed else 12)
        # gripper_dim = 0 if 'wipe' in self.env.unwrapped.__class__.__name__.lower() else 1
        # if gripper_dim > 0:
        #     action_dim += gripper_dim

        self.action_space = spaces.Box(
            low=-1.0, high=1.0, shape=(action_dim,), dtype=np.float32
        )
        print(f"▶️ Action space strictly set to [-1, 1] with shape: {self.action_space.shape}")

        # --- TRUE RESIDUAL RL: STATE TRACKERS ---
        
        self.extra_obs_dim = self.prior_dim + 1 # +1 for the confidence weight 'w'
        
        self.current_prior = np.zeros(self.prior_dim, dtype=np.float32)
        self.current_w = 0.0

        # Observation space expansion
        obs = self.env.reset()
        flat_obs = self._flatten_obs(obs)
        
        expanded_obs_shape = (flat_obs.shape[0] + self.extra_obs_dim,)

        print('▶️ Base Observation space shape: ', flat_obs.shape)
        print('▶️ Expanded Residual Observation space shape: ', expanded_obs_shape)

        self.observation_space = spaces.Box(
            low=-np.inf, high=np.inf, shape=expanded_obs_shape, dtype=np.float32
        )

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

        return np.concatenate(new_obs).astype(np.float32)

    def reset(self, seed=None, options=None):
        super().reset(seed=seed)
        # Robosuite's native reset doesn't take a seed directly in the call usually,
        # but we can set numpy's seed if needed.
        if seed is not None:
            np.random.seed(seed)
            
        # obs = self.env.reset()
        obs = self.gym_env.reset()
        flat_obs = self._flatten_obs(obs)

        if self.llm_planner is not None:
            self.llm_planner.reset()
            # ✅ FIX: Initialize prior from planner's default suggestion
            self.current_prior = self.llm_planner._last.action_prior.copy()[:self.prior_dim]
            self.current_w = self.llm_planner._last.confidence
        else:
            self.current_prior = np.zeros(self.prior_dim, dtype=np.float32)
            self.current_w = 0.0
        
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

        if self.is_eval:
            self.kp_history.clear()
        
        # Clear image buffers
        self.last_wrist_image = None
        self.last_view_image = None
        
        return flat_obs, {}

    def step(self, rl_action):
        # if self.llm_planner is not None:
        #     suggestion = self.llm_planner.query(self._last_obs_dict)
        #     # Blend RL action with LLM prior entirely in normalized [-1,1] space.
        #     # action_prior covers the first 9 dims: [6 Mandel params, 3 kp_rot].
        #     w = suggestion.confidence
        #     action = action.copy()
        #     n = len(suggestion.action_prior)  # 9 for use_spd_manifold
        #     action[:n] = (1 - w) * action[:n] + w * suggestion.action_prior
        #     self._current_llm_mode = suggestion.mode

        # --- TRUE RESIDUAL RL: BLEND USING VISIBLE STATE ---
        # The agent outputted `rl_action` based on seeing `self.current_prior`.
        # We blend them here BEFORE calculating the Riemannian math.
        action = rl_action.copy()
        if self.llm_planner is not None:
            # print('prior_dim:', self.prior_dim, self.current_prior.shape, action.shape)
            action[:self.prior_dim] = (1.0 - self.current_w) * action[:self.prior_dim] + (self.current_w * self.current_prior)

        if self.use_spd_manifold:
            mandel_params = action[:6].copy()

            log_min, log_max = math.log(self.min_kp), math.log(self.max_kp)
            mandel_params[0:3] = log_min + 0.5 * (mandel_params[0:3] + 1.0) * (log_max - log_min)
            mandel_params[3:6] = mandel_params[3:6] * 0.2
            
            mandel_tensor = torch.tensor(mandel_params, dtype=torch.float32).unsqueeze(0)
            kp_matrix = spd_grl_map(mandel_tensor).squeeze(0).detach().numpy() 

            kp_rot_raw = action[6:9]
            kp_rot_scaled = self.min_kp + 0.5 * (kp_rot_raw + 1.0) * (self.max_kp - self.min_kp)

            kp_rot_scaled = np.array([300.0, 300.0, 300.0], dtype=np.float32)
            trajectory_command = np.array([0.0, 0.0, -0.025, 0.0, 0.0, 0.0], dtype=np.float32)
            gripper_command = np.array([1.0], dtype=np.float32)
            
            robosuite_action = np.concatenate([
                kp_matrix.flatten(),    # 9 elements: 3x3 matrix flattened
                kp_rot_scaled,          # Scaled rotational stiffness
                trajectory_command,     # Override pos+ori with a fixed downward trajectory for testing
                gripper_command         # Override gripper command to always closed for testing
            ])

            # robosuite_action = np.concatenate([
            #     kp_matrix.flatten(),    # 9 elements: 3x3 matrix flattened
            #     kp_rot_scaled,          # Scaled rotational stiffness
            #     action[9:],             # pos + ori + gripper
            # ])

            physical_kp_vals = np.concatenate([np.diag(kp_matrix), kp_rot_scaled])
            kp_matrix_3x3 = kp_matrix
        elif self.use_diag_manifold:
            kp_raw = action[:6].copy()
            log_min, log_max = np.log(self.min_kp), np.log(self.max_kp)
            
            # Linearly map [-1, 1] to logarithmic space
            log_kp = log_min + 0.5 * (kp_raw + 1.0) * (log_max - log_min)
            
            # The Scalar Exponential Map (Flattens the SPD manifold to a diagonal)
            kp_scaled = np.exp(log_kp)
            
            robosuite_action = np.concatenate([
                kp_scaled,      # 6 elements (Trans + Rot Kp)
                action[6:],     # The rest of the action space (pos, ori, gripper)
            ])
            physical_kp_vals = kp_scaled
            kp_matrix_3x3 = np.diag(kp_scaled[:3])
        elif self.use_fixed:
            # ✅ FIXED BASELINE MATH
            low, high = self.gym_env.action_space.low, self.gym_env.action_space.high
            robosuite_action = low + 0.5 * (action + 1.0) * (high - low)

            robot = self.gym_env.unwrapped.robots[0]
            fixed_kp = robot.composite_controller.part_controller_config['right']['kp']
            physical_kp_vals = np.ones(6) * fixed_kp
            kp_matrix_3x3 = np.diag(physical_kp_vals[:3])
        else:
            # Standard baseline execution
            low, high = self.gym_env.action_space.low, self.gym_env.action_space.high
            kp_trans_scaled = low[:self.prior_dim] + 0.5 * (action + 1.0) * (high[:self.prior_dim] - low[:self.prior_dim])
            # physical_kp_vals = robosuite_action[:self.prior_dim]
            
            kp_rot_scaled = np.array([300.0, 300.0, 300.0], dtype=np.float32)
            trajectory_command = np.array([0.0, 0.0, -0.05, 0.0, 0.0, 0.0], dtype=np.float32)
            gripper_command = np.array([1.0], dtype=np.float32)
            
            robosuite_action = np.concatenate([
                kp_trans_scaled,    # 9 elements: 3x3 matrix flattened
                kp_rot_scaled,          # Scaled rotational stiffness
                trajectory_command,     # Override pos+ori with a fixed downward trajectory for testing
                gripper_command         # Override gripper command to always closed for testing
            ])
            
            # print(kp_trans_scaled.shape, kp_rot_scaled.shape)
            physical_kp_vals = np.concatenate([kp_trans_scaled, kp_rot_scaled])
            kp_matrix_3x3 = np.diag(physical_kp_vals[:3])

            # print(robosuite_action, physical_kp_vals)

        
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

        self.prev_Kp = safe_kp.copy()

        # Peak Angular Acceleration (Lie Group Stability)
        # Extract current angular velocity from Robosuite's proprioception
        # Adjust 'robot0_eef_vel_ang' if your observation key is slightly different
        robot = self.env.unwrapped.robots[0]
        current_ang_vel = robot._hand_ang_vel["right"]
        ang_accel = np.linalg.norm((current_ang_vel - self.prev_ang_vel) / self.dt)
        self.ang_accel_history.append(ang_accel)
        self.prev_ang_vel = current_ang_vel.copy()

        self.log_info(physical_kp_vals, step_return)

        return flat_obs, reward, terminated, truncated, info
    
    def log_info(self, physical_kp_vals, step_return):
        obs, reward, terminated, truncated, info = step_return

        if self.llm_planner is not None:
            # print(f"📊 Step {self.episode_steps} | LLM Mode: {self._current_llm_mode} | Prior Confidence: {self.llm_planner.prior_weight:.2f}")
            info["llm/impedance_mode"] = self._current_llm_mode
            info["llm/prior_confidence"] = self.llm_planner.prior_weight

        raw_success = self.gym_env._check_success()
        info["is_success"] = bool(raw_success)

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
        robot = self.gym_env.robots[0]

        # Check Safety (Joint Limits)
        try:
            if robot.check_q_limits():
                self.violation_count += 1
        except AttributeError:
            # Failsafe just in case
            pass

    def log_contact_forces(self):
        robot = self.gym_env.robots[0]
        try:
            ee_force = max([
                np.linalg.norm(np.array(robot.recent_ee_forcetorques[arm].current[:3]))
                for arm in robot.arms
            ])
        except Exception as e:
            ee_force = 0.0

        self.force_history.append(ee_force)
        # self.episode_force_sum += ee_force