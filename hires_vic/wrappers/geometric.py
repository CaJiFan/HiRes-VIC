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
    ):
        super().__init__(env)

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

        self.use_llm_prior = kwargs.get("use_llm_prior", False)
        self.llm_planner = None
        if self.use_llm_prior:
            from hires_vic.llm.impedance_planner import LLMImpedancePlanner
            self.llm_planner = LLMImpedancePlanner(
                query_every_n_steps=kwargs.get("llm_query_interval", 50),
                prior_weight=kwargs.get("llm_prior_weight", 0.4),
            )

        print(f"""
            🔧 Robosuite Wrapper Initialized
            | SPD: {self.use_spd_manifold}
            | Lie Group: {self.use_lie_group}
            | Diag Manifold: {self.use_diag_manifold}
            | Fixed: {self.use_fixed}
        """)

        action_dim = 6 if self.use_fixed else 12
        action_dim += 3 if self.use_spd_manifold else 0
        
        gripper_dim = self.env.action_dim - (18 if self.use_spd_manifold else 6 if self.use_fixed else 12)
        if gripper_dim > 0:
            action_dim += gripper_dim

        self.action_space = spaces.Box(
            low=-1.0, high=1.0, shape=(action_dim,), dtype=np.float32
        )
        print(f"▶️ Action space strictly set to [-1, 1] with shape: {self.action_space.shape}")

        # Observation space 
        obs = self.env.reset()
        flat_obs = self._flatten_obs(obs)

        print('▶️ Observation space shape after flattening: ', flat_obs.shape)

        self.observation_space = spaces.Box(
            low=-np.inf, high=np.inf, shape=flat_obs.shape, dtype=np.float32
        )

    def _flatten_obs(self, obs):
        """
        Flattens the Robosuite dictionary into a single vector for the RL agent.
        Selects only the useful keys (proprioception + object state).
        Applies Lie Group Logarithmic Map to orientation if using GRL.
        """
        raw_obs = self.env.unwrapped._get_observations()
        proprio_state_keys = [
            'robot0_joint_pos', 'robot0_joint_pos_cos', 'robot0_joint_pos_sin',
            'robot0_joint_vel', 'robot0_joint_acc', 'robot0_eef_pos', 'robot0_eef_quat', 
            'robot0_eef_quat_site', 'robot0_gripper_qpos', 'robot0_gripper_qvel', 'robot0_contact'
        ]
        # print(f'initial obs ({len(obs[0])}) {obs}')
        
        object_state = np.array(raw_obs['object-state']).flatten()
        proprio_state = []
        
        for key in proprio_state_keys:
            if 'quat' in key and self.use_lie_group:
                quat = raw_obs[key]
                rot_mat = T.quat2mat(quat)
                rot_tensor = torch.tensor(rot_mat, dtype=torch.float32).unsqueeze(0)
                
                # log map: SO(3) Manifold -> so(3) TxM Lie Algebra
                omega = so3_log_map(rot_tensor).squeeze(0).detach().numpy() 
                proprio_state.append(omega)
            else:
                proprio_state.append(np.array(raw_obs[key]).flatten())
        

        return np.concatenate([object_state] + proprio_state).astype(np.float32)

    def reset(self, seed=None, options=None):
        super().reset(seed=seed)
        # Robosuite's native reset doesn't take a seed directly in the call usually,
        # but we can set numpy's seed if needed.
        if seed is not None:
            np.random.seed(seed)
            
        obs = self.env.reset()
        flat_obs = self._flatten_obs(obs)

        if self.llm_planner is not None:
            self.llm_planner.reset()

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
        return flat_obs, {}

    def step(self, action):
        if self.llm_planner is not None:
            suggestion = self.llm_planner.query(self._last_obs_dict)
            # Blend RL action with LLM prior in log-space (for SPD manifold part)
            # action[:6] = Mandel log-space params (if use_spd_manifold)
            w = suggestion.confidence
            action = action.copy()
            action[:6] = (1 - w) * action[:6] + w * suggestion.log_kp_prior
            action[6:9] = (1 - w) * action[6:9] + w * suggestion.kp_rot_prior
            # Log mode for analysis
            self._current_llm_mode = suggestion.mode


        if self.use_spd_manifold:
            mandel_params = action[:6].copy()

            log_min, log_max = math.log(self.min_kp), math.log(self.max_kp)
            mandel_params[0:3] = log_min + 0.5 * (mandel_params[0:3] + 1.0) * (log_max - log_min)
            mandel_params[3:6] = mandel_params[3:6] * 0.2
            
            mandel_tensor = torch.tensor(mandel_params, dtype=torch.float32).unsqueeze(0)
            kp_matrix = spd_grl_map(mandel_tensor).squeeze(0).detach().numpy() 

            kp_rot_raw = action[6:9]
            kp_rot_scaled = self.min_kp + 0.5 * (kp_rot_raw + 1.0) * (self.max_kp - self.min_kp)
            
            robosuite_action = np.concatenate([
                kp_matrix.flatten(),    # 9 elements: 3x3 matrix flattened
                kp_rot_scaled,          # Scaled rotational stiffness
                action[9:],             # pos + ori + gripper
            ])
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
            low, high = self.env.action_space.low, self.env.action_space.high
            robosuite_action = low + 0.5 * (action + 1.0) * (high - low)

            robot = self.env.unwrapped.robots[0]
            fixed_kp = robot.composite_controller.part_controller_config['right']['kp']
            physical_kp_vals = np.ones(6) * fixed_kp
            kp_matrix_3x3 = np.diag(physical_kp_vals[:3])
        else:
            # Standard baseline execution
            low, high = self.env.action_space.low, self.env.action_space.high
            robosuite_action = low + 0.5 * (action + 1.0) * (high - low)
            physical_kp_vals = robosuite_action[:6]
            kp_matrix_3x3 = np.diag(physical_kp_vals[:3])

        step_return = self.env.step(robosuite_action)
        self.episode_steps += 1
       
        obs, reward, terminated, truncated, info = step_return

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

        return self._flatten_obs(obs), reward, terminated, truncated, info
    
    def log_info(self, physical_kp_vals, step_return):
        obs, reward, terminated, truncated, info = step_return

        if self.llm_planner is not None:
            info["llm/impedance_mode"] = self._current_llm_mode
            info["llm/prior_confidence"] = self.llm_planner.prior_weight

        raw_success = self.env._check_success()
        info["is_success"] = bool(raw_success)

        if self.is_eval:
            self.kp_history.append(physical_kp_vals.copy())
        
        raw_obs = self.env.unwrapped._get_observations()
        contact = bool(raw_obs["robot0_contact"])
        self.episode_contact_steps += int(contact)

        self.log_contact_forces()
        self.check_joint_violations()

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
            info["physics/contact_step_ratio"] = self.episode_contact_steps / max(1, self.episode_steps)
            
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
                    # "eval/contact_step_ratio": self.episode_contact_steps / max(1, self.episode_steps)
                })
            
            if hasattr(self.env.env, 'num_markers') and hasattr(self.env.env, 'wiped_markers'):
                total_markers = self.env.env.num_markers
                wiped_markers = len(self.env.env.wiped_markers)
                percent_wiped = wiped_markers / total_markers if total_markers > 0 else 0
                info["physics/raw_wipe_percentage"] = percent_wiped
                
                if self.is_eval and len(self.kp_history) > 0:
                    print(f"Eval | Total Markers: {total_markers} | Wiped Markers: {wiped_markers} | % Wiped: {percent_wiped:.2%}")
                    wandb.log({"eval/raw_wipe_percentage": percent_wiped})
            
    def check_joint_violations(self):
        robot = self.env.robots[0]

        # Check Safety (Joint Limits)
        try:
            if robot.check_q_limits():
                self.violation_count += 1
        except AttributeError:
            # Failsafe just in case
            pass

    def log_contact_forces(self):
        robot = self.env.robots[0]
        try:
            ee_force = max([
                np.linalg.norm(np.array(robot.recent_ee_forcetorques[arm].current[:3]))
                for arm in robot.arms
            ])
        except Exception as e:
            ee_force = 0.0

        self.force_history.append(ee_force)
        # self.episode_force_sum += ee_force