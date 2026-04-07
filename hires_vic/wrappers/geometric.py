import torch
import math
import gymnasium as gym
import numpy as np
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

        if self.is_eval:
            self.kp_history = []
        
        # Internal counters for logging
        self.episode_force_sum = 0.0
        self.episode_steps = 0
        self.violation_count = 0

        self.ep_kp_trans_x = 0.0
        self.ep_kp_trans_y = 0.0
        self.ep_kp_trans_z = 0.0
        self.ep_kp_rot_x = 0.0
        self.ep_kp_rot_y = 0.0
        self.ep_kp_rot_z = 0.0

        self.min_kp, self.max_kp  = (1, 300)

        print(f"""
            🔧 Robosuite Wrapper Initialized \
            | SPD: {self.use_spd_manifold} \
            | Lie Group: {self.use_lie_group} \
            | Diag Manifold: {self.use_diag_manifold}
        """)

        action_dim = 3 if self.use_fixed else 6
        action_dim += 6 if self.use_spd_manifold else 3
        
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
        # ✅ THE FIX: Bypass the GymWrapper and grab the raw dictionary directly
        raw_obs = self.env.unwrapped._get_observations()
        # print(raw_obs.keys(), obs[0].shape)
        keys_to_use = ['robot0_eef_pos', 'robot0_eef_quat', 'robot0_gripper_qpos', 'object-state']
        
        values = []
        for key in keys_to_use:
            # We now check against raw_obs instead of the flattened argument
            if key not in raw_obs:
                raise KeyError(f"CRITICAL: Required observation key '{key}' is missing from the environment.")
            
            # --- LIE GROUP OBSERVATION MAPPING ---
            if key == 'robot0_eef_quat' and self.use_lie_group:
                quat = raw_obs[key]
                rot_mat = T.quat2mat(quat) # Assuming T is robosuite.utils.transform_utils
                rot_tensor = torch.tensor(rot_mat, dtype=torch.float32).unsqueeze(0)
                
                # log map: SO(3) Manifold -> so(3) TxM Lie Algebra
                omega = so3_log_map(rot_tensor).squeeze(0).detach().numpy() 
                
                values.append(omega)
            else:
                # Standard Euclidean flattening
                values.append(np.array(raw_obs[key]).flatten())
        
        return np.concatenate(values).astype(np.float32)

    def reset(self, seed=None, options=None):
        super().reset(seed=seed)
        # Robosuite's native reset doesn't take a seed directly in the call usually,
        # but we can set numpy's seed if needed.
        if seed is not None:
            np.random.seed(seed)
            
        obs = self.env.reset()
        flat_obs = self._flatten_obs(obs)
        
        self.episode_force_sum = 0.0
        self.episode_steps = 0
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
            
        else:
            # Standard baseline execution
            low, high = self.env.action_space.low, self.env.action_space.high
            robosuite_action = low + 0.5 * (action + 1.0) * (high - low)
            physical_kp_vals = robosuite_action[:6]

        obs, reward, terminated, truncated, info = self.env.step(robosuite_action)
        self.episode_steps += 1

        raw_success = self.env._check_success()
        info["is_success"] = bool(raw_success)
        
        flat_obs = self._flatten_obs(obs)

        # gymwrapper_env = self.env
        robot = self.env.robots[0]
    
        # # Get Contact Forces
        try:
            ee_force = max([
                np.linalg.norm(np.array(robot.recent_ee_forcetorques[arm].current[:3]))
                for arm in robot.arms
            ])
        except Exception as e:
            ee_force = 0.0


        # Check Safety (Joint Limits)
        try:
            if robot.check_q_limits():
                self.violation_count += 1
                # print(f"[SAFETY VIOLATION] Joint limit exceeded at step {self.episode_steps}. Total Violations: {self.violation_count}")
        except AttributeError:
            # Failsafe just in case
            pass

        if self.is_eval:
            self.kp_history.append(physical_kp_vals.copy())

        # LOGGING 
        self.episode_force_sum += ee_force
        self.ep_kp_trans_x += physical_kp_vals[0]
        self.ep_kp_trans_y += physical_kp_vals[1]
        self.ep_kp_trans_z += physical_kp_vals[2]
        self.ep_kp_rot_x += physical_kp_vals[3]
        self.ep_kp_rot_y += physical_kp_vals[4]
        self.ep_kp_rot_z += physical_kp_vals[5]

        # Log Episode Averages (Only when episode ends)
        if terminated or truncated:
            # Makes it robust to other envs besides Wipe
            if hasattr(self.env.env, 'num_markers') and hasattr(self.env.env, 'wiped_markers'):
                total_markers = self.env.env.num_markers
                wiped_markers = len(self.env.env.wiped_markers)
                percent_wiped = wiped_markers / total_markers if total_markers > 0 else 0
                info["physics/raw_wipe_percentage"] = percent_wiped
                
                if self.is_eval and len(self.kp_history) > 0:
                    print(f"Eval | Total Markers: {total_markers} | Wiped Markers: {wiped_markers} | % Wiped: {percent_wiped:.2%}")
                    wandb.log({"eval/raw_wipe_percentage": percent_wiped})

            if self.is_eval and len(self.kp_history) > 0:
                history_array = np.array(self.kp_history)
                eval_kp_avgs = np.mean(history_array, axis=0)

                wandb.log({
                    "eval/kp_trans_x_avg": eval_kp_avgs[0],
                    "eval/kp_trans_y_avg": eval_kp_avgs[1],
                    "eval/kp_trans_z_avg": eval_kp_avgs[2],
                    "eval/kp_rot_x_avg": eval_kp_avgs[3],
                    "eval/kp_rot_y_avg": eval_kp_avgs[4],
                    "eval/kp_rot_z_avg": eval_kp_avgs[5]
                })
            
            info["physics/joint_violation_count"] = self.violation_count
            info["physics/kp_trans_x_avg"] = self.ep_kp_trans_x / max(1, self.episode_steps)
            info["physics/kp_trans_y_avg"] = self.ep_kp_trans_y / max(1, self.episode_steps)
            info["physics/kp_trans_z_avg"] = self.ep_kp_trans_z / max(1, self.episode_steps)
            info["physics/kp_rot_x_avg"] = self.ep_kp_rot_x / max(1, self.episode_steps)
            info["physics/kp_rot_y_avg"] = self.ep_kp_rot_y / max(1, self.episode_steps)
            info["physics/kp_rot_z_avg"] = self.ep_kp_rot_z / max(1, self.episode_steps)
            info["physics/avg_force"] = self.episode_force_sum / max(1, self.episode_steps)

        # return obs, reward, terminated, truncated, info
        return flat_obs, reward, terminated, truncated, info