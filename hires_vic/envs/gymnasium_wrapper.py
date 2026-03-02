import torch
import math
import gymnasium as gym
import numpy as np
import robosuite as suite
from gymnasium import spaces
from hires_vic.geometry.riemannian import spd_grl_map
from hires_vic.geometry.lie_groups import so3_log_map
from robosuite.utils import transform_utils as T

class RobosuiteGymnasiumWrapper(gym.Env):
    def __init__(self, env_name, robots, controller_configs=None, task_kwargs=None):
        """
        Wraps a Robosuite environment to be compatible with Gymnasium.
        """
        # Ensure task_kwargs is a dictionary
        if task_kwargs is None:
            task_kwargs = {}

        # Default settings (can be overridden by task_kwargs)
        # We use .pop() so we don't pass them twice to suite.make()
        has_renderer = task_kwargs.pop("has_renderer", False)
        has_offscreen_renderer = task_kwargs.pop("has_offscreen_renderer", False)
        use_camera_obs = task_kwargs.pop("use_camera_obs", False)
        use_object_obs = task_kwargs.pop("use_object_obs", True)
        reward_shaping = task_kwargs.pop("reward_shaping", True)
        
        # 1. Load the underlying Robosuite Env
        self.env = suite.make(
            env_name,
            robots=robots,
            controller_configs=controller_configs,
            has_renderer=has_renderer,                   # Use the variable, not hardcoded False
            has_offscreen_renderer=has_offscreen_renderer,
            use_camera_obs=use_camera_obs,
            use_object_obs=use_object_obs,
            reward_shaping=reward_shaping,
            **task_kwargs # Pass any remaining arguments (like horizon, etc.)
        )

        # 2. Define Action Space (Continuous)
        # Robosuite actions are usually [dx, dy, dz, ax, ay, az, gripper]
        # low, high = self.env.action_spec
        self.real_low, self.real_high = self.env.action_spec
        # print(f"Robosuite Action Spec: Low={self.real_low}, High={self.real_high}")
        # self.action_space = spaces.Box(low=low, high=high, dtype=np.float32)
        self.is_grl_controller = controller_configs is not None and "GRL_OSC" in str(controller_configs)
        
        if self.is_grl_controller:
            # GRL Arm always needs exactly 15 dimensions (6 Mandel + 3 RotKp + 3 Pos + 3 Ori)
            # Anything leftover in Robosuite's native action_dim belongs to the gripper
            gripper_dim = self.env.action_dim - 18
            sb3_action_dim = 15 + gripper_dim
            
            self.action_space = spaces.Box(low=-1.0, high=1.0, shape=(sb3_action_dim,), dtype=np.float32)
            
            print("\n" + "="*50)
            print(f"🚀 GRL CONTROLLER DETECTED")
            print(f"▶️ Native Env Dim: {self.env.action_dim} | Gripper Dim: {gripper_dim}")
            print(f"▶️ Overriding SB3 Action Space to {sb3_action_dim}!")
            print("="*50 + "\n")
        else:
            self.action_space = spaces.Box(low=-1.0, high=1.0, shape=(self.env.action_dim,), dtype=np.float32)
            print(f"▶️ Standard Controller Detected: Action space set to {self.env.action_dim}")


        # 3. Define Observation Space
        # We need to run one reset to see the shape of the observations
        obs = self.env.reset()
        # print(f"Robosuite Observation Keys: {obs.keys()}")
        # print(f"Sample Observation Shapes: {{key: np.array(value).shape for key, value in obs.items()}}")
        flat_obs = self._flatten_obs(obs)

        # print('▶️ Observation space shape after flattening: ', flat_obs.shape)

        self.observation_space = spaces.Box(
            low=-np.inf, high=np.inf, shape=flat_obs.shape, dtype=np.float32
        )

    def _flatten_obs(self, obs_dict):
        """
        Flattens the Robosuite dictionary obs into a single vector for the RL agent.
        Selects only the useful keys (proprioception + object state).
        Applies Lie Group Logarithmic Map to orientation if using GRL.
        """
        keys_to_use = ['robot0_eef_pos', 'robot0_eef_quat', 'robot0_gripper_qpos', 'object-state']
        
        values = []
        for key in keys_to_use:
            if key not in obs_dict:
                continue
                
            if key == 'robot0_eef_quat' and getattr(self, 'is_grl_controller', False):
                # --- LIE GROUP OBSERVATION MAPPING ---
                # 1. Get Robosuite's native 4D quaternion (x, y, z, w)
                quat = obs_dict[key]
                
                # 2. Convert Quaternion to 3x3 Rotation Matrix (The Lie Group SO(3))
                rot_mat = T.quat2mat(quat)
                
                # 3. Convert to PyTorch tensor with batch dimension for your map
                rot_tensor = torch.tensor(rot_mat, dtype=torch.float32).unsqueeze(0)
                
                # 4. Apply Log Map to get the 3D axis-angle vector (The Lie Algebra so(3))
                omega = so3_log_map(rot_tensor).squeeze(0).detach().numpy()
                
                values.append(omega)
            else:
                # Standard Euclidean flattening
                values.append(np.array(obs_dict[key]).flatten())
        
        return np.concatenate(values).astype(np.float32)

    def reset(self, seed=None, options=None):
        """
        Gymnasium reset requires a seed and returns (obs, info).
        """
        super().reset(seed=seed)
        # Robosuite's native reset doesn't take a seed directly in the call usually,
        # but we can set numpy's seed if needed.
        if seed is not None:
            np.random.seed(seed)
            
        obs_dict = self.env.reset()
        flat_obs = self._flatten_obs(obs_dict)
        return flat_obs, {}

    # def step(self, action):
    #     """
    #     Gymnasium step returns (obs, reward, terminated, truncated, info).
    #     """
    #     # print(f"Action shape: {action.shape}")  # Debug print to check action values
    #     scaled_action = self.real_low + (0.5 * (action + 1.0) * (self.real_high - self.real_low))
    #     obs_dict, reward, done, info = self.env.step(scaled_action)

    #     raw_success = self.env._check_success()
    #     info["is_success"] = bool(raw_success)
        
    #     flat_obs = self._flatten_obs(obs_dict)
        
    #     # Robosuite returns 'done' as a boolean. 
    #     # In Gymnasium, we split this into 'terminated' (task success/fail) and 'truncated' (timeout).
    #     # Since Robosuite usually handles timeout internally, we can treat done as terminated.
    #     terminated = done
    #     truncated = False # You can add a step counter here if you want strict timeouts
        
    #     return flat_obs, reward, terminated, truncated, info
    
    def step(self, action):
        # action from SB3 is strictly in [-1, 1]
        # print(f'len(action): {len(action)}')  # Debug print to check action values
        if getattr(self, 'is_grl_controller', False):
            mandel_params = action[0:6].copy()
            min_kp, max_kp = 10.0, 200.0
            log_min = math.log(min_kp)
            log_max = math.log(max_kp)
            
            mandel_params[0:3] = log_min + 0.5 * (mandel_params[0:3] + 1.0) * (log_max - log_min)
            # off_diag_scale = (log_max - log_min) / 2.0
            off_diag_scale = 0.2
            mandel_params[3:6] = mandel_params[3:6] * off_diag_scale
            
            mandel_tensor = torch.tensor(mandel_params, dtype=torch.float32).unsqueeze(0)
            Kp_matrix = spd_grl_map(mandel_tensor).squeeze(0).detach().numpy()
            Kp_flat = Kp_matrix.flatten() 
            
            # --- 2. ROTATIONAL STIFFNESS ---
            kp_rot_raw = action[6:9]
            kp_rot_scaled = min_kp + 0.5 * (kp_rot_raw + 1.0) * (max_kp - min_kp)
            
            # --- 3. CONSTRUCT TASK-AGNOSTIC ROBOSUITE ACTION ---
            # action[15:] will safely grab the 1 gripper command for NutAssembly, 
            # or it will return an empty array [] for Wipe!
            robosuite_action = np.concatenate([
                Kp_flat,             # 9 elements
                kp_rot_scaled,       # 3 elements
                action[9:12],        # 3 elements
                action[12:15],       # 3 elements
                action[15:]          # Remaining elements (Gripper, if any)
            ])
            # print(f"GRL Action: {action}")
            # print(f"Mapped Robosuite Action: {robosuite_action}")
            obs_dict, reward, done, info = self.env.step(robosuite_action)
            
        else:
            # Your standard baseline execution
            # print('real high: ', self.real_high)
            # print('real low: ', self.real_low)
            scaled_action = self.real_low + (0.5 * (action + 1.0) * (self.real_high - self.real_low))
            obs_dict, reward, done, info = self.env.step(scaled_action)

        raw_success = self.env._check_success()
        info["is_success"] = bool(raw_success)
        
        flat_obs = self._flatten_obs(obs_dict)
        terminated = done
        truncated = False 
        
        return flat_obs, reward, terminated, truncated, info

    def render(self):
        return self.env.render()

    def close(self):
        self.env.close()


class RobosuitePhysicsWrapper(gym.Wrapper):
    """
    A comprehensive wrapper for Robosuite environments to:
    1. Log physics metrics (Stiffness profile, Contact Forces, Safety Violations).
    2. Apply 'Safety Penalties' (Force & Stiffness) to the reward function.
    
    Args:
        env (gym.Env): The Gym-wrapped Robosuite environment.
        stiffness_penalty (float): Penalty coefficient for high stiffness (e.g., 0.01).
        force_penalty (float): Penalty coefficient for high contact forces (e.g., 0.1).
        max_force_threshold (float): Force limit (Newtons) before penalty kicks in (e.g., 20.0).
        terminate_on_unsafe (bool): If True, ends the episode immediately upon safety violation.
    """
    def __init__(self, env, stiffness_penalty=0.0, force_penalty=0.0, max_force_threshold=30.0, terminate_on_unsafe=False):
        super().__init__(env)
        self.stiffness_penalty = stiffness_penalty
        self.force_penalty = force_penalty
        self.max_force_threshold = max_force_threshold
        self.terminate_on_unsafe = terminate_on_unsafe
        
        # Internal counters for logging
        self.episode_stiffness_sum = 0.0
        self.episode_force_sum = 0.0
        self.episode_steps = 0
        self.violation_count = 0

    def reset(self, **kwargs):
        self.episode_stiffness_sum = 0.0
        self.episode_force_sum = 0.0
        self.episode_steps = 0
        self.violation_count = 0
        return self.env.reset(**kwargs)

    def step(self, action):
        obs, reward, terminated, truncated, info = self.env.step(action)
        self.episode_steps += 1

        # print(f'Length of action {len(action)} | Action received: {action}')  # Debug print to check action values

        # --- A. EXTRACT PHYSICS DATA ---
        # Access the raw Robosuite environment (unwrapped)
        # We need to loop because sometimes there are multiple wrapper layers
        # base_env = self.env.unwrapped
        base_env = self.env.env
        robot = base_env.robots[0]
        # print(self.env.env.unwrapped)  # Debug print to check the base environment
    
        # Get Contact Forces
        try:
            ee_force = max([
                np.linalg.norm(np.array(robot.recent_ee_forcetorques[arm].current[:3]))
                for arm in robot.arms
            ])
        except Exception as e:
            ee_force = 0.0

        # Get Stiffness (Kp) from the FRONT of the array
        action_len = len(action)
        min_kp, max_kp = 10.0, 200.0 # Align this with your config
        
        try:
            if action_len in [15, 16]:
                # --- GRL MODE (16D) ---
                # Layout: Kp_trans_mandel(6), Kp_rot(3), pos(3), ori(3), gripper(1)
                
                # We extract the 3 diagonals from the Mandel parameters (indices 0, 1, 2)
                kp_trans_diags = action[0:3] 
                # We extract the 3 rotational stiffness diagonals (indices 6, 7, 8)
                kp_rot = action[6:9] 
                
                kp_vals = np.concatenate([kp_trans_diags, kp_rot])
                
                stiffness_percentage = np.mean((kp_vals + 1.0) / 2.0)
                physical_stiffness = min_kp + (stiffness_percentage * (max_kp - min_kp))

            elif action_len == 13:
                # Layout: Kp(6), pos(3), ori(3), gripper(1)
                kp_vals = action[0:6]
                
                stiffness_percentage = np.mean((kp_vals + 1.0) / 2.0)
                physical_stiffness = min_kp + (stiffness_percentage * (max_kp - min_kp))

            else:
                # --- FIXED MODE ---
                stiffness_percentage = 0.0
                physical_stiffness = 150.0 

        except Exception as e:
            stiffness_percentage = 0.0 
            physical_stiffness = 0.0

        # 3. Check Safety (Joint Limits)
        is_unsafe = 0
        try:
            if robot.check_q_limits():
                is_unsafe = 1
                self.violation_count += 1
                # print(f"[SAFETY VIOLATION] Joint limit exceeded at step {self.episode_steps}. Total Violations: {self.violation_count}")
        except AttributeError:
            # Failsafe just in case
            pass

        # --- B. APPLY PENALTIES (REWARD MODIFICATION) ---
        
        # 1. Force Penalty (Soft Constraint)
        # "If you push harder than 30N, you lose points"
        force_penalty_val = 0.0
        if self.force_penalty > 0 and ee_force > self.max_force_threshold:
            excess_force = ee_force - self.max_force_threshold
            force_penalty_val = self.force_penalty * excess_force
            reward -= force_penalty_val # Subtract from total reward

        # 2. Stiffness Penalty (Energy Efficiency)
        # "Minimize stiffness unless necessary"
        stiffness_penalty_val = 0.0
        if self.stiffness_penalty > 0:
            stiffness_penalty_val = self.stiffness_penalty * (stiffness_percentage**2)
            reward -= stiffness_penalty_val

        # --- C. LOGGING ---
        # Update cumulative stats
        self.episode_stiffness_sum += physical_stiffness
        self.episode_force_sum += ee_force

        # Log instantaneous metrics (for debugging spikes)
        info["physics/stiffness_step"] = physical_stiffness
        info["physics/force_step"] = ee_force
        info["reward/force_penalty"] = force_penalty_val
        info["reward/stiffness_penalty"] = stiffness_penalty_val
        info["safety/joint_violation"] = is_unsafe

        # Log Episode Averages (Only when episode ends)
        if terminated or truncated:
            avg_stiffness = self.episode_stiffness_sum / max(1, self.episode_steps)
            avg_force = self.episode_force_sum / max(1, self.episode_steps)
            
            info["physics/avg_stiffness"] = avg_stiffness
            info["physics/avg_force"] = avg_force
            info["physics/max_force_violation_count"] = self.violation_count

        return obs, reward, terminated, truncated, info