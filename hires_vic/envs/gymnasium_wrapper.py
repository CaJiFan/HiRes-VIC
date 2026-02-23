import gymnasium as gym
import numpy as np
import robosuite as suite
from gymnasium import spaces

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
        # self.action_space = spaces.Box(low=low, high=high, dtype=np.float32)
        self.action_space = spaces.Box(
            low=-1.0, 
            high=1.0, 
            shape=self.real_low.shape, 
            dtype=np.float32
        )

        # 3. Define Observation Space
        # We need to run one reset to see the shape of the observations
        obs = self.env.reset()
        flat_obs = self._flatten_obs(obs)
        self.observation_space = spaces.Box(
            low=-np.inf, high=np.inf, shape=flat_obs.shape, dtype=np.float32
        )

    def _flatten_obs(self, obs_dict):
        """
        Flattens the Robosuite dictionary obs into a single vector for the RL agent.
        Selects only the useful keys (proprioception + object state).
        """
        keys_to_use = ['robot0_eef_pos', 'robot0_eef_quat', 'robot0_gripper_qpos', 'object-state']
        
        # Note: 'object-state' might be named differently depending on the task (e.g. 'door_pos')
        # Check obs_dict.keys() if you switch tasks.
        values = []
        for key in keys_to_use:
            if key in obs_dict:
                values.append(np.array(obs_dict[key]).flatten())
            else:
                # Fallback for task-specific keys if strictly needed
                pass 
        
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

    def step(self, action):
        """
        Gymnasium step returns (obs, reward, terminated, truncated, info).
        """
        # print(f"Action shape: {action.shape}")  # Debug print to check action values
        scaled_action = self.real_low + (0.5 * (action + 1.0) * (self.real_high - self.real_low))
        obs_dict, reward, done, info = self.env.step(scaled_action)
        
        flat_obs = self._flatten_obs(obs_dict)
        
        # Robosuite returns 'done' as a boolean. 
        # In Gymnasium, we split this into 'terminated' (task success/fail) and 'truncated' (timeout).
        # Since Robosuite usually handles timeout internally, we can treat done as terminated.
        terminated = done
        truncated = False # You can add a step counter here if you want strict timeouts
        
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
        # 1. Step the environment
        obs, reward, terminated, truncated, info = self.env.step(action)
        self.episode_steps += 1

        # --- A. EXTRACT PHYSICS DATA ---
        # Access the raw Robosuite environment (unwrapped)
        # We need to loop because sometimes there are multiple wrapper layers
        # base_env = self.env.unwrapped
        base_env = self.env.env
        robot = base_env.robots[0]
        # print(self.env.env.unwrapped)  # Debug print to check the base environment
        
        # 1. Get Stiffness (Kp)
        # Assumption: OSC_POSE controller with variable_kp.
        # Action structure is usually [pos(3), ori(3), stiffness(6), gripper(1)]
        # We take the mean of the 6 stiffness values (indices 6 to 12)
        try:
            kp_vals = action[0:6]
            # Map [-1, 1] action to actual Kp scale if needed, but raw action is fine for trends
            # current_stiffness = np.mean(np.abs(kp_vals)) 
            stiffness_percentage = np.mean((kp_vals + 1.0) / 2.0)
            min_kp, max_kp = 10.0, 200.0
            physical_stiffness = min_kp + (stiffness_percentage * (max_kp - min_kp))
        except IndexError:
            stiffness_percentage = 0.0 
            physical_stiffness = 0.0 

        # 2. Get Contact Forces
        # Robosuite robots have a property 'ee_force' (Fz) and 'ee_torque'
        # Norm of the 3D force vector at the end-effector
        try:
            ee_force = max([
                np.linalg.norm(np.array(robot.recent_ee_forcetorques[arm].current[:3]))
                for arm in robot.arms
            ])
        except Exception as e:
            ee_force = 0.0

        # if self.episode_steps % 100 == 0:  # Debug print for the first 10 steps and then every 50 steps
        #     print(f"Raw action received in PhysicsWrapper: {action}")  # Debug print to check action valuesw force values
        #     print(f"[DEBUG WRAPPER] Step: {self.episode_steps} | "
        #           f"Kp Action Mean: {stiffness_percentage:.2f} | "
        #           f"EE Force: {ee_force:.2f} N | "
        #           f"Joint Limits Check: {base_env.check_robot_join_limits():.2f} N | "
        #           f"Base Reward: {reward:.3f}")
            

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