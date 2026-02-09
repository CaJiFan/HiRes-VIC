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
        
        # 1. Load the underlying Robosuite Env
        self.env = suite.make(
            env_name,
            robots=robots,
            controller_configs=controller_configs,
            has_renderer=has_renderer,                   # Use the variable, not hardcoded False
            has_offscreen_renderer=has_offscreen_renderer,
            use_camera_obs=use_camera_obs,
            use_object_obs=True,
            reward_shaping=True,
            control_freq=20,
            **task_kwargs # Pass any remaining arguments (like horizon, etc.)
        )

        # 2. Define Action Space (Continuous)
        # Robosuite actions are usually [dx, dy, dz, ax, ay, az, gripper]
        low, high = self.env.action_spec
        self.action_space = spaces.Box(low=low, high=high, dtype=np.float32)

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
        obs_dict, reward, done, info = self.env.step(action)
        
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