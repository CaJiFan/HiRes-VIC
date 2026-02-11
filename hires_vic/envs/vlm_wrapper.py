import numpy as np
import gymnasium as gym
from gymnasium import spaces

class VLMSimulatorWrapper(gym.ObservationWrapper):
    def __init__(self, env, noise_std=0.02):
        """
        :param env: The Gym environment
        :param noise_std: Standard deviation of VLM error (e.g., 2cm)
        """
        super().__init__(env)
        self.noise_std = noise_std
        self.current_goal_estimate = None
        
        # 1. Update Observation Space
        # We append 3 numbers (x, y, z) to the existing observation
        low = np.concatenate([env.observation_space.low, [-np.inf]*3])
        high = np.concatenate([env.observation_space.high, [np.inf]*3])
        self.observation_space = spaces.Box(low=low, high=high, dtype=np.float32)

    def reset(self, **kwargs):
        obs, info = self.env.reset(**kwargs)
        
        # 2. Get Ground Truth (Cheating from Simulator)
        # In Robosuite, we can access the underlying sim to find the door handle
        # Note: You might need to adjust 'door_handle_pos' key based on exact env
        true_pos = self.unwrapped.env.sim.data.body_xpos[self.unwrapped.env.door_handle_body_id]
        
        # 3. Add "VLM Noise"
        # We generate a noise vector ONCE per episode (Simulating one VLM snapshot)
        noise = np.random.normal(0, self.noise_std, size=3)
        self.current_goal_estimate = true_pos + noise
        
        # 4. Attach to Observation
        return np.concatenate([obs, self.current_goal_estimate]), info

    def observation(self, obs):
        # 5. Keep the goal constant during the episode
        return np.concatenate([obs, self.current_goal_estimate])