import gymnasium as gym
import numpy as np

class ActionNormalizerWrapper(gym.ActionWrapper):
    """
    Normalizes actions from [-1, 1] to the environment's action_spec range.
    Assumes the wrapped env has a Box action space.
    """
    def __init__(self, env):
        super().__init__(env)
        self.action_low = self.env.action_space.low
        self.action_high = self.env.action_space.high
        # Normalize to [-1, 1] for the wrapper's action space
        self.action_space = gym.spaces.Box(low=-1, high=1, shape=self.env.action_space.shape, dtype=np.float32)
    
    def action(self, action):
        # Scale from [-1, 1] to [low, high]
        print(f"Original Action: {action}")
        scaled_action = self.action_low + (action + 1) * (self.action_high - self.action_low) / 2
        print(f"Scaled Action: {scaled_action}")
        return scaled_action