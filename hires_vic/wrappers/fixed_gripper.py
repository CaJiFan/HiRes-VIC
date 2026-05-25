import gymnasium as gym
import numpy as np

class FixedGripperWrapper(gym.ActionWrapper):
    """
    Hides the gripper action dimension from the RL agent, permanently forcing 
    the gripper closed (+1.0) during the RL control phase.
    """
    def __init__(self, env):
        super().__init__(env)
        
        # 1. Grab the original action space (e.g., 16D)
        old_space = self.env.action_space
        
        # 2. Create a new action space that is exactly 1 dimension smaller (e.g., 15D)
        self.action_space = gym.spaces.Box(
            low=old_space.low[:-1],
            high=old_space.high[:-1],
            dtype=old_space.dtype
        )

        print(f"▶️ New action space with Fixed gripper: {self.action_space.shape}")

    def action(self, act):
        # 3. Intercept the RL agent's action (15D) and append the Closed command
        # +1.0 for Robosuite means Grasp/Close
        return np.concatenate([act, [1.0]], dtype=np.float32)