from robosuite.environments.manipulation.wipe import Wipe
import numpy as np

class DeltaWipe(Wipe):
    """
    A custom Wipe environment that uses Potential-Based Reward Shaping.
    Bypasses Robosuite's absolute distance reward and replaces it with a Delta.
    """
    def __init__(self, **kwargs):
        super().__init__(**kwargs)
        self.prev_potential = 0.0
        
        # STEAL THE MULTIPLIER: Save the true value, but tell Robosuite's 
        # base reward function to use 0.0 so it stops giving "hovering" points.
        self.true_distance_multiplier = self.distance_multiplier
        self.distance_multiplier = 0.0 

    def _compute_potential(self):
        """Calculates the exact distance reward Robosuite normally uses."""
        if len(self.wiped_markers) >= self.num_markers:
            return 0.0
            
        _, _, mean_pos_to_things_to_wipe = self._get_wipe_information()
        mean_dist = np.linalg.norm(mean_pos_to_things_to_wipe)
        
        # Calculate the raw potential
        potential = self.true_distance_multiplier * (1 - np.tanh(self.distance_th_multiplier * mean_dist))
        
        # Apply Robosuite's internal scaling logic (from the bottom of their reward function)
        if self.reward_scale:
            potential *= self.reward_scale * self.reward_normalization_factor
            
        return potential

    def reset(self):
        obs = super().reset()
        # Initialize the potential at the start of the episode
        self.prev_potential = self._compute_potential()
        return obs

    def reward(self, action=None):
        # 1. Get the base reward. Because we set self.distance_multiplier = 0,
        # this safely includes wipes, contacts, and penalties, but NO hovering points!
        base_reward = super().reward(action)
        
        # 2. Calculate our Potential-Based Delta
        current_potential = self._compute_potential()
        delta_reward = current_potential - self.prev_potential
        self.prev_potential = current_potential
        
        # 3. Combine for the final, Safe RL aligned reward
        return base_reward + delta_reward