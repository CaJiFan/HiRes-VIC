import torch
import gym
from geometry import so3_exp_map, so3_log_map, spd_cholesky_map

class GeometricResidualWrapper(gym.ActionWrapper):
    """
    Wraps the environment to handle Geometric Inputs and Outputs.
    
    State Space:
        - Instead of (quat_curr, quat_target), we return the Geodesic Error in Tangent Space.
        - Ref: "The distance is the norm of the orientation difference in the tangent space".
        
    Action Space:
        - The RL agent outputs a flat vector of size 12 (3 pos + 3 ori + 6 stiff).
        - We interpret this vector using Exp maps and Cholesky maps.
    """
    
    def __init__(self, env):
        super().__init__(env)
        
    def observation(self, obs):
        """
        Assume obs contains ['current_pose', 'target_pose'] as matrices (3,3) or quats.
        We convert the orientation difference to the Lie Algebra.
        """
        R_curr = obs['current_rot'] # (3, 3)
        R_targ = obs['target_rot']  # (3, 3)
        
        # 1. Compute Relative Rotation: R_diff = R_curr.T @ R_targ
        # "The difference between two orientations can be computed as Log(y^-1 x)".
        R_diff = torch.bmm(R_curr.transpose(1, 2), R_targ)
        
        # 2. Map to Tangent Space (3D vector)
        geo_error_ori = so3_log_map(R_diff)
        
        # 3. Positional Error
        pos_error = obs['target_pos'] - obs['current_pos']
        
        # Return concatenated "Linearized" State
        # This vector is what the Neural Network sees.
        return torch.cat([pos_error, geo_error_ori], dim=-1)

    def action(self, action):
        """
        Process the 'flat' action from the RL agent into geometric control commands.
        action: (12,) -> [delta_pos(3), delta_ori_tangent(3), stiffness_params(6)]
        """
        action = torch.as_tensor(action).unsqueeze(0) # Batchify
        
        # Split action heads
        delta_pos = action[:, 0:3]
        delta_ori_tangent = action[:, 3:6] # This is "epsilon_tau" from the paper
        stiffness_params = action[:, 6:12]
        
        # 1. Process Orientation (Lie Group Approach)
        # Convert tangent twist to rotation matrix update
        R_update = so3_exp_map(delta_ori_tangent)
        
        # Apply update: R_new = R_curr @ R_update
        # "The new state s' is obtained by composing s and a: s' = s . a".
        # Note: In a real robot step, you'd send R_update as a velocity command or 
        # compute the new target pose for the impedance controller.
        
        # 2. Process Stiffness (Riemannian Approach)
        K_matrix = spd_cholesky_map(stiffness_params)
        
        return {
            "pos_delta": delta_pos,
            "rot_delta": R_update,    # Valid SO(3) matrix
            "stiffness": K_matrix     # Valid SPD matrix
        }