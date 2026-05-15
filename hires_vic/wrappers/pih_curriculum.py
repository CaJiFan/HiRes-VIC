import gymnasium as gym
import torch
import sapien

class InsertionCurriculumWrapper(gym.Wrapper):
    """
    Intercepts env.reset() to execute a scripted batched policy that grasps 
    the peg from the table and aligns it with the hole before handing 
    control to the RL agent.
    """
    def __init__(self, env, setup_steps=90):
        super().__init__(env)
        self.setup_steps = setup_steps

    def reset(self, seed=None, options=None):
        obs, info = self.env.reset(seed=seed, options=options)
        
        device = obs.device if isinstance(obs, torch.Tensor) else "cpu"
        n_envs = getattr(self.env.unwrapped, "num_envs", 1)
        action_dim = self.env.action_space.shape[-1]
        
        for step in range(self.setup_steps):
            # 1. Get current gripper position
            tcp_pos = self.env.unwrapped.agent.tcp.pose.p
            
            # 2. Get target grasp pose from the environment's own reward logic
            # The env explicitly uses an offset of [-0.06, 0, 0] to grab the peg tail
            tgt_gripper_pos = (self.env.unwrapped.peg.pose * sapien.Pose([-0.06, 0, 0])).p
            
            # 3. Get pre-insertion pose
            # The hole points along the local +X axis of box_hole_pose. 
            # We hover 15cm (-0.15) outside the hole.
            pre_insert_pos = (self.env.unwrapped.box_hole_pose * sapien.Pose([-0.15, 0, 0])).p
            
            # 4. Phase-based state machine
            if step < 20:
                # Phase A: Hover 10cm directly above the peg
                target_pos = tgt_gripper_pos + torch.tensor([0, 0, 0.1], device=device)
                gripper_act = 1.0 # Open
            elif step < 40:
                # Phase B: Drop down to the peg tail
                target_pos = tgt_gripper_pos
                gripper_act = 1.0 # Open
            elif step < 60:
                # Phase C: Close the gripper firmly
                target_pos = tgt_gripper_pos
                gripper_act = -1.0 # Close
            else:
                # Phase D: Fly to the hole
                target_pos = pre_insert_pos
                gripper_act = -1.0 # Keep closed
                
            # Proportional (P) controller for smooth movement
            delta_pos = (target_pos - tcp_pos) * 5.0
            delta_pos = torch.clamp(delta_pos, -1.0, 1.0)
            
            # Construct the native action
            scripted_action = torch.zeros((n_envs, action_dim), device=device)
            scripted_action[:, :3] = delta_pos
            scripted_action[:, -1] = gripper_act
            
            # Step the underlying environment silently
            obs, _, _, _, info = self.env.step(scripted_action)

        # Control is handed to the RL agent! The robot is holding the peg right in front of the hole.
        return obs, info