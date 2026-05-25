import gymnasium as gym
import torch
import sapien
from scipy.spatial.transform import Rotation as R
from mani_skill.utils.structs.pose import Pose
import numpy as np


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


class ManiskillTeleportWrapper(gym.Wrapper):
    """
    Bypasses the P-controller entirely! Instantly teleports the peg into the robot's 
    hand, and teleports the box directly in front of the robot. 
    """
    def __init__(self, env, setup_steps=15):
        super().__init__(env)
        self.setup_steps = setup_steps

    def reset(self, seed=None, options=None):
        # 1. Reset the environment normally (randomizes shapes and sizes)
        obs, info = self.env.reset(seed=seed, options=options)
        
        env_unwrapped = self.env.unwrapped
        device = obs.device if isinstance(obs, torch.Tensor) else "cpu"
        n_envs = getattr(env_unwrapped, "num_envs", 1)
        action_dim = self.env.action_space.shape[-1]
        
        # 2. Teleport the Peg into the gripper
        tcp_pose = env_unwrapped.agent.tcp.pose
        
        # The env's reward logic assumes a grasp offset of [-0.06, 0, 0] relative to the peg.
        # By inverting this, we put the peg perfectly inside the TCP.
        offset_p = torch.zeros((n_envs, 3), device=device)
        offset_p[:, 0] = 0.06
        peg_offset = Pose.create_from_pq(p=offset_p)
        
        new_peg_pose = tcp_pose * peg_offset
        env_unwrapped.peg.set_pose(new_peg_pose)
        
        # 3. Teleport the Box directly in front of the gripper
        peg_lengths = env_unwrapped.peg_half_sizes[:, 0]
        
        # Place the hole exactly 2cm (0.02) in front of the peg tip
        hole_offset_p = torch.zeros((n_envs, 3), device=device)
        hole_offset_p[:, 0] = 0.06 + peg_lengths + 0.08
        
        hole_target_pose = tcp_pose * Pose.create_from_pq(p=hole_offset_p)
        
        # Apply the inverse hole offset to perfectly position the outer box
        new_box_pose = hole_target_pose * env_unwrapped.box_hole_offsets.inv()
        env_unwrapped.box.set_pose(new_box_pose)
        
        # 4. Settle the physics (Close the gripper tightly)
        scripted_action = torch.zeros((n_envs, action_dim), device=device)
        scripted_action[:, -1] = -1.0 # Force gripper closed
        
        for _ in range(self.setup_steps):
            obs, _, _, _, info = self.env.step(scripted_action)

        # Hand control to the RL Agent!
        return obs, info


class RobosuiteScriptedPrimitiveWrapper(gym.Wrapper):
    """
    Scripted motion primitive for Robosuite-based PiH/NutAssembly envs.
    Performs: hover -> descend -> close gripper -> lift
    Uses the outer `GeometricWrapper`'s RL action-space (normalized [-1,1]).
    """
    def __init__(self, env, setup_steps=90, hover_height=0.05, approach_gain=5.0, is_eval=False, eval_approach_gain=None, eval_setup_steps=None):
        super().__init__(env)
        self.is_eval = bool(is_eval)
        # allow evaluation mode to use a different (gentler) setup
        if self.is_eval and eval_setup_steps is not None:
            self.setup_steps = int(eval_setup_steps)
        else:
            self.setup_steps = int(setup_steps)
        self.hover_height = float(hover_height)
        # Choose gentler gains during evaluation to avoid aggressive/twitchy motion
        if self.is_eval:
            if eval_approach_gain is not None:
                self.approach_gain = float(eval_approach_gain)
            else:
                self.approach_gain = float(approach_gain) * 0.3
        else:
            self.approach_gain = float(approach_gain)

    def _determine_pos_index(self):
        # Heuristic matching GeometricWrapper's action layout
        inner = self.env
        if hasattr(inner, 'use_spd_manifold') and inner.use_spd_manifold:
            return 9
        # diag or baseline maps pos at index 6
        return 6
    
    def _find_candidate_object_pos(self, raw_obs, eef_pos=None):
        if not isinstance(raw_obs, dict) or eef_pos is None:
            return None

        # 1. Prioritize Relative Nut Coordinates (The Math Fix!)
        for k, v in raw_obs.items():
            if 'nut' in k.lower() and 'to_robot0_eef_pos' in k.lower():
                rel_pos = np.asarray(v).flatten()[:3]
                # Absolute target = Current EEF position + Vector to the Nut
                return eef_pos + rel_pos

        # 2. Check for absolute Nut coordinates
        for k, v in raw_obs.items():
            if 'nut' in k.lower() and 'to_' not in k.lower():
                arr = np.asarray(v)
                if arr.size >= 3:
                    return arr.flatten()[:3]
                    
        # 3. Fallback: Relative Peg/Bolt coordinates
        for k, v in raw_obs.items():
            if ('peg' in k.lower() or 'bolt' in k.lower()) and 'to_robot0_eef_pos' in k.lower():
                rel_pos = np.asarray(v).flatten()[:3]
                return eef_pos + rel_pos

        # 4. Fallback: Absolute Peg/Bolt coordinates
        for k, v in raw_obs.items():
            lk = k.lower()
            if 'peg' in lk or 'bolt' in lk or 'screw' in lk:
                arr = np.asarray(v)
                if arr.size >= 3:
                    return arr.flatten()[:3]

        # 5. Last Resort: Scan dict for any 3-length arrays and pick nearest to EEF
        candidates = []
        for k, v in raw_obs.items():
            try:
                arr = np.asarray(v).flatten()
            except Exception:
                continue
            if arr.size == 3 or arr.size == 7:
                pos = arr[:3]
                candidates.append(pos)

        if candidates:
            dists = [np.linalg.norm(c - eef_pos) for c in candidates]
            idx = int(np.argmin(dists))
            return candidates[idx]

        return None
    
    def reset(self, seed=None, options=None):
        # Reset the underlying env normally first
        obs, info = self.env.reset(seed=seed, options=options)

        # Try to read raw observations to locate peg/hole
        raw_obs = None
        try:
            raw_obs = self.env.unwrapped._get_observations()
        except Exception:
            try:
                raw_obs = self.env.unwrapped.env._get_observations()
            except Exception:
                raw_obs = None

        eef_pos = None
        if isinstance(raw_obs, dict) and 'robot0_eef_pos' in raw_obs:
            eef_pos = np.asarray(raw_obs['robot0_eef_pos']).flatten()

        peg_pos = self._find_candidate_object_pos(raw_obs, eef_pos)
        if peg_pos is None:
            return obs, info

        action_dim = int(self.env.action_space.shape[-1])
        pos_idx = self._determine_pos_index()
        gripper_idx = max(0, action_dim - 1)

        # Temporarily ask inner env to respect our gripper commands
        try:
            setattr(self.env, 'suppress_forced_gripper', True)
        except Exception:
            pass

        for step in range(self.setup_steps):
            try:
                current_raw = self.env.unwrapped._get_observations()
            except Exception:
                current_raw = raw_obs
            tcp_pos = np.asarray(current_raw.get('robot0_eef_pos', eef_pos)) if current_raw is not None else eef_pos

            # Phases: approach hover -> descend -> close -> lift
            phase = step / max(1, self.setup_steps)
            
            # 1. Hover above the nut (Fingers OPEN: -1.0)
            if phase < 0.33:
                target = peg_pos + np.array([0.0, 0.0, self.hover_height])
                gripper_act = -1.0 
                
            # 2. Descend to the nut (Fingers OPEN: -1.0)
            elif phase < 0.66:
                target = peg_pos
                gripper_act = -1.0 
                
            # 3. Grasp the nut (Fingers CLOSE: 1.0) 
            # Note: Increased phase end from 0.72 to 0.85 to give MuJoCo ~17 steps to securely clamp!
            elif phase < 0.85:
                target = peg_pos
                gripper_act = 1.0 
                
            # 4. Lift the nut (Fingers CLOSE: 1.0)
            else:
                target = peg_pos + np.array([0.0, 0.0, self.hover_height])
                gripper_act = 1.0 

            delta_pos = (target - tcp_pos) * self.approach_gain
            delta_pos = np.clip(delta_pos, -1.0, 1.0)

            scripted_action = np.zeros((action_dim,), dtype=np.float32)
            scripted_action[pos_idx:pos_idx + 3] = delta_pos
            scripted_action[gripper_idx] = float(gripper_act)

            # Step the environment with the RL-level scripted action
            obs, _, _, _, info = self.env.step(scripted_action)

        # Clear the temporary flag so subsequent RL control can re-assert gripper state
        try:
            setattr(self.env, 'suppress_forced_gripper', False)
        except Exception:
            pass

        # Done — hand control back to RL agent
        return obs, info


class RobosuiteTeleportWrapper(gym.Wrapper):
    """
    Bypasses the pick phase! Teleports the nut directly into the robot's 
    hand and settles the physics to start the episode ready for insertion.
    """
    def __init__(self, env, setup_steps=15, is_eval=False):
        super().__init__(env)
        self.setup_steps = setup_steps
        self.is_eval = is_eval
        self.frames = [] # For debugging: capture frames during the setup phase
    
    def _capture_frame(self):
        """Capture a single RGB frame from the video env (render or raw obs fallback)."""
        frame = None
        try:
            frame = self.env.render()
        except Exception:
            frame = None

        if frame is None:
            raw_obs = None
            try:
                raw_obs = self.env.env.unwrapped._get_observations()
            except Exception:
                try:
                    raw_obs = self.env.unwrapped._get_observations()
                except Exception:
                    raw_obs = None

            if isinstance(raw_obs, dict):
                img_key = None
                for k in ("frontview_image", "agentview_image"):
                    if k in raw_obs:
                        img_key = k
                        break
                if img_key is None:
                    for k in raw_obs.keys():
                        if any(x in k.lower() for x in ("image", "camera", "rgb")):
                            img_key = k
                            break
                if img_key is not None:
                    try:
                        frame = raw_obs[img_key]
                    except Exception:
                        frame = None

        if frame is not None:
            if isinstance(frame, torch.Tensor):
                frame = frame.cpu().numpy()
            frame = np.asarray(frame, dtype=np.uint8)
            if frame.ndim == 4:
                frame = frame[0]
            try:
                frame = np.flipud(frame)
            except Exception:
                pass
        return frame

    def reset(self, **kwargs):
        print('Resetting...')
        obs = self.env.reset(**kwargs)

        sim = self.env.unwrapped.sim
        # 2. Get the TCP (Gripper) Pose
        # Robosuite caches this in the unwrapped environment
        raw_obs = self.env.unwrapped._get_observations()
        eef_pos = raw_obs['robot0_eef_pos']
        eef_quat = raw_obs['robot0_eef_quat'] # [x, y, z, w]
        # 3. Calculate where you want the Nut to be
        # (e.g., perfectly centered between the fingers)
        nut_target_pos = eef_pos + np.array([0.048, 0.0, 0.015]) # Small Z offset so it sits in the pads

        r_eef = R.from_quat(eef_quat)
        r_flip = R.from_euler('z', np.pi)
        r_nut = r_eef * r_flip
        flipped_quat = r_nut.as_quat() # Returns standard [x, y, z, w]

        # mujoco expects [w, x, y, z] 
        nut_target_quat_mujoco = np.array([flipped_quat[3], flipped_quat[0], flipped_quat[1], flipped_quat[2]])
        
        # 4. Generate Randomized Peg Hover Target
        try:
            peg_key = 'peg1' if 'square' in type(self.env.unwrapped).__name__.lower() else 'peg2'
            peg_id = sim.model.body_name2id(peg_key)
            peg_base_pos = np.array(sim.data.body_xpos[peg_id])
            
            # Curriculum Noise: +/- 2.0cm offset so the RL agent is forced to search!
            noise_x = np.random.uniform(-0.020, 0.020)
            noise_y = np.random.uniform(-0.020, 0.020)
            
            # Hover ~12cm above the base of the peg
            hover_target = peg_base_pos + np.array([noise_x-0.048, noise_y, 0.12])
        except Exception as e:
            print("Peg hover target generation failed! Check the peg body name.", e)
            hover_target = eef_pos 

        action_dim = self.env.action_space.shape[0] # Expects raw 7D array
        
        for dummy_t in range(self.setup_steps):
            scripted_action = np.zeros(action_dim, dtype=np.float32)
            current_eef = self.env.unwrapped._get_observations()['robot0_eef_pos']
            gain = 10.0
            mid_target = hover_target + np.array([0.0, 0.0, 0.04]) 

            if dummy_t < self.setup_steps // 2.75:
                pos_error = (mid_target - current_eef)
            else:
                pos_error = (hover_target - current_eef)
            
            pos_error *= gain * 0.05 # Scale down for stability
            
            if action_dim < 16:
                pos_idx = 6
                scripted_action[0:pos_idx] = np.array([-0.89, -0.89, -0.52,  -0.8, -0.5, -0.5]) 
                # 
            else:
                pos_idx = 9
                scripted_action[0:pos_idx] = np.array([0.0, -0.0, 0.5, 0, 0, 0, -0.8, -0.5, -0.5]) 
            scripted_action[pos_idx:pos_idx+3] = np.clip(pos_error, -1.0, 1.0)
            scripted_action[-1] = 1.0 

            obs, _, _, _, info = self.env.step(scripted_action)

            if dummy_t == 2:
                try:
                    joint_name = "SquareNut_joint0" 
                    
                    # Find exactly where in the giant array the nut lives
                    qpos_addr = sim.model.get_joint_qpos_addr(joint_name)
                    qvel_addr = sim.model.get_joint_qvel_addr(joint_name)
                    
                    # Teleport: Overwrite the 7 positional values (X, Y, Z, Qw, Qx, Qy, Qz)
                    sim.data.qpos[qpos_addr[0] : qpos_addr[1]] = np.concatenate([nut_target_pos, nut_target_quat_mujoco])
                    
                    # Kill momentum: Overwrite the 6 velocity values (Linear X/Y/Z, Angular X/Y/Z) to zero
                    sim.data.qvel[qvel_addr[0] : qvel_addr[1]] = np.zeros(6)
                    
                    # Tell MuJoCo to apply these hardcoded changes immediately!
                    sim.forward()
                    
                except Exception as e:
                    print("Teleport failed! Check the joint name of the object.", e)
            
            self.frames.append(self._capture_frame())

        try:
            setattr(self.env, 'suppress_forced_gripper', False)
        except Exception:
            pass

        return obs, info