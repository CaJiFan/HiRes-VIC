import torch
import math
import gymnasium as gym
import numpy as np
import robosuite as suite
from gymnasium import spaces
from hires_vic.geometry.riemannian import spd_grl_map
from hires_vic.geometry.lie_groups import so3_log_map
from robosuite.utils import transform_utils as T
import matplotlib.pyplot as plt
import wandb


class RobosuiteGymnasiumWrapper(gym.Env):
    def __init__(self, env_name, robots, use_spd_manifold=False, use_lie_group=False, controller_configs=None, task_kwargs=None):
        """
        Wraps a Robosuite environment to be compatible with Gymnasium.
        """ 

        self.use_spd_manifold = use_spd_manifold
        self.use_lie_group = use_lie_group
        print(f"🔧 Robosuite Wrapper Initialized | SPD: {self.use_spd_manifold} | Lie Group: {self.use_lie_group}")

        if task_kwargs is None:
            task_kwargs = {}

        # Default settings (can be overridden by task_kwargs)
        # We use .pop() so we don't pass them twice to suite.make()
        has_renderer = task_kwargs.pop("has_renderer", False)
        has_offscreen_renderer = task_kwargs.pop("has_offscreen_renderer", False)
        use_camera_obs = task_kwargs.pop("use_camera_obs", False)
        use_object_obs = task_kwargs.pop("use_object_obs", True)
        reward_shaping = task_kwargs.pop("reward_shaping", True)

        self.min_kp, self.max_kp  = np.array(task_kwargs.pop("kp_limits", [50.0, 300.0]))
        
        self.env = suite.make(
            env_name,
            robots=robots,
            controller_configs=controller_configs,
            has_renderer=has_renderer,
            has_offscreen_renderer=has_offscreen_renderer,
            use_camera_obs=use_camera_obs,
            use_object_obs=use_object_obs,
            reward_shaping=reward_shaping,
            **task_kwargs
        )

        # Action space
        action_dim = 9 # rot kp + pos + ori

        action_dim += 6 if self.use_spd_manifold else 3
        
        gripper_dim = self.env.action_dim - (18 if self.use_spd_manifold else 12)
        if gripper_dim > 0:
            action_dim += gripper_dim

        self.action_space = spaces.Box(low=-1.0, high=1.0, shape=(action_dim,), dtype=np.float32)

        self.real_low, self.real_high = self.env.action_spec
        
        if self.use_spd_manifold:
            # GRL Arm always needs exactly 15 dimensions (6 Mandel + 3 RotKp + 3 Pos + 3 Ori)
            print(f"🚀 GRL CONTROLLER DETECTED WITH ACTION DIM {action_dim} | Gripper Dim: {gripper_dim}")
        else:
            print(f"🚀 STANDARD OSC CONTROLLER DETECTED WITH ACTION DIM {action_dim} | Gripper Dim: {gripper_dim}")

        print(f'Max and min Kp values for GRL mapping: {self.min_kp}, {self.max_kp}')


        # Observation space 
        obs = self.env.reset()
        flat_obs = self._flatten_obs(obs)

        print('▶️ Observation space shape after flattening: ', flat_obs.shape)

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
            
            # --- LIE GROUP OBSERVATION MAPPING ---
            if key == 'robot0_eef_quat' and self.use_lie_group:
                quat = obs_dict[key]
                rot_mat = T.quat2mat(quat)
                rot_tensor = torch.tensor(rot_mat, dtype=torch.float32).unsqueeze(0)
                omega = so3_log_map(rot_tensor).squeeze(0).detach().numpy() # log map: SO(3) Manifold -> so(3) TxM Lie Algebra
                
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

    def step(self, action):
        idx = 0

        if self.use_spd_manifold:
            mandel_params = action[idx:idx+6].copy()
            idx+=6

            log_min = math.log(self.min_kp)
            log_max = math.log(self.max_kp)
            
            mandel_params[0:3] = log_min + 0.5 * (mandel_params[0:3] + 1.0) * (log_max - log_min)
            # off_diag_scale = (log_max - log_min) / 2.0
            off_diag_scale = 0.2
            mandel_params[3:6] = mandel_params[3:6] * off_diag_scale
            
            mandel_tensor = torch.tensor(mandel_params, dtype=torch.float32).unsqueeze(0)
            Kp_matrix = spd_grl_map(mandel_tensor).squeeze(0).detach().numpy()
            Kp_flat = Kp_matrix.flatten() 
            
            # Rotational stiffness
            kp_rot_raw = action[6:9]
            kp_rot_scaled = self.min_kp + 0.5 * (kp_rot_raw + 1.0) * (self.max_kp - self.min_kp)
            
            # action[15:] will safely grab the 1 gripper command for NutAssembly, 
            # or it will return an empty array [] for Wipe!
            robosuite_action = np.concatenate([
                Kp_flat,             # 9 elements
                kp_rot_scaled,       # 3 elements
                action[9:12],        # 3 elements: pos
                action[12:15],       # 3 elements: ori
                action[15:]          # Remaining elements (Gripper, if any)
            ])
            
        else:
            # Standard baseline execution
            robosuite_action = self.real_low + (0.5 * (action + 1.0) * (self.real_high - self.real_low))

        # print(f"Scaled Robosuite Action ({len(robosuite_action)}): {robosuite_action}")
        obs_dict, reward, done, info = self.env.step(robosuite_action)

        # raw_success = self.env._check_success()
        # info["is_success"] = bool(raw_success)

        total_markers = self.env.num_markers
        wiped_markers = len(self.env.wiped_markers)
        percent_wiped = wiped_markers / total_markers
        
        info["is_success"] = percent_wiped
        
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
    def __init__(self, env, is_eval=False, stiffness_penalty=0.0, force_penalty=0.0, max_force_threshold=30.0, terminate_on_unsafe=False):
        super().__init__(env)
        self.stiffness_penalty = stiffness_penalty
        self.force_penalty = force_penalty
        self.max_force_threshold = max_force_threshold
        self.terminate_on_unsafe = terminate_on_unsafe
        self.is_eval = is_eval

        if self.is_eval:
            self.kp_history = []
        
        # Internal counters for logging
        self.episode_stiffness_sum = 0.0
        self.episode_force_sum = 0.0
        self.episode_steps = 0
        self.violation_count = 0

        self.ep_kp_trans_x = 0.0
        self.ep_kp_trans_y = 0.0
        self.ep_kp_trans_z = 0.0
        self.ep_kp_rot_x = 0.0
        self.ep_kp_rot_y = 0.0
        self.ep_kp_rot_z = 0.0

    def reset(self, **kwargs):
        self.episode_stiffness_sum = 0.0
        self.episode_force_sum = 0.0
        self.episode_steps = 0
        self.violation_count = 0

        self.ep_kp_trans_x = 0.0
        self.ep_kp_trans_y = 0.0
        self.ep_kp_trans_z = 0.0
        self.ep_kp_rot_x = 0.0
        self.ep_kp_rot_y = 0.0
        self.ep_kp_rot_z = 0.0

        if self.is_eval:
            self.kp_history.clear()
        return self.env.reset(**kwargs)

    def step(self, action):
        # print(f"Raw Action - PhysicsWrapper ({len(action)}): {action}")
        obs, reward, terminated, truncated, info = self.env.step(action)
        self.episode_steps += 1

        gymwrapper_env = self.env
        robosuite_env = gymwrapper_env.env
        robot = robosuite_env.robots[0]

        min_kp, max_kp = gymwrapper_env.min_kp, gymwrapper_env.max_kp
        # print('KP LIMITS!!')
        # print(min_kp, max_kp)
    
        # Get Contact Forces
        try:
            ee_force = max([
                np.linalg.norm(np.array(robot.recent_ee_forcetorques[arm].current[:3]))
                for arm in robot.arms
            ])
        except Exception as e:
            ee_force = 0.0

        
        try:
            if gymwrapper_env.use_spd_manifold: 
                # Layout: Kp_trans_mandel(6), Kp_rot(3), pos(3), ori(3), gripper(1)
                
                # We extract the 3 diagonals from the Mandel parameters (indices 0, 1, 2)
                kp_trans_diags = action[0:3] 
                # We extract the 3 rotational stiffness diagonals (indices 6, 7, 8)
                kp_rot = action[6:9] 
                
                kp_vals = np.concatenate([kp_trans_diags, kp_rot])
            else: 
                # Layout: Kp(6), pos(3), ori(3), gripper(1)
                kp_vals = action[0:6]

            kp_vals_percentage = (kp_vals + 1.0) / 2.0
            physical_kp_vals = min_kp + (kp_vals_percentage * (max_kp - min_kp))            
                
            stiffness_percentage = np.mean((kp_vals + 1.0) / 2.0)
            physical_stiffness = min_kp + (stiffness_percentage * (max_kp - min_kp))

        except Exception as e:
            print(f"Error extracting stiffness from action: {e}")
            stiffness_percentage = 0.0 
            physical_stiffness = 0.0

        if self.is_eval:
            self.kp_history.append(physical_kp_vals.copy())

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
        
        # Force Penalty (Soft Constraint)
        # force_penalty_val = 0.0
        # if self.force_penalty > 0 and ee_force > self.max_force_threshold:
        #     excess_force = ee_force - self.max_force_threshold
        #     force_penalty_val = self.force_penalty * excess_force
        #     reward -= force_penalty_val 

        # Stiffness Penalty (Energy Efficiency)
        stiffness_penalty_val = 0.0
        if self.stiffness_penalty > 0:
            stiffness_penalty_val = self.stiffness_penalty * (stiffness_percentage**2)
            reward -= stiffness_penalty_val

        # LOGGING 
        self.episode_stiffness_sum += physical_stiffness
        self.episode_force_sum += ee_force
        self.ep_kp_trans_x += physical_kp_vals[0]
        self.ep_kp_trans_y += physical_kp_vals[1]
        self.ep_kp_trans_z += physical_kp_vals[2]
        self.ep_kp_rot_x += physical_kp_vals[3]
        self.ep_kp_rot_y += physical_kp_vals[4]
        self.ep_kp_rot_z += physical_kp_vals[5]

        # # Log instantaneous metrics (for debugging spikes)
        # info["physics/stiffness_step"] = physical_stiffness
        # info["physics/force_step"] = ee_force
        # # info["reward/force_penalty"] = force_penalty_val
        # info["reward/stiffness_penalty"] = stiffness_penalty_val
        # info["safety/joint_violation"] = is_unsafe

        # info["physics/kp_trans_x"] = self.ep_kp_trans_x
        # info["physics/kp_trans_y"] = self.ep_kp_trans_y
        # info["physics/kp_trans_z"] = self.ep_kp_trans_z
        # info["physics/kp_rot_x"] = self.ep_kp_rot_x
        # info["physics/kp_rot_y"] = self.ep_kp_rot_y
        # info["physics/kp_rot_z"] = self.ep_kp_rot_z


        # Log Episode Averages (Only when episode ends)
        if terminated or truncated:
            # avg_stiffness = self.episode_stiffness_sum / max(1, self.episode_steps)
            # info["physics/avg_stiffness"] = avg_stiffness

            total_markers = gymwrapper_env.num_markers
            wiped_markers = len(gymwrapper_env.wiped_markers)
            percent_wiped = wiped_markers / total_markers

            print('total markers', total_markers, 'wiped markers', wiped_markers, '%', percent_wiped )

            info["physics/raw_wipe_percentage"] = percent_wiped

            if self.is_eval and len(self.kp_history) > 0:
                history_array = np.array(self.kp_history) # Shape: (timesteps, 6)

                eval_kp_avgs = np.mean(history_array, axis=0)

                # 2. Send the plots AND the exact numerical averages to WandB!
                wandb.log({
                    "eval/kp_trans_x_avg": eval_kp_avgs[0],
                    "eval/kp_trans_y_avg": eval_kp_avgs[1],
                    "eval/kp_trans_z_avg": eval_kp_avgs[2],
                    "eval/kp_rot_x_avg": eval_kp_avgs[3],
                    "eval/kp_rot_y_avg": eval_kp_avgs[4],
                    "eval/kp_rot_z_avg": eval_kp_avgs[5]
                })
                
            #     # --- Figure 1: Translational Stiffness ---
            #     fig_trans, ax_trans = plt.subplots(figsize=(10, 6), dpi=150)
            #     ax_trans.plot(history_array[:, 0], label="Kp_trans_X")
            #     ax_trans.plot(history_array[:, 1], label="Kp_trans_Y")
            #     ax_trans.plot(history_array[:, 2], label="Kp_trans_Z", linewidth=2, linestyle='--')
            #     ax_trans.set_title("Translational Impedance Profile")
            #     ax_trans.set_xlabel("Timesteps")
            #     ax_trans.set_ylabel("Stiffness (N/m)")
                
            #     # Dynamic Y-Limits based on Robosuite's physical limits!
            #     # We add a 5% margin so the lines don't touch the literal top/bottom of the chart
            #     trans_margin = (max_kp - min_kp) * 0.05
            #     ax_trans.set_ylim(min_kp - trans_margin, max_kp + trans_margin) 
                
            #     ax_trans.legend(loc="upper right")
            #     ax_trans.grid(True)
            #     fig_trans.tight_layout()

            #     # --- Figure 2: Rotational Stiffness ---
            #     fig_rot, ax_rot = plt.subplots(figsize=(10, 6), dpi=150)
            #     ax_rot.plot(history_array[:, 3], label="Kp_rot_X")
            #     ax_rot.plot(history_array[:, 4], label="Kp_rot_Y")
            #     ax_rot.plot(history_array[:, 5], label="Kp_rot_Z", linewidth=2, linestyle='--')
            #     ax_rot.set_title("Rotational Impedance Profile")
            #     ax_rot.set_xlabel("Timesteps")
            #     ax_rot.set_ylabel("Stiffness (Nm/rad)")
                
            #     # Dynamic Y-Limits for rotation
            #     rot_margin = (max_kp - min_kp) * 0.05
            #     ax_rot.set_ylim(min_kp - rot_margin, max_kp + rot_margin) 
                
            #     ax_rot.legend(loc="upper right")
            #     ax_rot.grid(True)
            #     fig_rot.tight_layout()

            #     # Send both plots directly to WandB as separate panels
            #     wandb.log({
            #         "eval/kp_trans_profile": wandb.Image(fig_trans),
            #         "eval/kp_rot_profile": wandb.Image(fig_rot)
            #     })
                
            #     # Close both figures to prevent memory leaks!
            #     plt.close(fig_trans)
            #     plt.close(fig_rot)


            avg_force = self.episode_force_sum / max(1, self.episode_steps)
            
            info["physics/max_force_violation_count"] = self.violation_count
            info["physics/kp_trans_x_avg"] = self.ep_kp_trans_x / max(1, self.episode_steps)
            info["physics/kp_trans_y_avg"] = self.ep_kp_trans_y / max(1, self.episode_steps)
            info["physics/kp_trans_z_avg"] = self.ep_kp_trans_z / max(1, self.episode_steps)
            info["physics/kp_rot_x_avg"] = self.ep_kp_rot_x / max(1, self.episode_steps)
            info["physics/kp_rot_y_avg"] = self.ep_kp_rot_y / max(1, self.episode_steps)
            info["physics/kp_rot_z_avg"] = self.ep_kp_rot_z / max(1, self.episode_steps)
            info["physics/avg_force"] = avg_force

        return obs, reward, terminated, truncated, info