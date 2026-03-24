import torch
import numpy as np
from stable_baselines3 import PPO, SAC, TD3
from sb3_contrib import TQC
import argparse
import os 
import sys
from time import sleep
import imageio

sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))
from hires_vic.envs.gymnasium_wrapper import RobosuiteGymnasiumWrapper, RobosuitePhysicsWrapper
from scripts.plot_ellipsoids import save_ellipsoid_plot

import robosuite as suite
import robosuite.controllers.parts.controller_factory as factory
from robosuite import load_composite_controller_config

from hires_vic.envs.riemannian_controller import RiemannianController

factory.arm_controllers.OperationalSpaceController = RiemannianController


from robosuite.environments.manipulation.wipe import Wipe
from robosuite.environments.base import register_env
from stable_baselines3.common.monitor import Monitor
from stable_baselines3.common.vec_env import DummyVecEnv

class TiltedWipe(Wipe):
    """Custom Wipe environment with a tilted table."""
    def __init__(self, tilt_angle_degrees=15.0, **kwargs):
        self.tilt_angle_rad = np.radians(tilt_angle_degrees)
        super().__init__(**kwargs)

    def _load_model(self):
        super()._load_model()
        table = self.model.mujoco_arena.table_body
        table.set("euler", f"0 {self.tilt_angle_rad:.4f} 0")

register_env(TiltedWipe)

# =================================================================
# 3. ARGUMENT PARSING
# =================================================================
def parse_args():
    parser = argparse.ArgumentParser(description="Evaluate RL on Robosuite")
    parser.add_argument("--env", type=str, default="Door", help="Name of the Robosuite environment")
    parser.add_argument("--run_name", type=str, default="baseline", help="Name for logging/loading models")
    parser.add_argument("--algorithm", type=str, default="SAC", help="RL Algorithm to use")
    parser.add_argument("--episodes", type=int, default=10, help="Number of episodes to evaluate")
    parser.add_argument("--render", action="store_true", help="Turn on the MuJoCo viewer popup")
    parser.add_argument("--seed", type=int, default=1, help="Random seed (e.g., 1, 2, 3)")
    # Ablation Flags
    parser.add_argument("--use_spd", action="store_true", help="Enable Riemannian SPD stiffness mapping")
    parser.add_argument("--use_lie", action="store_true", help="Enable Lie Group orientation prior")
    
    return parser.parse_args()

# =================================================================
# 4. ENVIRONMENT CREATION
# =================================================================
def make_env(args):
    controller_config = None
    # is_vic = "VIC" in args.run_name
    is_vic = True
    kp_limits = [20, 200] # [50, 300]
    env_name = args.env

    if is_vic:
        controller_config = load_composite_controller_config(controller="BASIC", robot="panda")

        phantom_parts = ["left", "torso", "head", "base", "legs"]
        for part in phantom_parts:
            controller_config["body_parts"].pop(part, None)
        
        arm_config = controller_config["body_parts"]["right"]
        arm_config["type"] = "OSC_POSE"

        # "variable_kp": Agent outputs [Pos, Ori, Kp]. Damping (Kd) is auto-calculated.
        # "variable":  Agent outputs [Pos, Ori, Kp, Kd]. Both are learned.
        # "fixed": Agent outputs [Pos, Ori]. Kp is constant.
        arm_config["impedance_mode"] = "riemannian_kp" if args.use_spd else "variable_kp"
        
        # 0 = Completely limp (gravity comp only), 300 = Very stiff
        arm_config["kp_limits"] = kp_limits 
        arm_config["damping_ratio_limits"] = [1.0, 1.0] # Force critical damping
    
        task_kwargs = {
            "has_renderer": False,
            "has_offscreen_renderer": True,
            "use_camera_obs": False,
            "reward_shaping": True,
            "horizon": 300 if env_name == "Door" else 1000, # Shorter episodes for Door
            "control_freq": 20,              #  50ms per step
            "kp_limits": kp_limits if is_vic else None,  # Only pass kp_limits if using VIC
            "task_config": {
                "arm_limit_collision_penalty": -10.0,  # penalty for reaching joint limit or arm collision (except the wiping tool) with the table
                "wipe_contact_reward": 0.01,  # reward for contacting something with the wiping tool
                "unit_wiped_reward": 50.0,  # reward per peg wiped
                "ee_accel_penalty": 0,  # penalty for large end-effector accelerations
                "excess_force_penalty_mul": 0.05,  # penalty for each step that the force is over the safety threshold
                "distance_multiplier": 5.0,  # multiplier for the dense reward inversely proportional to the mean location of the pegs to wipe
                "distance_th_multiplier": 5.0,  # multiplier in the tanh function for the aforementioned reward
                # settings for table top
                "table_full_size": [0.5, 0.8, 0.05],  # Size of tabletop
                "table_offset": [0.15, 0, 0.9],  # Offset of table (z dimension defines max height of table)
                "table_friction": [0.03, 0.005, 0.0001],  # Friction parameters for the table
                "table_friction_std": 0,  # Standard deviation to sample different friction parameters for the table each episode
                "table_height": 0.0,  # Additional height of the table over the default location
                "table_height_std": 0.0,  # Standard deviation to sample different heigths of the table each episode
                "line_width": 0.04,  # Width of the line to wipe (diameter of the pegs)
                "two_clusters": False,  # if the dirt to wipe is one continuous line or two
                "coverage_factor": 0.6,  # how much of the table surface we cover
                "num_markers": 25,  # How many particles of dirt to generate in the environment
                # settings for thresholds
                "contact_threshold": 1.0,  # Minimum eef force to qualify as contact [N]
                "pressure_threshold": 0.5,  # force threshold (N) to overcome to get increased contact wiping reward
                "pressure_threshold_max": 60.0,  # maximum force allowed (N)
                # misc settings
                "print_results": False,  # Whether to print results or not
                "get_info": False,  # Whether to grab info after each env step if not
                "use_robot_obs": True,  # if we use robot observations (proprioception) as input to the policy
                "use_contact_obs": True,  # if we use a binary observation for whether robot is in contact or not
                "early_terminations": True,  # Whether we allow for early terminations or not
                "use_condensed_obj_obs": False,  # Whether to use condensed object observation representation (only applicable if obj obs is active)

            }
        }

    if args.env == "TiltedWipe":
        task_kwargs["tilt_angle_degrees"] = 45.0

    env = RobosuiteGymnasiumWrapper(
        env_name=env_name,
        robots="Panda",
        controller_configs=controller_config,
        task_kwargs=task_kwargs,
        use_spd_manifold=args.use_spd, 
        use_lie_group=args.use_lie
    )
        
    stiff_penalty = 0.01 if is_vic else 0.0
    
    phy_env = RobosuitePhysicsWrapper(
        env, 
        stiffness_penalty=stiff_penalty, 
        force_penalty=0.02, 
        max_force_threshold=35.0
    )

    # env = Monitor(env)
    return phy_env, env

# =================================================================
# 5. MAIN EVALUATION LOOP
# =================================================================
def main():
    args = parse_args()
    device = "cuda" if torch.cuda.is_available() else "cpu"
    
    env_name = args.env
    run_name = f'{args.algorithm}_{env_name.upper()}_{args.run_name}_SEED_{args.seed}'

    model_path = f"./outputs/models/{run_name}_final"
    video_dir = f"./outputs/videos/{run_name}"
    os.makedirs(video_dir, exist_ok=True)
    
    env, wrapper = make_env(args)

    print(f"Loading {args.algorithm} model from {model_path}...")
    if args.algorithm == "PPO":
        model = PPO.load(model_path, env=env, device=device)
    elif args.algorithm == "SAC":
        model = SAC.load(model_path, env=env, device=device)
    elif args.algorithm == "TD3":
        model = TD3.load(model_path, env=env, device=device)
    elif args.algorithm == "TQC":
        model = TQC.load(model_path, env=env, device=device)
    
    print(f"\nStarting Formal Evaluation over {args.episodes} episodes...")
    successes = 0
    
    for ep in range(args.episodes):
        obs, info = env.reset()
        done = False
        truncated = False
        ep_reward = 0.0
        step_counter = 0

        video_path = os.path.join(video_dir, f"eval_ep_{ep+1:02d}.mp4")
        video_writer = imageio.get_writer(video_path, fps=20)
        
        while not (done or truncated):
            action, _states = model.predict(obs, deterministic=True)
            obs, reward, done, truncated, info = env.step(action)
            ep_reward += reward
            step_counter += 1

            frame = wrapper.env.sim.render(camera_name="frontview", width=1024, height=1024)[::-1]
            
            # 3. Append the frame to the video
            video_writer.append_data(frame)
            
            if args.render:
                wrapper.render()

            # if args.use_spd and step_counter % 50 == 0:
            #     # Navigate through the Robosuite hierarchy to your RiemannianController
            #     custom_controller = wrapper.env.robots[0].composite_controller.part_controllers['right']
                
            #     # Extract the matrix you defined in set_goal()!
            #     Kp_pos_3x3 = custom_controller.kp_pos_matrix
                
            #     # Save the plot
            #     save_ellipsoid_plot(Kp_pos_3x3, episode=ep, step=step_counter)

            print(reward)
            sleep(0.01)  # Slow down for better visualization

        video_writer.close()
        print(f"🎥 Saved video for Episode {ep+1} to {video_path}")

        is_success = wrapper.env._check_success()
        if is_success:
            successes += 1
            
        print(f"Episode {ep + 1}/{args.episodes} | Reward: {ep_reward:.2f} | Success: {is_success}")

    print("=====================================================")
    print(f"Evaluation Complete for {args.algorithm} on {args.env}")
    print(f"Ablations -> SPD: {args.use_spd} | Lie Group: {args.use_lie}")
    print(f"Overall Success Rate: {(successes / args.episodes) * 100:.1f}%")
    print("=====================================================")

    env.close()

if __name__ == "__main__":
    main()