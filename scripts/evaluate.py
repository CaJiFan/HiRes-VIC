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
from scripts.plot_ellipsoids import save_ellipsoid_plot

import robosuite as suite
import robosuite.controllers.parts.controller_factory as factory
from robosuite import load_composite_controller_config
from robosuite.wrappers import GymWrapper

from hires_vic import envs
from hires_vic.envs.riemannian_controller import RiemannianController
from hires_vic.wrappers import GeometricWrapper
from stable_baselines3.common.vec_env import SubprocVecEnv, VecMonitor, DummyVecEnv

# Inject custom Riemannian controller
factory.arm_controllers.OperationalSpaceController = RiemannianController


WIPE_TASK_CONFIG = {
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
    "use_condensed_obj_obs": True,  # Whether to use condensed object observation representation (only applicable if obj obs is active)
}


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
    
    # Ablation Flags (Must exactly match the training run!)
    parser.add_argument("--use_spd", action="store_true", help="Enable Riemannian SPD stiffness mapping")
    parser.add_argument("--use_lie", action="store_true", help="Enable Lie Group orientation prior")
    parser.add_argument("--use_diag", action="store_true", help="Enable Diagonal stiffness matrix")
    parser.add_argument("--use_fixed", action="store_true", help="Use fixed stiffness (Baseline)")
    
    return parser.parse_args()

# =================================================================
# 4. ENVIRONMENT CREATION
# =================================================================
def make_env(args):
    controller_config = load_composite_controller_config(controller="BASIC", robot="panda")

    phantom_parts = ["left", "torso", "head", "base", "legs"]
    for part in phantom_parts:
        controller_config["body_parts"].pop(part, None)
    
    arm_config = controller_config["body_parts"]["right"]
    arm_config["type"] = "OSC_POSE"

    # ✅ Determine exact Impedance Mode based on architecture flags
    if args.use_fixed:
        arm_config["impedance_mode"] = "fixed"
        arm_config["kp"] = 500
    elif args.use_spd:
        arm_config["impedance_mode"] = "riemannian_kp"
    else:
        arm_config["impedance_mode"] = "variable_kp"
    
    kp_limits = [1, 300]
    arm_config["kp_limits"] = kp_limits 
    arm_config["damping_ratio_limits"] = [1.0, 1.0] # Force critical damping

    # 1. Base Robosuite Environment
    env = suite.make(
        env_name=args.env,
        robots="Panda",
        controller_configs=controller_config,
        has_renderer=False,
        use_object_obs=True,
        has_offscreen_renderer=True,
        use_camera_obs=False,
        reward_shaping=True,
        horizon=300,
        # Task specific kwargs
        task_config=WIPE_TASK_CONFIG
    )

    # 2. Wrap for standard Gym formatting
    env = GymWrapper(env)

    # 3. Apply the custom Geometric action/observation math
    env = GeometricWrapper(
        env=env,
        use_spd_manifold=args.use_spd,
        use_lie_group=args.use_lie,
        use_diag_manifold=args.use_diag,
        use_fixed=args.use_fixed,
        is_eval=False
    )

    # Returning env twice so the main loop unpacks safely (phy_env, env)
    return env, env

# =================================================================
# 5. MAIN EVALUATION LOOP
# =================================================================
def main():
    args = parse_args()
    device = "cuda" if torch.cuda.is_available() else "cpu"
    
    env_name = args.env
    run_name = f'{args.algorithm}_{env_name.upper()}_{args.run_name}_SEED_{args.seed}'

    model_path = f"./outputs/models/{run_name}_final"
    video_dir = f"./outputs/videos/NM25/{run_name}"
    os.makedirs(video_dir, exist_ok=True)
    
    env, wrapper = make_env(args)
    # eval_env = DummyVecEnv([eval_env_fn])
    # eval_env = VecMonitor(eval_env)

    print(f"Loading {args.algorithm} model from {model_path}...")
    if args.algorithm == "PPO":
        model = PPO.load(model_path, env=env, device=device)
    elif args.algorithm == "SAC":
        model = SAC.load(model_path, env=env, device=device)
    elif args.algorithm == "TD3":
        model = TD3.load(model_path, env=env, device=device)
    elif args.algorithm == "TQC":
        model = TQC.load(model_path, env=env, device=device)
    else:
        raise ValueError(f"Unknown algorithm: {args.algorithm}")
    
    print(f"\nStarting Formal Evaluation over {args.episodes} episodes...")
    
    # Tracking metrics for the ICRA paper
    successes = 0
    total_wipe_percentages = []
    
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

            # Safely extract the frame
            frame = env.unwrapped.sim.render(camera_name="frontview", width=1024, height=1024)[::-1]
            video_writer.append_data(frame)
            
            if args.render:
                env.render()

            print(f"\rStep: {step_counter} | Reward: {reward:.4f}", end="")
            sleep(0.05)

        video_writer.close()
        
        # ✅ Calculate the true Wipe Percentage directly from the Robosuite simulation state
        unwrapped_env = env.unwrapped
        num_wiped = len(unwrapped_env.wiped_markers)
        total_markers = unwrapped_env.num_markers
        wipe_percentage = num_wiped / total_markers
        total_wipe_percentages.append(wipe_percentage)

        is_success = unwrapped_env._check_success()
        if is_success:
            successes += 1
            
        print(f"\n🎥 Saved video to {video_path}")
        print(f"Episode {ep + 1}/{args.episodes} | Return: {ep_reward:.2f} | Wiped: {wipe_percentage*100:.1f}% | Success: {is_success}\n")

    # Aggregate final metrics
    mean_wipe_percentage = np.mean(total_wipe_percentages) * 100
    success_rate = (successes / args.episodes) * 100

    print("=====================================================")
    print(f"Evaluation Complete: {run_name}")
    print(f"Ablations -> SPD: {args.use_spd} | Lie: {args.use_lie} | Diag: {args.use_diag} | Fixed: {args.use_fixed}")
    print(f"Mean Spatial Clearance (Wipe %): {mean_wipe_percentage:.2f}%")
    print(f"Binary Success Rate:             {success_rate:.1f}%")
    print("=====================================================")

    env.close()

if __name__ == "__main__":
    main()