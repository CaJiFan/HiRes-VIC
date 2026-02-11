import torch
from stable_baselines3 import PPO, SAC, TD3
from sb3_contrib import TQC

import os 
import sys
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))
from hires_vic.envs.gymnasium_wrapper import RobosuiteGymnasiumWrapper

from robosuite import load_composite_controller_config

def parse_args():
    import argparse
    parser = argparse.ArgumentParser(description="Train PPO on Robosuite with Geometric Residuals")
    parser.add_argument("--env", type=str, default="Door", help="Name of the Robosuite environment")
    parser.add_argument("--run_name", type=str, default="baseline", help="Name for logging and saving models")
    parser.add_argument("--algorithm", type=str, default="PPO", help="RL Algorithm to use (default: PPO)")
    return parser.parse_args()

def make_env(run_name, env_name, rank, seed=0):
    controller_config = None

    if "VIC" in run_name:
        controller_config = load_composite_controller_config(controller="BASIC", robot="panda")

        arm_config = controller_config["body_parts"]["right"]
        arm_config["type"] = "OSC_POSE"
        arm_config["impedance_mode"] = "variable_kp"
        arm_config["kp_limits"] = [10, 200] # TODO: Ask Adriá and Bernard about this!
        arm_config["damping_ratio_limits"] = [1.0, 1.0] # Force critical damping

    horizon = 1000 # Default horizon
    if env_name == "Door":
        horizon = 500
        
    env = RobosuiteGymnasiumWrapper(
        env_name=env_name,
        robots="Panda",
        controller_configs=controller_config,
        task_kwargs={
            "has_renderer": True,
            "has_offscreen_renderer": False,
            "horizon": horizon, 
            "control_freq": 50
        }
    )
    # env.reset(seed=seed + rank) # Distinct seed for each worker
    return env

def main():
    args = parse_args()

    device = "cuda" if torch.cuda.is_available() else "cpu"
    env_name = args.env
    # model_path = "./outputs/models/ppo_door_baseline_final" 
    model_path = f"./outputs/models/{args.algorithm}_{env_name.lower()}_{device}_{args.run_name}_final"
    
    env = make_env(args.run_name, env_name, rank=0)

    if args.algorithm == "PPO":
        model = PPO.load(model_path)
    elif args.algorithm == "SAC":
        model = SAC.load(model_path)
    elif args.algorithm == "TD3":
        model = TD3.load(model_path)
    elif args.algorithm == "TQC":
        model = TQC.load(model_path)
    
    
    # 4. Evaluation Loop
    obs, _ = env.reset()
    print("Running evaluation... Press Ctrl+C to stop.")
    
    try:
        while True:
            # Predict action (deterministic=True is better for evaluation)
            action, _states = model.predict(obs, deterministic=True)
            
            # Step environment
            obs, reward, terminated, truncated, info = env.step(action)
            env.render()

            print(f"Action: {action}, Reward: {reward:.2f}, Terminated: {terminated}, Truncated: {truncated}")

            if terminated or truncated:
                obs, _ = env.reset()
                
    except KeyboardInterrupt:
        print("Closing environment...")
        env.close()

if __name__ == "__main__":
    main()