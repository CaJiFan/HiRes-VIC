import torch
import numpy as np
from stable_baselines3 import PPO, SAC, TD3
from sb3_contrib import TQC, RecurrentPPO

import os 
import sys
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))
from hires_vic.envs.gymnasium_wrapper import RobosuiteGymnasiumWrapper, RobosuitePhysicsWrapper

from robosuite import load_composite_controller_config

def parse_args():
    import argparse
    parser = argparse.ArgumentParser(description="Evaluate RL on Robosuite")
    parser.add_argument("--env", type=str, default="Door", help="Name of the Robosuite environment")
    parser.add_argument("--run_name", type=str, default="baseline", help="Name for logging and saving models")
    parser.add_argument("--algorithm", type=str, default="PPO", help="RL Algorithm to use (default: PPO)")
    return parser.parse_args()

def make_env(run_name, env_name, rank=0, seed=0):
    controller_config = None
    is_vic = "VIC" in run_name

    if is_vic:
        controller_config = load_composite_controller_config(controller="BASIC", robot="panda")
        arm_config = controller_config["body_parts"]["right"]
        arm_config["type"] = "OSC_POSE"
        arm_config["impedance_mode"] = "variable_kp"
        arm_config["kp_limits"] = [10, 200]
        arm_config["damping_ratio_limits"] = [1.0, 1.0] 

    horizon = 500 if env_name == "Door" else 1000
        
    env = RobosuiteGymnasiumWrapper(
        env_name=env_name,
        robots="Panda",
        controller_configs=controller_config,
        task_kwargs={
            "has_renderer": True,
            "has_offscreen_renderer": False,
            "reward_shaping": True,
            "horizon": horizon, 
            "control_freq": 20 # <--- Ensure this matches your 20Hz training frequency!
        }
    )
    
    # ---> ADDED THE PHYSICS WRAPPER <---
    # We add this so we can read the exact same info dict as training
    stiff_penalty = 0.01 if is_vic else 0.0
    env = RobosuitePhysicsWrapper(
        env, 
        stiffness_penalty=stiff_penalty, 
        force_penalty=0.1, 
        max_force_threshold=20.0
    )

    return env

def main():
    args = parse_args()
    device = "cuda" if torch.cuda.is_available() else "cpu"
    env_name = args.env
    
    # Make sure this path exactly matches what your train script outputs
    model_path = f"./outputs/models/{args.algorithm}_{env_name.lower()}_{device}_{args.run_name}_final"
    
    env = make_env(args.run_name, env_name)

    # ---> PROPERLY LOAD RECURRENTPPO <---
    print(f"Loading {args.algorithm} model from {model_path}...")
    if args.algorithm == "PPO":
        # We assume PPO means RecurrentPPO based on your training script
        model = RecurrentPPO.load(model_path, env=env, device=device)
    elif args.algorithm == "SAC":
        model = SAC.load(model_path, env=env, device=device)
    elif args.algorithm == "TD3":
        model = TD3.load(model_path, env=env, device=device)
    elif args.algorithm == "TQC":
        model = TQC.load(model_path, env=env, device=device)
    
    # 4. Evaluation Loop
    obs, _ = env.reset()
    
    # ---> LSTM STATE TRACKING SETUP <---
    lstm_states = None 
    # Episode starts array tells the LSTM when to reset its memory
    episode_starts = np.ones((1,), dtype=bool) 
    
    print("Running evaluation... Press Ctrl+C to stop.")
    
    try:
        while True:
            # ---> PASS STATES IN AND OUT OF PREDICT <---
            if args.algorithm == "PPO":
                action, lstm_states = model.predict(
                    obs, 
                    state=lstm_states, 
                    episode_start=episode_starts, 
                    deterministic=True
                )
            else:
                action, _states = model.predict(obs, deterministic=True)
            
            # Step environment
            obs, reward, terminated, truncated, info = env.step(action)
            env.render()

            # Update episode_starts for the next step
            done = terminated or truncated
            episode_starts = np.array([done], dtype=bool)

            # Optional: Print out the physics info live!
            force = info.get("physics/force_step", 0.0)
            stiffness = info.get("physics/stiffness_step", 0.0)

            print(f"Action: {action}")
            print(f"Reward: {reward:.2f} | Force: {force:.2f} N | Stiff: {stiffness:.1f} | Done: {done}")

            if done:
                obs, _ = env.reset()
                
    except KeyboardInterrupt:
        print("Closing environment...")
        env.close()

if __name__ == "__main__":
    main()