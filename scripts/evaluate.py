import torch
import numpy as np
from stable_baselines3 import PPO, SAC, TD3
from sb3_contrib import TQC, RecurrentPPO

from stable_baselines3.common.evaluation import evaluate_policy
from stable_baselines3.common.monitor import Monitor
from stable_baselines3.common.vec_env import DummyVecEnv

import os 
import sys
import time
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))
from hires_vic.envs.gymnasium_wrapper import RobosuiteGymnasiumWrapper, RobosuitePhysicsWrapper
from robosuite import load_composite_controller_config

def parse_args():
    import argparse
    parser = argparse.ArgumentParser(description="Evaluate RL on Robosuite")
    parser.add_argument("--env", type=str, default="Door", help="Name of the Robosuite environment")
    parser.add_argument("--run_name", type=str, default="baseline", help="Name for logging and saving models")
    parser.add_argument("--algorithm", type=str, default="PPO", help="RL Algorithm to use (default: PPO)")
    parser.add_argument("--episodes", type=int, default=10, help="Number of episodes to evaluate")
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

    horizon = 300 if env_name == "Door" else 1000
        
    env = RobosuiteGymnasiumWrapper(
        env_name=env_name,
        robots="Panda",
        controller_configs=controller_config,
        task_kwargs={
            "has_renderer": True,
            "has_offscreen_renderer": False,
            "reward_shaping": True,
            "horizon": horizon, 
            "control_freq": 20
        }
    )
    
    stiff_penalty = 0.01 if is_vic else 0.0
    env = RobosuitePhysicsWrapper(
        env, 
        stiffness_penalty=stiff_penalty, 
        force_penalty=0.1, 
        max_force_threshold=20.0
    )

    env = Monitor(env)
    return env

def render_delay_callback(_locals, _globals):
    time.sleep(0.01)
    return True


def main():
    args = parse_args()
    device = "cuda" if torch.cuda.is_available() else "cpu"
    env_name = args.env
    
    model_path = f"./outputs/models/{args.algorithm}_{env_name.lower()}_{args.run_name}_final"
    
    # Create the environment and wrap it in a VecEnv (required for evaluate_policy)
    env = make_env(args.run_name, env_name)
    vec_env = DummyVecEnv([lambda: env])

    print(f"Loading {args.algorithm} model from {model_path}...")
    if args.algorithm == "PPO":
        model = PPO.load(model_path, env=vec_env, device=device)
    elif args.algorithm == "SAC":
        model = SAC.load(model_path, env=vec_env, device=device)
    elif args.algorithm == "TD3":
        model = TD3.load(model_path, env=vec_env, device=device)
    elif args.algorithm == "TQC":
        model = TQC.load(model_path, env=vec_env, device=device)
    
    print(f"Running formal evaluation over {args.episodes} episodes...")
    

    # This replaces your entire while loop!
    mean_reward, std_reward = evaluate_policy(
        model, 
        vec_env, 
        n_eval_episodes=args.episodes, 
        deterministic=True, 
        render=True,
        callback=render_delay_callback
    )

    # Note: evaluate_policy automatically prints progress if you pass return_episode_rewards=False
    print("=====================================================")
    print(f"Evaluation Complete for {args.algorithm} on {env_name}")
    print(f"Mean Reward: {mean_reward:.2f} +/- {std_reward:.2f}")
    print("=====================================================")

    vec_env.close()

if __name__ == "__main__":
    main()