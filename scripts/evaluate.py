import gymnasium as gym
from stable_baselines3 import PPO, SAC

import os 
import sys
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))
from hires_vic.envs.gymnasium_wrapper import RobosuiteGymnasiumWrapper

def parse_args():
    import argparse
    parser = argparse.ArgumentParser(description="Train PPO on Robosuite with Geometric Residuals")
    parser.add_argument("--env", type=str, default="Door", help="Name of the Robosuite environment")
    parser.add_argument("--run_name", type=str, default="baseline", help="Name for logging and saving models")
    parser.add_argument("--algorithm", type=str, default="PPO", help="RL Algorithm to use (default: PPO)")
    parser.add_argument("--total_timesteps", type=int, default=500_000, help="Total training timesteps")
    return parser.parse_args()

def main():
    args = parse_args()
    env_name = args.env
    # model_path = "./outputs/models/ppo_door_baseline_final" 
    model_path = f"./outputs/models/{args.algorithm}_{env_name.lower()}_{args.run_name}_final"
    
    # 2. Create Environment with Rendering
    # Note: 'has_renderer=True' is crucial here!
    env = RobosuiteGymnasiumWrapper(
        env_name=env_name,
        robots="Panda",
        task_kwargs={"has_renderer": True, "has_offscreen_renderer": False}
    )
    
    if args.algorithm == "PPO":
        model = PPO.load(model_path)
    elif args.algorithm == "SAC":
        model = SAC.load(model_path)
    
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