import gymnasium as gym
from stable_baselines3 import PPO, SAC
from stable_baselines3.common.callbacks import CheckpointCallback

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
    run_name = f'{args.algorithm}_{env_name.lower()}_{args.run_name}'
    
    # 2. Create Environment
    # We use the wrapper we created earlier
    env = RobosuiteGymnasiumWrapper(
        env_name=env_name,
        robots="Panda",
        controller_configs=None # Uses OSC_POSE by default
    )
    
    # 3. Create Model
    # For the "Blind" baseline, we use the standard MlpPolicy
    if args.algorithm == "PPO":
        model = PPO(
            "MlpPolicy", 
            env, 
            verbose=1, 
            tensorboard_log=f"./outputs/logs/{run_name}",
            learning_rate=3e-4,
            batch_size=64
        )

    elif args.algorithm == "SAC":
        model = SAC(
            "MlpPolicy",
            env,
            verbose=1,
            tensorboard_log=f"./outputs/logs/{run_name}",
            learning_rate=3e-4,
            batch_size=512
        )
    
    # 4. Setup Callbacks (Save every 10k steps)
    checkpoint_callback = CheckpointCallback(
        save_freq=10000, 
        save_path=f"./outputs/checkpoints/{run_name}",
        name_prefix="model"
    )
    
    # 5. Train
    print(f"Starting training for {run_name}...")
    model.learn(total_timesteps=500_000, callback=checkpoint_callback)
    
    # 6. Save Final Model
    model.save(f"./outputs/models/{run_name}_final")
    print("Training Complete!")

if __name__ == "__main__":
    main()