import torch
import numpy as np
from stable_baselines3 import PPO, SAC, TD3
try:
    from sb3_contrib import TQC, RecurrentPPO
except ImportError:
    print("TQC not found. Please install sb3-contrib: `pip install sb3-contrib`")
from stable_baselines3.common.vec_env import SubprocVecEnv, VecMonitor
from stable_baselines3.common.callbacks import CheckpointCallback
from stable_baselines3.common.noise import NormalActionNoise, OrnsteinUhlenbeckActionNoise

from robosuite import load_composite_controller_config

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
    parser.add_argument("--n_envs", type=int, default=4, help="Number of parallel environments")
    parser.add_argument("--total_timesteps", type=int, default=500_000, help="Total training timesteps")
    parser.add_argument("--checkpoint", type=str, required=False, help="Path to the .zip checkpoint file")
    return parser.parse_args()

def make_env(run_name, env_name, rank, seed=0):
    """
    Utility function for multiprocessed env.
    """
    def _init():
        controller_config = None

        if "VIC" in run_name:
            controller_config = load_composite_controller_config(controller="BASIC", robot="panda")
            # print(f"Initial Controller: {controller_config}")

            arm_config = controller_config["body_parts"]["right"]
            arm_config["type"] = "OSC_POSE"

            # "variable_kp": Agent outputs [Pos, Ori, Kp]. Damping (Kd) is auto-calculated.
            # "variable":  Agent outputs [Pos, Ori, Kp, Kd]. Both are learned.
            # "fixed": Agent outputs [Pos, Ori]. Kp is constant. (This was your Experiment 1)
            arm_config["impedance_mode"] = "variable_kp"
            
            # 0 = Completely limp (gravity comp only), 300 = Very stiff
            arm_config["kp_limits"] = [10, 200] # TODO: Ask Adriá and Bernard about this!
            
            # 0 = Bouncy, 1 = Critical Damping (No overshoot), >1 = Sluggish
            # We let the agent learn this or auto-scale it.
            arm_config["damping_ratio_limits"] = [1.0, 1.0] # Force critical damping
            # print(f"Using VIC controller config: {controller_config}")

        
        horizon = 1000 # Default horizon
        if env_name == "Door":
            horizon = 500
        
        env = RobosuiteGymnasiumWrapper(
            env_name=env_name,
            robots="Panda",
            controller_configs=controller_config,
            task_kwargs={
                "has_renderer": False,
                "horizon": horizon, 
                "control_freq": 50
            }
        )
        env.reset(seed=seed + rank) # Distinct seed for each worker
        return env
    return _init


def main():
    args = parse_args()

    # if args.algorithm in ["SAC", "TD3", "TQC"]:
    #     args.n_envs = min(args.n_envs, 4) # Cap envs to 4 to save GPU memory/CPU
    #     print(f"Algorithm is {args.algorithm}: Clamping n_envs to {args.n_envs}")

    device = "cuda" if torch.cuda.is_available() else "cpu"
    print(f"Training on {device.upper()} with {args.n_envs} parallel environments.")

    env_name = args.env
    run_name = f'{args.algorithm}_{env_name.lower()}_{device}_{args.run_name}'

    # Create Vectorized Environment
    env_fns = [make_env(run_name, args.env, i) for i in range(args.n_envs)]
    env = SubprocVecEnv(env_fns)
    env = VecMonitor(env)

    custom_objects = {
        "learning_rate": 3e-4,
        "lr_schedule": lambda _: 3e-4,
        "clip_range": lambda _: 0.1,
    }

    if args.algorithm == "PPO":
        # n_envs = 12
        # env = SubprocVecEnv([make_env(run_name, args.env, i) for i in range(n_envs)])
        # env = VecMonitor(env)
        # model = PPO(
        #     "MlpPolicy", 
        #     env, 
        #     verbose=1, 
        #     tensorboard_log=f"./outputs/logs/{run_name}",
        #     learning_rate=3e-4,
        #     batch_size=2048, # Scale batch size with number of envs
        #     # n_steps=1536 // args.n_envs,
        #     n_steps=512,
        #     ent_coef=0.01,           # Encourage exploration
        #     use_sde=True,            # Smooth robotic noise
        #     sde_sample_freq=8,       # Change noise every 8 steps
        #     clip_range=0.1,          # Stability for variable Kp
        #     device="cpu",
        #     n_epochs=10              # More epochs for better convergence with smaller batch size

        # )
        print(">>> Using RecurrentPPO (LSTM) for PPO...")
        if args.checkpoint:
            print(f"Loading model from checkpoint: {args.checkpoint}")
            model = RecurrentPPO.load(args.checkpoint, env=env, custom_objects=custom_objects, device=device)
        else:
            
            model = RecurrentPPO(
                "MlpLstmPolicy", 
                env,
                verbose=1,
                tensorboard_log=f"./outputs/logs/{run_name}",
                learning_rate=3e-4,
                n_steps=2048,           # Large buffer for LSTM to learn long sequences
                batch_size=512,         # Smaller batch (better for LSTM gradients)
                n_epochs=10,            # [KEEP] Squeeze data efficiency
                ent_coef=0.01,          # [KEEP] Force exploration
                clip_range=0.1,         # [KEEP] Prevent catastrophic forgetting
                use_sde=True,           # [KEEP] Smooth robotic motion
                sde_sample_freq=8,      # [KEEP] Consistency over 8 steps (~0.4s)
                policy_kwargs={
                    "enable_critic_lstm": True, 
                    "lstm_hidden_size": 256,
                    "net_arch": dict(pi=[256, 256], vf=[256, 256]),
                    "optimizer_kwargs": {"weight_decay": 1e-5} # Optional: helps LSTMs generalize
                },
                device=device
            )

        # print(f'Clip range: {model.clip_range}')
        
    elif args.algorithm == "SAC":
        # n_envs = 4
        # env = SubprocVecEnv([make_env(run_name, args.env, i) for i in range(n_envs)])
        # env = VecMonitor(env)
        if args.checkpoint:
            print(f"Loading model from checkpoint: {args.checkpoint}")
            model = SAC.load(args.checkpoint, env=env, custom_objects=custom_objects, device=device)
        else:   
            model = SAC(
                "MlpPolicy",
                env,
                verbose=1,
                tensorboard_log=f"./outputs/logs/{run_name}",
                learning_rate=3e-4,
                batch_size=512,
                buffer_size=1_000_000,
                tau=0.002,              # For soft updates of the target network
                target_entropy=-16.0 , # Encourage exploration (tune based on action space)
                train_freq=1,        # Train every step
                gradient_steps=4,    # Take 4 gradient steps to match 4 new data points
                use_sde=True,            # Smooth robotic noise
                sde_sample_freq=8,
                device=device
            )

    elif args.algorithm == "TD3":
        # TD3 needs explicit action noise for exploration
        n_actions = env.action_space.shape[-1]
        action_noise = OrnsteinUhlenbeckActionNoise(
            mean=np.zeros(n_actions), 
            sigma=0.2 * np.ones(n_actions) # Exploration magnitude
        )

        # action_noise = NormalActionNoise(
        #     mean=np.zeros(n_actions), 
        #     sigma=0.1 * np.ones(n_actions) # Standard deviation (0.1 is standard)
        # )
        if args.checkpoint:
            print(f"Loading model from checkpoint: {args.checkpoint}")
            model = TD3.load(args.checkpoint, env=env, custom_objects=custom_objects, device=device)
        else:
            model = TD3(
                "MlpPolicy",
                env,
                action_noise=action_noise, # Crucial for TD3
                verbose=1,
                tensorboard_log=f"./outputs/logs/{run_name}",
                learning_rate=3e-4,
                buffer_size=1_000_000,
                batch_size=512,
                tau=0.002,
                train_freq=1,
                gradient_steps=1,
                policy_delay=2,         # Update policy every 2 critic updates
                device=device
            )

    elif args.algorithm == "TQC":
        # TQC: Truncated Quantile Critics (Distributional SAC)
        # Great for contact-rich tasks to handle "crash" variance
        
        policy_kwargs = dict(
            n_critics=5,             # Number of critic networks (default: 2 in SAC)
            n_quantiles=25,
            net_arch=[256, 256],
        )
        if args.checkpoint:
            print(f"Loading model from checkpoint: {args.checkpoint}")
            model = TQC.load(args.checkpoint, env=env, custom_objects=custom_objects, device=device)
        else:
            model = TQC(
                "MlpPolicy",
                env,
                top_quantiles_to_drop_per_net=2, # Safety: Drop top 2.5% optimistic estimates
                verbose=1,
                tensorboard_log=f"./outputs/logs/{run_name}",
                learning_rate=3e-4,
                buffer_size=1_000_000,
                batch_size=512,
                tau=0.002,
                train_freq=1,
                gradient_steps=1,
                ent_coef="auto",
                use_sde=True,              # TQC supports SDE!
                sde_sample_freq=8,
                policy_kwargs=policy_kwargs,
                device=device
            )

        # ------------------------------------------------------------------
   
    # UNIVERSAL REPLAY BUFFER CHECK (For SAC, TD3, TQC)
    # ------------------------------------------------------------------
    if args.checkpoint and args.algorithm in ["SAC", "TD3", "TQC"]:
        # Check if buffer is empty
        if model.replay_buffer is None or model.replay_buffer.size() == 0:
            print(f">>> WARNING: {args.algorithm} Replay Buffer is EMPTY.")
            print("    The agent has 'Amnesia'. It knows the policy but forgot past experiences.")
            print("    Training will be flat for a while until it collects new data.")
    # ------------------------------------------------------------------


    remaining_steps = args.total_timesteps - model.num_timesteps
    if remaining_steps <= 0:
        print(f"Model already trained for {model.num_timesteps} steps. Target is {args.total_timesteps}.")
        # print("Increasing target by +1M steps...")
        # remaining_steps = 1_000_000
    
    print(f'Running for {remaining_steps} timesteps...')

    checkpoint_callback = CheckpointCallback(
        save_freq=1_000_000, 
        save_path=f"./outputs/checkpoints/{run_name}",
        name_prefix="model",
        save_replay_buffer=True,
        save_vecnormalize=True
    )
    
    # Train
    print(f"Starting training for {run_name}...")
    model.learn(
        total_timesteps=remaining_steps, 
        callback=checkpoint_callback, 
        reset_num_timesteps=False
    )
    
    # Save Final Model
    model.save(f"./outputs/models/{run_name}_final")
    print("Training Complete!")

if __name__ == "__main__":
    main()