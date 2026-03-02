import os 
import sys
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))

import wandb
from wandb.integration.sb3 import WandbCallback

import torch
import numpy as np
from stable_baselines3 import PPO, SAC, TD3
try:
    from sb3_contrib import TQC, RecurrentPPO
except ImportError:
    print("TQC not found. Please install sb3-contrib: `pip install sb3-contrib`")
from stable_baselines3.common.vec_env import SubprocVecEnv, VecMonitor
from stable_baselines3.common.callbacks import CheckpointCallback
from hires_vic.utils.callbacks import RobosuiteLoggingCallback
from stable_baselines3.common.noise import NormalActionNoise, OrnsteinUhlenbeckActionNoise

from robosuite import load_composite_controller_config
import robosuite as suite
from hires_vic.envs.gymnasium_wrapper import RobosuiteGymnasiumWrapper, RobosuitePhysicsWrapper

import robosuite.controllers.parts.controller_factory as factory
from copy import deepcopy
from hires_vic.envs.custom_osc import GRL_OperationalSpaceController 

# 1. Save a reference to Robosuite's original hardcoded factory
original_arm_factory = factory.arm_controller_factory

# 2. Define our custom factory function that intercepts the process
def custom_arm_controller_factory(name, params):
    if name == "GRL_OSC":
        # We must replicate the interpolator logic that Robosuite normally 
        # applies to OSC_POSE so your controller gets the right timing data
        interpolator = None
        if params.get("interpolation") == "linear":
            from robosuite.utils.traj_utils import LinearInterpolator
            interpolator = LinearInterpolator(
                ndim=params["ndim"],
                controller_freq=(1 / params["sim"].model.opt.timestep),
                policy_freq=params["policy_freq"],
                ramp_ratio=params["ramp_ratio"],
            )

        ori_interpolator = None
        if interpolator is not None:
            interpolator.set_states(dim=3)  # Pos control uses dim 3
            ori_interpolator = deepcopy(interpolator)
            ori_interpolator.set_states(ori="euler")
            
        params["control_ori"] = True
        
        # Return YOUR custom controller instead of raising a ValueError!
        return GRL_OperationalSpaceController(
            interpolator_pos=interpolator, 
            interpolator_ori=ori_interpolator, 
            **params
        )

    return original_arm_factory(name, params)

# 3. OVERWRITE the function inside the Robosuite module at runtime!
factory.arm_controller_factory = custom_arm_controller_factory


def parse_args():
    import argparse
    parser = argparse.ArgumentParser(description="Train PPO on Robosuite with Geometric Residuals")
    parser.add_argument("--env", type=str, default="Door", help="Name of the Robosuite environment")
    parser.add_argument("--run_name", type=str, default="baseline", help="Name for logging and saving models")
    parser.add_argument("--algorithm", type=str, default="PPO", help="RL Algorithm to use (default: PPO)")
    parser.add_argument("--n_envs", type=int, default=4, help="Number of parallel environments")
    parser.add_argument("--total_timesteps", type=int, default=1_000_000, help="Total training timesteps")
    parser.add_argument("--checkpoint", type=str, required=False, help="Path to the .zip checkpoint file")
    return parser.parse_args()

def make_env(run_name, env_name, rank, seed=0):
    """
    Utility function for multiprocessed env.
    """
    def _init():
        controller_config = None
        is_vic = "VIC" in run_name

        if is_vic:
            controller_config = load_composite_controller_config(controller="BASIC", robot="panda")
            # print(f"Initial Controller: {controller_config}")
            phantom_parts = ["left", "torso", "head", "base", "legs"]
    
            # 2. Safely remove them from the dictionary if they exist
            for part in phantom_parts:
                controller_config["body_parts"].pop(part, None)


            print(controller_config)
            arm_config = controller_config["body_parts"]["right"]
            # arm_config["type"] = "OSC_POSE"
            arm_config["type"] = "GRL_OSC"

            # "variable_kp": Agent outputs [Pos, Ori, Kp]. Damping (Kd) is auto-calculated.
            # "variable":  Agent outputs [Pos, Ori, Kp, Kd]. Both are learned.
            # "fixed": Agent outputs [Pos, Ori]. Kp is constant. (This was your Experiment 1)
            # arm_config["impedance_mode"] = "variable_kp"
            arm_config["impedance_mode"] = "fixed" 
            
            # 0 = Completely limp (gravity comp only), 300 = Very stiff
            arm_config["kp_limits"] = [10, 200] # TODO: Ask Adriá and Bernard about this!
            
            # 0 = Bouncy, 1 = Critical Damping (No overshoot), >1 = Sluggish
            # We let the agent learn this or auto-scale it.
            arm_config["damping_ratio_limits"] = [1.0, 1.0] # Force critical damping
            # print(f"Using VIC controller config: {controller_config}")
        
        task_kwargs = {
            "has_renderer": False,
            "has_offscreen_renderer": False,
            "use_camera_obs": False,
            "reward_shaping": True,
            "horizon": 300 if env_name == "Door" else 1000, # Shorter episodes for Door
            "control_freq": 20,              #  50ms per step
        }
        
        env = RobosuiteGymnasiumWrapper(
            env_name=env_name,
            robots="Panda",
            controller_configs=controller_config,
            task_kwargs=task_kwargs
        )
        
        stiff_penalty = 0.01 if is_vic else 0.0
        
        env = RobosuitePhysicsWrapper(
            env, 
            stiffness_penalty=stiff_penalty, 
            force_penalty=0.02, 
            max_force_threshold=35.0
        )

        # print(f"Initialized environment: {env_name} | VIC Mode: {is_vic}")
        # print(f"Applying stiffness penalty: {stiff_penalty}")
        # print(f"Max force threshold: 35.0N")
        # print(f"Force penalty: 0.02 per Newton above threshold")
       
        env.reset(seed=seed + rank) # Distinct seed for each worker
        return env
    return _init


def main():
    args = parse_args()

    device = "cuda" if torch.cuda.is_available() else "cpu"
    print(f"Training on {device.upper()} with {args.n_envs} parallel environments.")

    env_name = args.env
    run_name = f'{args.algorithm}_{env_name.lower()}_{args.run_name}'

    wandb.init(
        project="HiRes-VIC",          # Name of your project dashboard
        name=run_name,                # Name of this specific run
        sync_tensorboard=True,        # MAGIC: Automatically uploads SB3 & Custom metrics!
        monitor_gym=True,             # Auto-logs video if your env produces them
        save_code=True,               # Saves your train.py so you know what code you ran
        config={
            "algorithm": args.algorithm,
            "env": args.env,
            "total_timesteps": args.total_timesteps,
            "n_envs": args.n_envs,
            "is_vic": "VIC" in run_name
        }
    )

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
        print(">>> Using PPO...")
        if args.checkpoint:
            print(f"Loading model from checkpoint: {args.checkpoint}")
            # model = RecurrentPPO.load(args.checkpoint, env=env, custom_objects=custom_objects, device=device)
            model = PPO.load(args.checkpoint, env=env, custom_objects=custom_objects, device=device)
        else:
            model = PPO(
                "MlpPolicy", 
                env, 
                verbose=1, 
                tensorboard_log=f"./outputs/logs/{run_name}",
                learning_rate=3e-4,
                batch_size=64,
                # n_steps=1536 // args.n_envs,
                n_steps=512,
                ent_coef=0.01,           # Encourage exploration
                use_sde=True,            # Smooth robotic noise
                sde_sample_freq=8,       # Change noise every 8 steps
                clip_range=0.1,          # Stability for variable Kp
                device=device,
                gamma=0.99,
                n_epochs=20              # More epochs for better convergence with smaller batch size

            )

        # print(f'Clip range: {model.clip_range}')
        
    elif args.algorithm == "SAC":
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
                batch_size=256,
                buffer_size=1_000_000,
                tau=0.002,              # For soft updates of the target network
                target_entropy="auto", # Encourage exploration (tune based on action space)
                train_freq=1,        # Train every step
                gradient_steps=2,    # Take 4 gradient steps to match 4 new data points
                use_sde=False,            # Smooth robotic noise
                # sde_sample_freq=8,
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
                batch_size=256,
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
                use_sde=False,
                # sde_sample_freq=8,
                policy_kwargs=policy_kwargs,
                device=device
            )

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

    logging_callback = RobosuiteLoggingCallback()

    wandb_callback = WandbCallback(
        gradient_save_freq=0,
        model_save_path=None,
        verbose=2,
    )
    
    # Train
    print(f"Starting training for {run_name}...")
    model.learn(
        total_timesteps=remaining_steps, 
        callback=[checkpoint_callback, logging_callback, wandb_callback], 
        reset_num_timesteps=False
    )
    
    # Save Final Model
    model.save(f"./outputs/models/{run_name}_final")
    print("Training Complete!")

if __name__ == "__main__":
    main()