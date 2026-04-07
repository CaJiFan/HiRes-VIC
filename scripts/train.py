import os 
import sys
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))

import wandb
from wandb.integration.sb3 import WandbCallback

import torch
import numpy as np
from copy import deepcopy

from robosuite import load_composite_controller_config
import robosuite.controllers.parts.controller_factory as factory

from stable_baselines3 import PPO, SAC, TD3
from stable_baselines3.common.vec_env import SubprocVecEnv, VecMonitor, DummyVecEnv
from stable_baselines3.common.callbacks import CheckpointCallback, EvalCallback
from stable_baselines3.common.utils import set_random_seed
from stable_baselines3.common.noise import NormalActionNoise, OrnsteinUhlenbeckActionNoise
from sb3_contrib import TQC, RecurrentPPO

from hires_vic.envs.riemannian_controller import RiemannianController
from hires_vic.utils.callbacks import RobosuiteLoggingCallback
from hires_vic.envs.gymnasium_wrapper import RobosuiteGymnasiumWrapper, RobosuitePhysicsWrapper

factory.arm_controllers.OperationalSpaceController = RiemannianController

def parse_args():
    import argparse
    parser = argparse.ArgumentParser(description="Train PPO on Robosuite with Geometric Residuals")
    parser.add_argument("--env", type=str, default="Door", help="Name of the Robosuite environment")
    parser.add_argument("--run_name", type=str, default="baseline", help="Name for logging and saving models")
    parser.add_argument("--algorithm", type=str, default="PPO", help="RL Algorithm to use (default: PPO)")
    parser.add_argument("--n_envs", type=int, default=8, help="Number of parallel environments")
    parser.add_argument("--stiff_penalty", type=float, default=0.01, help="Number of parallel environments")
    parser.add_argument("--total_timesteps", type=int, default=5_000_000, help="Total training timesteps")
    parser.add_argument("--use_spd", action="store_true", help="Enable Riemannian SPD stiffness")
    parser.add_argument("--use_lie", action="store_true", help="Enable Lie Group orientation prior")
    parser.add_argument("--use_diag", action="store_true", help="Enable Diagonal SPD Riemannian Manifold")
    parser.add_argument("--kp_max", type=float, default=300.0, help="Maximum stiffness limit (N/m)")
    parser.add_argument("--kp_min", type=float, default=0.0, help="Minimum stiffness limit (N/m)")
    parser.add_argument("--use_condensed_obj_obs", action="store_true", help="Enable condensed object observation representation")
    parser.add_argument("--seed", type=int, default=1, help="Random seed (e.g., 1, 2, 3)")
    parser.add_argument("--checkpoint", type=str, required=False, help="Path to the .zip checkpoint file")
    parser.add_argument("--num_markers", type=int, default=10, help="Number of dirt particles in the environment")
    return parser.parse_args()

def make_env(args, is_eval=False, rank=0, seed=0):
    """
    Utility function for multiprocessed env.
    """
    def _init():
        controller_config = None
        # is_vic = "VIC" in run_name
        env_name = args.env
        is_vic = True
        # kp_limits = [1, 1000] #[20, 200]
        kp_limits = [args.kp_min, args.kp_max]

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
            "has_offscreen_renderer": False,
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
                "num_markers": args.num_markers,  # How many particles of dirt to generate in the environment
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
                "use_condensed_obj_obs": args.use_condensed_obj_obs,  # Whether to use condensed object observation representation (only applicable if obj obs is active)

            }
        }

        if "TILTED" in env_name.upper():
            print(f">>> Using tilted variant with 45 degree tilt and condensed {args.use_condensed_obj_obs}...")
            task_kwargs["tilt_angle_degrees"] = 45.0
        
        env = RobosuiteGymnasiumWrapper(
            env_name=env_name,
            robots="Panda",
            controller_configs=controller_config,
            task_kwargs=task_kwargs,
            use_spd_manifold=args.use_spd,
            use_lie_group=args.use_lie,
            use_diag_manifold=args.use_diag
        )
        
        
        env = RobosuitePhysicsWrapper(
            env, 
            is_eval=is_eval,
            stiffness_penalty=args.stiff_penalty, 
            force_penalty=0.02, # 0.02 
            max_force_threshold=35.0
        )

       
        env.reset(seed=seed + rank) # Distinct seed for each worker
        return env
    return _init

def main():
    args = parse_args()
    set_random_seed(args.seed)

    device = "cuda" if torch.cuda.is_available() else "cpu"
    print(f"Training on {device.upper()} with {args.n_envs} parallel environments.")
    print(f"🚀 Starting training for {args.env} with SEED: {args.seed}")

    env_name = args.env
    # Not including seed in the run name so that we can group different seeds together in WandB UI.
    # The seed is logged as a config parameter instead.
    wandb_run_name = f'{args.algorithm}_{env_name.upper()}_{args.run_name}' 
    run_name = f'{wandb_run_name}_SEED_{args.seed}'

    wandb.init(
        project="HiRes-VIC",
        name=wandb_run_name,
        sync_tensorboard=True,
        monitor_gym=True,
        save_code=True,
        config={
            "algorithm": args.algorithm,
            "env": args.env,
            "total_timesteps": args.total_timesteps,
            "n_envs": args.n_envs,
            "is_vic": True,
            "use_spd": args.use_spd,
            "use_lie": args.use_lie,
            "use_diag": args.use_diag,
            "kp_max": args.kp_max
        }
    )

    # Create Vectorized Environment
    env_fns = [make_env(args, is_eval=False, rank=i) for i in range(args.n_envs)]
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
                seed=args.seed,
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
                batch_size=512,
                buffer_size=1_000_000,
                tau=0.002,              # For soft updates of the target network
                target_entropy="auto", # Encourage exploration (tune based on action space)
                train_freq=1,        # Train every step
                gradient_steps=args.n_envs,    # Take 4 gradient steps to match 4 new data points
                use_sde=False,            # Smooth robotic noise
                # sde_sample_freq=8,
                seed=args.seed,
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
                seed=args.seed,
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
                seed=args.seed,
                device=device
            )

        # ------------------------------------------------------------------
   

    remaining_steps = args.total_timesteps - model.num_timesteps
    # if remaining_steps <= 0:
    #     print(f"Model already trained for {model.num_timesteps} steps. Target is {args.total_timesteps}.")
    #     # print("Increasing target by +1M steps...")
    #     # remaining_steps = 1_000_000
    
    print(f'Running for {remaining_steps} timesteps...')
    eval_env_fn = make_env(args, is_eval=True, rank=0, seed=42)
    eval_env = DummyVecEnv([eval_env_fn])
    eval_env = VecMonitor(eval_env)
    
    eval_callback = EvalCallback(
        eval_env,
        best_model_save_path=f"./logs/best_models/{run_name}/",
        log_path=f"./logs/eval/{run_name}/",
        eval_freq=max(200_000 // args.n_envs, 1),
        n_eval_episodes=10, # Run 10 deterministic episodes
        deterministic=True,
        render=False
    )
    # checkpoint_callback = CheckpointCallback(
    #     save_freq=1_000_000, 
    #     save_path=f"./outputs/checkpoints/{run_name}",
    #     name_prefix="model",
    #     save_replay_buffer=True,
    #     save_vecnormalize=True
    # )

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
        callback=[logging_callback, wandb_callback, eval_callback], 
        reset_num_timesteps=False
    )
    
    # Save Final Model
    model.save(f"./outputs/models/{run_name}_final")
    print("Training Complete!")

if __name__ == "__main__":
    main()