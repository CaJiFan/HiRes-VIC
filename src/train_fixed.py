import os
import sys
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))

import argparse
import numpy as np
import torch

import warnings
warnings.filterwarnings("ignore", message=".*precision lowered by casting to float32.*")

# Robosuite
from hires_vic.utils.callbacks import RobosuiteLoggingCallback
import robosuite as suite
from robosuite.wrappers import GymWrapper
from robosuite import load_composite_controller_config
from hires_vic import envs
from hires_vic.wrappers import WipeMetricWrapper, GeometricWrapper
from hires_vic.envs.riemannian_controller import RiemannianController
import robosuite.controllers.parts.controller_factory as factory
factory.arm_controllers.OperationalSpaceController = RiemannianController

# Stable Baselines 3
from stable_baselines3 import SAC
from stable_baselines3.common.callbacks import CheckpointCallback, EvalCallback
from stable_baselines3.common.monitor import Monitor
from stable_baselines3.common.utils import set_random_seed
from stable_baselines3.common.vec_env import SubprocVecEnv, VecMonitor, DummyVecEnv

# Optional: WandB for logging (highly recommended)
import wandb
from wandb.integration.sb3 import WandbCallback


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
    "num_markers": 5,  # How many particles of dirt to generate in the environment
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


def parse_args():
    parser = argparse.ArgumentParser(description="Train SAC on robosuite TiltedWipe with different controllers and numbers of markers. Logs to WandB and Tensorboard.")
    
    # Environment Args
    parser.add_argument("--env", type=str, default="TiltedWipe", help="Robosuite environment name")
    parser.add_argument("--seed", type=int, default=0, help="Random seed")
    parser.add_argument("--total_timesteps", type=int, default=1_000_000, help="Total training steps")
    parser.add_argument("--num_markers", type=int, default=5, help="Number of dirt markers for Wipe task")
    parser.add_argument("--n_envs", type=int, default=8, help="Number of parallel environments")
    parser.add_argument("--algorithm", type=str, default="SAC", help="Algorithm name for logging")
    parser.add_argument("--use_spd", action="store_true", help="Enable Riemannian SPD stiffness")
    parser.add_argument("--use_lie", action="store_true", help="Enable Lie Group orientation prior")
    parser.add_argument("--use_diag", action="store_true", help="Enable Diagonal SPD Riemannian Manifold")
    parser.add_argument("--use_fixed", action="store_true", help="Enable fixed stiffness (no VIC, but still learn the residual on top of the fixed controller)")
    parser.add_argument("--fixed_kp", type=int, default=150, help="Fixed kp value")
    parser.add_argument("--gamma", type=float, default=0.99, help="gamma parameter for SAC algorithm")
    parser.add_argument("--use_llm_prior", action="store_true")
    parser.add_argument("--llm_backend", type=str, default="ollama", choices=["openai", "ollama"])
    parser.add_argument("--llm_query_interval", type=int, default=50)
    parser.add_argument("--llm_prior_weight", type=float, default=0.4)
    
    # Logging Args
    parser.add_argument("--run_name", type=str, required=True, help="Name of the run for logging/saving")
    
    return parser.parse_args()

def make_env(args, is_eval=False, rank=0, seed=0):
    def _init():
        # Load the base OSC_POSE config
        controller_config = load_composite_controller_config(controller="BASIC", robot="panda")

        phantom_parts = ["left", "torso", "head", "base", "legs"]
        for part in phantom_parts:
            controller_config["body_parts"].pop(part, None)
        
        arm_config = controller_config["body_parts"]["right"]
        arm_config["type"] = "OSC_POSE"
        arm_config["impedance_mode"] = "riemannian_kp" if args.use_spd else "fixed" if args.use_fixed else "variable_kp"
        arm_config["kp_limits"] = [1, 300] # default 
        arm_config["damping_ratio_limits"] = [1.0, 1.0] 

        if args.use_fixed:
            arm_config["kp"] = args.fixed_kp
            print(f"Initializing controller with fixed kp of {args.fixed_kp}")

        WIPE_TASK_CONFIG["num_markers"] = args.num_markers
        WIPE_TASK_CONFIG["use_condensed_obj_obs"] = True
        
        env = suite.make(
            env_name=args.env,
            robots="Panda",
            controller_configs=controller_config,
            has_renderer=False,
            use_object_obs=True,
            has_offscreen_renderer=False,
            use_camera_obs=False,
            reward_shaping=True,
            horizon=300,
            # Task specific kwargs
            task_config=WIPE_TASK_CONFIG
        )
        
        env = GymWrapper(env)
        env = GeometricWrapper(
            env=env,
            use_spd_manifold=args.use_spd,
            use_lie_group=args.use_lie,
            use_diag_manifold=args.use_diag,
            use_fixed=args.use_fixed,
            is_eval=is_eval,
            use_llm_prior=args.use_llm_prior,
            llm_backend=args.llm_backend,
            llm_query_interval=args.llm_query_interval,
            llm_prior_weight=args.llm_prior_weight,
        )

        env.reset(seed=seed + rank)
        return env
    
    return _init

def setup_evaluation_callback(args, run_name):
    # Evaluation 
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

    return eval_callback

def main():
    args = parse_args()
    set_random_seed(args.seed)

    device = "cuda" if torch.cuda.is_available() else "cpu"
    print(f"Training on {device.upper()} with {args.n_envs} parallel environments.")
    print(f"🚀 Starting training for {args.env} with SEED: {args.seed} and gamma: {args.gamma}")

    wandb_run_name = f'{args.algorithm}_{args.env.upper()}_{args.run_name}' 
    run_name = f'{wandb_run_name}_SEED_{args.seed}'
    
    run = wandb.init(
        project="HiRes-VIC",
        name=wandb_run_name,
        config=vars(args),
        sync_tensorboard=True,
        monitor_gym=True,
    )

    env_fns = [make_env(args, is_eval=False, rank=i) for i in range(args.n_envs)]
    env = SubprocVecEnv(env_fns)
    env = VecMonitor(env)

    model = SAC(
        "MlpPolicy",
        env,
        verbose=1,
        tensorboard_log=f"./outputs/logs/{run_name}",
        learning_rate=3e-4,
        batch_size=512,
        buffer_size=1_000_000,
        tau=0.002,                  # For soft updates of the target network
        target_entropy="auto",      # Encourage exploration (tune based on action space)
        gamma=args.gamma,           # default 0.99
        train_freq=1,               # Train every step
        gradient_steps=args.n_envs, # Take 4 gradient steps to match 4 new data points
        use_sde=False,              # Smooth robotic noise
        # sde_sample_freq=8,
        seed=args.seed,
        device=device
    )

    eval_callback = setup_evaluation_callback(args, run_name)
    
    logging_callback = RobosuiteLoggingCallback()

    wandb_callback = WandbCallback(
        gradient_save_freq=0,
        model_save_path=None,
        verbose=2,
    )

    # 6. Train!
    print(f"Starting training for {args.total_timesteps} steps...")
    model.learn(
        total_timesteps=args.total_timesteps,
        callback=[logging_callback, eval_callback, wandb_callback],
        reset_num_timesteps=False
    )

    # 7. Save final model and cleanup
    model.save(f"./outputs/models/{run_name}_final")
    env.close()
    run.finish()
    print("Training Complete!")

if __name__ == "__main__":
    main()