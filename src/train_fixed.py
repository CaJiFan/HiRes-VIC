import os
import sys
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))

import argparse
import random
import numpy as np
import torch
import logging
from scipy.spatial.transform import Rotation as R
from robosuite.utils.log_utils import ROBOSUITE_DEFAULT_LOGGER

# Suppress all robosuite warnings (like joint limits and macro files)
ROBOSUITE_DEFAULT_LOGGER.setLevel(logging.ERROR)

import warnings
warnings.filterwarnings("ignore", message=".*precision lowered by casting to float32.*")

import warnings
warnings.filterwarnings("ignore")

# Robosuite
from hires_vic.utils.callbacks import RobosuiteLoggingCallback, VideoRecorderCallback
import robosuite as suite
from robosuite.wrappers import GymWrapper
from robosuite import load_composite_controller_config
from hires_vic import envs
from hires_vic.wrappers import GeometricWrapper, FixedGripperWrapper, RobosuiteTeleportWrapper
from hires_vic.envs.riemannian_controller import RiemannianController
import robosuite.controllers.parts.controller_factory as factory
factory.arm_controllers.OperationalSpaceController = RiemannianController

# Stable Baselines 3
from stable_baselines3 import SAC
from stable_baselines3.common.callbacks import CheckpointCallback, EvalCallback, BaseCallback
from stable_baselines3.common.monitor import Monitor
from stable_baselines3.common.utils import set_random_seed
from stable_baselines3.common.vec_env import SubprocVecEnv, VecMonitor, DummyVecEnv

# Optional: WandB for logging (highly recommended)
import wandb
from wandb.integration.sb3 import WandbCallback


DEFAULT_WIPE_TASK_CONFIG = {
    "arm_limit_collision_penalty": -10.0,
    "wipe_contact_reward": 0.01,
    "unit_wiped_reward": 50.0,
    "ee_accel_penalty": 0,
    "excess_force_penalty_mul": 0.05,
    "distance_multiplier": 5.0,
    "distance_th_multiplier": 5.0,
    "table_full_size": [0.5, 0.8, 0.05],
    "table_offset": [0.15, 0, 0.9],
    "table_friction": [0.03, 0.005, 0.0001],
    "table_friction_std": 0,
    "table_height": 0.0,
    "table_height_std": 0.0,
    "line_width": 0.04,
    "two_clusters": False,
    "coverage_factor": 0.6,
    "num_markers": 5,
    "contact_threshold": 1.0,
    "pressure_threshold": 0.5,
    "pressure_threshold_max": 60.0,
    "print_results": False,
    "get_info": False,
    "use_robot_obs": True,
    "use_contact_obs": True,
    "early_terminations": True,
    "use_condensed_obj_obs": True,
}


def load_wipe_task_config():
    cfg_path = os.path.abspath(os.path.join(os.path.dirname(__file__), '..', 'configs', 'wipe_task_config.yaml'))
    try:
        import yaml
        with open(cfg_path, 'r') as f:
            cfg = yaml.safe_load(f)
            if isinstance(cfg, dict):
                return cfg
    except Exception:
        pass
    return DEFAULT_WIPE_TASK_CONFIG.copy()

def wipe_task_metrics_fn(env, info):
    metrics = {}
    try:
        raw_env = getattr(env, 'unwrapped', env)
        inner = getattr(raw_env, 'unwrapped', getattr(raw_env, 'env', raw_env))
        total_markers = getattr(inner, 'num_markers', None)
        wiped_markers = getattr(inner, 'wiped_markers', None)
        if total_markers is not None and wiped_markers is not None:
            percent_wiped = len(wiped_markers) / total_markers if total_markers > 0 else 0.0
            metrics['physics/raw_wipe_percentage'] = float(percent_wiped)
    except Exception:
        pass
    return metrics

def nutassembly_task_metrics_fn(env, info):
    metrics = {}
    try:
        raw_env = getattr(env, 'unwrapped', env)
        inner = getattr(raw_env, 'unwrapped', getattr(raw_env, 'env', raw_env))

        if 'success' in info:
            s = info['success']
            if hasattr(s, 'mean'):
                metrics['physics/nut_success'] = float(s.mean())
            else:
                try:
                    metrics['physics/nut_success'] = float(s)
                except Exception:
                    pass

        if hasattr(inner, 'assembled'):
            a = getattr(inner, 'assembled')
            if isinstance(a, bool):
                metrics['physics/nut_assembled'] = float(a)
            elif hasattr(a, '__len__'):
                metrics['physics/nut_assembled_count'] = float(len(a))

        if hasattr(inner, 'nuts'):
            nuts = getattr(inner, 'nuts')
            total = len(nuts) if hasattr(nuts, '__len__') else None
            assembled = 0
            try:
                for n in nuts:
                    if getattr(n, 'is_inserted', False) or getattr(n, 'inserted', False):
                        assembled += 1
            except Exception:
                assembled = 0
            if total:
                metrics['physics/raw_assembly_percentage'] = float(assembled) / total
                metrics['physics/nut_assembled_count'] = float(assembled)
    except Exception:
        pass
    return metrics

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
    parser.add_argument("--horizon", type=int, default=150, help="Horizon parameter for SAC algorithm")
    parser.add_argument("--use_llm_prior", action="store_true")
    parser.add_argument("--llm_backend", type=str, default="ollama", choices=["openai", "ollama"])
    parser.add_argument("--llm_query_interval", type=int, default=50)
    parser.add_argument("--llm_prior_weight", type=float, default=0.4)
    parser.add_argument("--record_video", action="store_true",
                        help="Record one eval episode as wandb video at each eval checkpoint")
    parser.add_argument("--video_fps", type=int, default=30)
    parser.add_argument("--primitive_init", type=str, default="teleport",
                        choices=["none", "teleport", "scripted", "both"],
                        help="Initialize episodes with a motion primitive: 'teleport', 'scripted', 'both', or 'none'.")
    parser.add_argument("--quat_debug", action="store_true",
                        help="Print quaternion diagnostic candidates for nut handle orientation.")
    
    # Logging Args
    parser.add_argument("--run_name", type=str, required=True, help="Name of the run for logging/saving")
    
    return parser.parse_args()

def make_video_env(args):
    """Create a single Robosuite env suitable for recording RGB frames.
    This is a best-effort offscreen renderer setup; it may require the
    environment to support offscreen rendering (has_offscreen_renderer=True).
    """
    controller_config = load_composite_controller_config(controller="BASIC", robot="panda")
    phantom_parts = ["left", "torso", "head", "base", "legs"]
    for part in phantom_parts:
        controller_config["body_parts"].pop(part, None)
    arm_config = controller_config["body_parts"]["right"]
    arm_config["type"] = "OSC_POSE"
    arm_config["impedance_mode"] = "riemannian_kp" if args.use_spd else "fixed" if args.use_fixed else "variable_kp"
    arm_config["kp_limits"] = [1, 300]
    arm_config["damping_ratio_limits"] = [1.0, 1.0]
    if args.use_fixed:
        arm_config["kp"] = args.fixed_kp

   # Determine task-specific kwargs / metrics
    env_lower = args.env.lower() if isinstance(args.env, str) else ''
    task_config = None
    task_metrics = None
    task_type = None
    is_eval = True

    if 'wipe' in env_lower:
        task_type = 'wipe'
        task_config = load_wipe_task_config()
        task_config["num_markers"] = args.num_markers
        task_config["use_condensed_obj_obs"] = True
        task_metrics = wipe_task_metrics_fn
    elif 'nutassembly' in env_lower:
        task_type = 'nutassembly'
        task_metrics = nutassembly_task_metrics_fn

    task_kwargs = {}
    if task_config is not None:
        task_kwargs['task_config'] = task_config

    env = suite.make(
        env_name=args.env,
        robots="Panda",
        controller_configs=controller_config,
        has_renderer=False,
        use_object_obs=True,
        has_offscreen_renderer=True,
        use_camera_obs=True,
        camera_names="frontview",    # "frontview" or "agentview" are best
        reward_shaping=True,
        horizon=args.horizon,
        **task_kwargs
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
            task_type=task_type,
            task_metrics_fn=task_metrics,
        )

    try:
        # if getattr(args, 'primitive_init', 'none') in ('scripted', 'both') and ('nutassembly' in env_lower or 'peg' in env_lower):
        #     env = RobosuiteScriptedPrimitiveWrapper(env, setup_steps=90, is_eval=is_eval)
        if getattr(args, 'primitive_init', 'none') in ('teleport', 'both') and ('nutassembly' in env_lower or 'peg' in env_lower):
            env = RobosuiteTeleportWrapper(env, setup_steps=140, is_eval=is_eval)
            env = FixedGripperWrapper(env)
    except Exception as e:
        print('Exception setting up primitive', e)
        pass

    return env

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

        # Determine task-specific kwargs / metrics
        env_lower = args.env.lower() if isinstance(args.env, str) else ''
        task_config = None
        task_metrics = None
        task_type = None

        if 'wipe' in env_lower:
            task_type = 'wipe'
            task_config = load_wipe_task_config()
            task_config["num_markers"] = args.num_markers
            task_config["use_condensed_obj_obs"] = True
            task_metrics = wipe_task_metrics_fn
        elif 'nutassembly' in env_lower:
            task_type = 'nutassembly'
            task_metrics = nutassembly_task_metrics_fn

        task_kwargs = {}
        if task_config is not None:
            task_kwargs['task_config'] = task_config

        env = suite.make(
            env_name=args.env,
            robots="Panda",
            controller_configs=controller_config,
            has_renderer=False,
            use_object_obs=True,
            has_offscreen_renderer=False,
            use_camera_obs=False,
            reward_shaping=True,
            horizon=args.horizon,
            **task_kwargs
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
            task_type=task_type,
            task_metrics_fn=task_metrics,
        )

        # for the NutAssembly envs 
        try:
            # if getattr(args, 'primitive_init', 'none') in ('scripted', 'both') and ('nutassembly' in env_lower or 'peg' in env_lower):
            #     env = RobosuiteScriptedPrimitiveWrapper(env, setup_steps=90, is_eval=is_eval)
            if getattr(args, 'primitive_init', 'none') in ('teleport', 'both') and ('nutassembly' in env_lower or 'peg' in env_lower):
                env = RobosuiteTeleportWrapper(env, setup_steps=140, is_eval=is_eval)
                env = FixedGripperWrapper(env)
        except Exception as e:
            print(f"Error occurred while initializing primitive wrapper: {e}")
            pass

        # env.reset(seed=seed + rank)
        env.action_space.seed(seed + rank)
        env.observation_space.seed(seed + rank)
        np.random.seed(seed + rank)
        random.seed(seed + rank)

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
        eval_freq=max(160_000 // args.n_envs, 1),
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

    # Optional: create a single env for recording evaluation videos
    video_env = None
    video_callback = None
    if getattr(args, 'record_video', False):
        try:
            video_env = make_video_env(args)
            video_callback = VideoRecorderCallback(video_env, eval_freq=eval_callback.eval_freq, fps=args.video_fps, primitive_init=args.primitive_init, quat_debug=args.quat_debug)
            print("#### Creating video environment for recording...")
        except Exception as e:
            print(f"Failed to create video env: {e}")

    # 6. Train!
    print(f"Starting training for {args.total_timesteps} steps...")
    callbacks = [logging_callback, eval_callback, wandb_callback]
    if video_callback is not None:
        callbacks.append(video_callback)

    model.learn(
        total_timesteps=args.total_timesteps,
        callback=callbacks,
        reset_num_timesteps=False
    )

    # 7. Save final model and cleanup
    model.save(f"./outputs/models/{run_name}_final")
    env.close()
    if video_env is not None:
        try:
            video_env.close()
        except Exception:
            pass
    run.finish()
    print("Training Complete!")

if __name__ == "__main__":
    main()