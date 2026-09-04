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
from hires_vic.wrappers import (
    GeometricWrapper,
    FixedGripperWrapper,
    RobosuiteTeleportWrapper,
    WipeTeleportWrapper,
    WipeDomainRandomizationWrapper,
)
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


TELEPORT_STEPS = 150

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
        # Peel back the wrappers to get to the core Robosuite environment
        raw_env = getattr(env, 'unwrapped', env)
        inner = getattr(raw_env, 'unwrapped', getattr(raw_env, 'env', raw_env))

        # 1. Check for standard wrapper 'success' info (Safe fallback)
        if 'success' in info:
            s = info['success']
            if hasattr(s, 'mean'):
                metrics['physics/nut_success'] = float(s.mean())
            else:
                try:
                    metrics['physics/nut_success'] = float(s)
                except Exception:
                    pass

        # 2. Extract specific NutAssembly physical metrics
        if hasattr(inner, 'objects_on_pegs'):
            # objects_on_pegs is an array (e.g., [1, 0] meaning one nut is on, one is not)
            on_pegs_array = getattr(inner, 'objects_on_pegs')
            assembled_count = float(sum(on_pegs_array))
            
            metrics['physics/nut_assembled_count'] = assembled_count
            
            # Calculate percentage based on mode
            # single_object_mode > 0 means the task only requires 1 nut
            required_nuts = 1.0 if getattr(inner, 'single_object_mode', 0) > 0 else float(len(on_pegs_array))
            
            metrics['physics/raw_assembly_percentage'] = min(1.0, assembled_count / required_nuts)

        # 3. Overall strict success (Did the environment declare the task completely solved?)
        if hasattr(inner, '_check_success'):
            is_success = inner._check_success()
            metrics['physics/env_check_success'] = 1.0 if is_success else 0.0

    except Exception as e:
        print(f"Metric extraction failed: {e}") 
        pass
        
    return metrics

def door_task_metrics_fn(env, info):
    """Metrics for the Robosuite Door environment.

    Reports:
      physics/door_success      — 1.0 if the door has been fully opened (env._check_success())
      physics/door_angle_deg    — current door hinge angle in degrees (0 = closed)
      physics/handle_grasped    — 1.0 if the robot is grasping the door handle
      physics/is_success        — mirror of door_success for EvalCallback compatibility
    """
    metrics = {}
    try:
        raw_env = getattr(env, 'unwrapped', env)
        inner = getattr(raw_env, 'unwrapped', getattr(raw_env, 'env', raw_env))

        # ── Task success ─────────────────────────────────────────────────────
        if hasattr(inner, '_check_success'):
            is_success = bool(inner._check_success())
            metrics['physics/door_success'] = 1.0 if is_success else 0.0
            metrics['physics/is_success'] = 1.0 if is_success else 0.0

        # Also surface is_success from GeometricWrapper if already set
        if 'is_success' in info:
            metrics['physics/is_success'] = float(info['is_success'])

        # ── Door hinge angle ─────────────────────────────────────────────────
        # Robosuite Door env exposes the door hinge joint via the sim
        try:
            sim = inner.sim
            # The door hinge joint is named 'door_hinge' in Robosuite
            hinge_id = sim.model.joint_name2id('door_hinge')
            hinge_angle = float(sim.data.qpos[sim.model.jnt_qposadr[hinge_id]])
            metrics['physics/door_angle_deg'] = float(np.degrees(hinge_angle))
        except Exception:
            pass

        # ── Handle grasp ─────────────────────────────────────────────────────
        try:
            # Robosuite Door exposes this as a sim contact check
            if hasattr(inner, 'door_handle_touch'):
                metrics['physics/handle_grasped'] = float(inner.door_handle_touch)
            elif hasattr(inner, '_check_grasp'):
                # Generic Robosuite grasp check against the handle geom
                grasped = inner._check_grasp(
                    gripper=inner.robots[0].gripper,
                    object_geoms=inner.door.door_handle
                )
                metrics['physics/handle_grasped'] = 1.0 if grasped else 0.0
        except Exception:
            pass

    except Exception as e:
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
    # Quality reward flags (arXiv:2502.12599 adaptation)
    parser.add_argument("--use_quality_reward", action="store_true",
                        help="Enable checkpoint-gated quality reward for scattered-marker wiping task")
    parser.add_argument("--use_sequential_waypoints", action="store_true", default=False,
                        help="Enforce sequential Y-sorted waypoint guidance. Default: False (nearest-mode). "
                             "Only enable if adding gripper_to_active_waypoint to obs.")
    parser.add_argument("--quality_f_target", type=float, default=15.0,
                        help="Target normal force (N) for the Gaussian force quality reward")
    parser.add_argument("--quality_sigma", type=float, default=15.0,
                        help="Std-dev (N) of the force quality Gaussian")
    parser.add_argument("--quality_r_checkpoint", type=float, default=0.08,
                        help="Checkpoint radius (m): EEF must be this close to earn quality reward")
    parser.add_argument("--quality_w_con", type=float, default=1.5,
                        help="Weight for checkpoint-gated contact reward")
    parser.add_argument("--quality_w_force", type=float, default=2.0,
                        help="Weight for force quality Gaussian reward")
    parser.add_argument("--quality_w_guide", type=float, default=1.5,
                        help="Weight for nearest-marker guidance reward")
    parser.add_argument("--quality_guide_scale", type=float, default=0.35,
                        help="Length scale (m) for r_guide — use ~35cm so gradient exists from hover height")
    parser.add_argument("--gamma", type=float, default=0.99, help="gamma parameter for SAC algorithm")
    parser.add_argument("--horizon", type=int, default=170, help="Horizon parameter for SAC algorithm")
    parser.add_argument("--lr", type=float, default=3e-4, help="Learning rate for SAC algorithm")


    parser.add_argument("--use_llm_prior", action="store_true")
    parser.add_argument("--add_prior_obs", action="store_true",
                        help="Append [prior_actions, confidence_w] to obs for ALL configs. "
                             "Off by default (clean obs). Enable only for LLM ablation comparisons.")
    parser.add_argument("--llm_backend", type=str, default="ollama", choices=["openai", "ollama"])
    parser.add_argument("--llm_query_interval", type=int, default=50)
    parser.add_argument("--llm_prior_weight", type=float, default=0.4)
    parser.add_argument("--llm_anneal_steps", type=int, default=0,
                   help="Linearly decay LLM prior weight to anneal_floor over this many per-env planner steps. "
                        "0 = no annealing (constant weight). Recommended: 0.75 * total_timesteps / n_envs.")
    parser.add_argument("--llm_anneal_floor", type=float, default=0.05,
                   help="Minimum prior weight after annealing (default 0.05). "
                        "Set to 0.0 to anneal completely to zero.")
    parser.add_argument("--llm_anneal_schedule", type=str, default="linear",
                   choices=["linear", "cosine"],
                   help="Annealing schedule for LLM prior weight: 'linear' (default) or "
                        "'cosine' (slow start, fast middle, slow end — prevents collapse).")
    parser.add_argument("--llm_model",          default="llama3.2")
    parser.add_argument("--llm_profile",        default="configs/nutassembly_robosuite_impedance_profile.yaml",
                   help="Path to YAML impedance profile for this task")
    
    # VLM-specific arguments
    parser.add_argument("--use_vlm", action="store_true", help="Use Vision Language Model instead of text-only LLM")
    parser.add_argument("--vlm_model", type=str, default="llava", 
                       choices=["llava", "llama3.2", "gpt-4o", "gpt-4o-mini"],
                       help="VLM model to use (for use_vlm=True)")
    parser.add_argument("--use_cameras", action="store_true", 
                       help="Enable camera observations in the environment (required for VLM)")
    parser.add_argument("--camera_names", type=str, default="frontview",
                       help="Comma-separated list of camera names to use")
    parser.add_argument("--vlm_image_size", type=int, default=224,
                       help="Image size for VLM (resized to this dimension)")



    parser.add_argument("--record_video", action="store_true",
                        help="Record one eval episode as wandb video at each eval checkpoint")
    parser.add_argument("--video_fps", type=int, default=20,
                        help="FPS for recorded WandB videos. Should match control_freq (default: 20 Hz)")
    parser.add_argument("--primitive_init", type=str, default="none",
                        choices=["none", "teleport", "scripted", "both"],
                        help="Initialize episodes with a motion primitive: 'teleport', 'scripted', 'both', or 'none'.")
    parser.add_argument("--use_domain_rand", action="store_true",
                        help="Enable domain randomization for TiltedWipe (tilt angle ±7°, table size 70-100%).")
    parser.add_argument("--early_terminate", action="store_true",
                        help="If set, enables early termination upon _check_success(). Default False (runs until horizon).")
    parser.add_argument("--use_ema", action="store_true",
                        help="Enable Universal Stiffness EMA Filter for smoothing.")
    parser.add_argument("--quat_debug", action="store_true",
                        help="Print quaternion diagnostic candidates for nut handle orientation.")
    parser.add_argument("--gamma_start", type=float, default=None,
                        help="If set, enables gamma curriculum: starts at gamma_start and linearly anneals "
                             "to --gamma over the first 50%% of training, then holds at --gamma. "
                             "E.g. --gamma_start 0.95 --gamma 0.9933. "
                             "Prevents the end-of-episode 'give-up' behaviour seen with low gamma "
                             "while avoiding Q-function overestimation spikes from hard gamma jumps.")
    parser.add_argument("--batch_size", type=int, default=512,
                        help="Batch size for SAC. Defaults to 512, but 1024 is recommended for stability "
                             "when using high gamma and dense rewards.")
    
    # Logging Args
    parser.add_argument("--run_name", type=str, required=True, help="Name of the run for logging/saving")
    
    parser.add_argument("--load_model", type=str, default=None, help="Path to best_model.zip to resume fine-tuning from")
    
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
        task_config["use_condensed_obj_obs"] = False
        if getattr(args, 'use_quality_reward', False):
            task_config["wipe_contact_reward"] = 0.0
            task_config["distance_multiplier"] = 0.0
            task_config["distance_th_multiplier"] = 0.0
            task_config["excess_force_penalty_mul"] = 0.0
        task_metrics = wipe_task_metrics_fn
    elif 'nutassembly' in env_lower:
        task_type = 'nutassembly'
        task_metrics = nutassembly_task_metrics_fn
    elif 'door' in env_lower:
        task_type = 'door'
        task_metrics = door_task_metrics_fn

    task_kwargs = {}
    if task_config is not None:
        task_kwargs['task_config'] = task_config

    enable_cameras = True
    # Honour --camera_names so passing e.g. frontview,robot0_eye_in_hand gives side-by-side video
    video_camera_names = (
        [c.strip() for c in args.camera_names.split(',')]
        if getattr(args, 'camera_names', None)
        else ['frontview']
    )
    env = suite.make(
        env_name=args.env,
        robots="Panda",
        controller_configs=controller_config,
        has_renderer=False,
        use_object_obs=True,
        has_offscreen_renderer=enable_cameras,
        use_camera_obs=enable_cameras,
        camera_names=video_camera_names,
        control_freq=20,
        reward_shaping=True,
        ignore_done=False,
        horizon=args.horizon if 'nutassembly' not in task_type else args.horizon + TELEPORT_STEPS,
        **task_kwargs
    )

    env = GymWrapper(env)

    try:
        if getattr(args, 'primitive_init', 'none') in ('teleport', 'both') and ('nutassembly' in env_lower or 'peg' in env_lower):
            env = RobosuiteTeleportWrapper(env, setup_steps=TELEPORT_STEPS, is_eval=is_eval)
    except Exception as e:
        print('Exception setting up primitive', e)
        pass

    # ── Wipe-specific wrappers ────────────────────────────────────────────
    if 'wipe' in env_lower:
        if getattr(args, 'use_domain_rand', False):
            env = WipeDomainRandomizationWrapper(
                env,
                tilt_min_deg=38.0,
                tilt_max_deg=52.0,
                size_scale_min=0.7,
                randomize_friction=True,
                is_eval=is_eval,
            )
        if getattr(args, 'primitive_init', 'none') in ('teleport', 'both'):
            env = WipeTeleportWrapper(
                env,
                tilt_angle_deg=45.0,
                hover_dist=0.15,
                is_eval=is_eval,
                randomize_pose=getattr(args, 'use_domain_rand', False),
            )
    
    # ✅ Auto-select profile: VLM overrides, then env-specific defaults
    llm_profile_path = args.llm_profile
    if args.use_llm_prior:
        if args.use_vlm:
            llm_profile_path = "configs/nutassembly_vlm_impedance_profile.yaml"
        elif 'door' in env_lower and args.llm_profile == "configs/nutassembly_robosuite_impedance_profile.yaml":
            # Auto-select Door profile when user hasn't explicitly overridden --llm_profile
            llm_profile_path = "configs/door_impedance_profile.yaml"
        elif 'wipe' in env_lower and args.llm_profile == "configs/nutassembly_robosuite_impedance_profile.yaml":
            # Auto-select Wipe profile when user hasn't explicitly overridden --llm_profile
            llm_profile_path = "configs/wipe_impedance_profile_HQ.yaml"
    
    
    if task_type == 'wipe':
        stiffness_penalty = 0.002
        success_bonus = 0.0      # Wipe uses continuous % reward, no binary ET
    elif task_type == 'nutassembly':
        stiffness_penalty = 0.01
        success_bonus = 0.0      # Removed per PI recommendation (was 5.0)
    elif task_type == 'door':
        stiffness_penalty = 0.001
        success_bonus = 0.0      # Removed per PI recommendation (was 5.0)
    else:
        stiffness_penalty = 0.0
        success_bonus = 0.0

    env = GeometricWrapper(
        env=env,
        stiffness_penalty=stiffness_penalty,
        success_bonus=success_bonus,
        early_terminate_on_success=getattr(args, 'early_terminate', False),
        use_spd_manifold=args.use_spd,
        use_lie_group=args.use_lie,
        use_diag_manifold=args.use_diag,
        use_fixed=args.use_fixed,
        is_eval=is_eval,
        use_llm_prior=args.use_llm_prior,
        use_ema=args.use_ema,
        llm_backend=args.llm_backend,
        llm_model=args.vlm_model if args.use_vlm else args.llm_model,
        llm_query_interval=args.llm_query_interval,
        llm_prior_weight=args.llm_prior_weight,
        llm_profile_path=llm_profile_path if args.use_llm_prior else None,
        llm_anneal_steps=getattr(args, 'llm_anneal_steps', 0),
        llm_anneal_floor=getattr(args, 'llm_anneal_floor', 0.05),
        llm_anneal_schedule=getattr(args, 'llm_anneal_schedule', 'linear'),
        task_type=task_type,
        task_metrics_fn=task_metrics,
        use_vision=args.use_vlm,
        add_prior_obs=getattr(args, 'add_prior_obs', False),
        use_quality_reward=getattr(args, 'use_quality_reward', False),
        use_sequential_waypoints=getattr(args, 'use_sequential_waypoints', True),
        quality_f_target=getattr(args, 'quality_f_target', 15.0),
        quality_sigma=getattr(args, 'quality_sigma', 15.0),
        quality_r_checkpoint=getattr(args, 'quality_r_checkpoint', 0.08),
        quality_w_con=getattr(args, 'quality_w_con', 1.5),
        quality_w_force=getattr(args, 'quality_w_force', 2.0),
        quality_w_guide=getattr(args, 'quality_w_guide', 1.5),
        quality_guide_scale=getattr(args, 'quality_guide_scale', 0.35),
    )

    
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
            task_config["use_condensed_obj_obs"] = False
            if getattr(args, 'use_quality_reward', False):
                task_config["wipe_contact_reward"] = 0.0
                task_config["distance_multiplier"] = 0.0
                task_config["distance_th_multiplier"] = 0.0
                task_config["excess_force_penalty_mul"] = 0.0
            task_metrics = wipe_task_metrics_fn
        elif 'nutassembly' in env_lower:
            task_type = 'nutassembly'
            task_metrics = nutassembly_task_metrics_fn
        elif 'door' in env_lower:
            task_type = 'door'
            task_metrics = door_task_metrics_fn

        task_kwargs = {}
        if task_config is not None:
            task_kwargs['task_config'] = task_config

        # ✅ Enable cameras if VLM is requested or explicitly enabled
        enable_cameras = args.use_vlm or getattr(args, 'use_cameras', False)
        camera_names_list = args.camera_names.split(',') if isinstance(args.camera_names, str) else ['wrist', 'frontview']
        camera_names_list = [c.strip() for c in camera_names_list]

        env = suite.make(
            env_name=args.env,
            robots="Panda",
            controller_configs=controller_config,
            has_renderer=False,
            use_object_obs=True,
            has_offscreen_renderer=enable_cameras,  # Enable rendering if cameras needed
            use_camera_obs=enable_cameras,           # Enable camera observations if requested
            camera_names=camera_names_list if enable_cameras else None,
            reward_shaping=True,
            control_freq=20,
            ignore_done=False,
            horizon=args.horizon if 'nutassembly' not in task_type else args.horizon + TELEPORT_STEPS,
            **task_kwargs
        )
        
        env = GymWrapper(env)

        # for the NutAssembly envs
        try:
            # if getattr(args, 'primitive_init', 'none') in ('scripted', 'both') and ('nutassembly' in env_lower or 'peg' in env_lower):
            #     env = RobosuiteScriptedPrimitiveWrapper(env, setup_steps=90, is_eval=is_eval)
            if getattr(args, 'primitive_init', 'none') in ('teleport', 'both') and ('nutassembly' in env_lower or 'peg' in env_lower):
                env = RobosuiteTeleportWrapper(env, setup_steps=TELEPORT_STEPS, is_eval=is_eval)
                # env = FixedGripperWrapper(env)
        except Exception as e:
            print(f"Error occurred while initializing primitive wrapper: {e}")
            pass

        # ── Wipe-specific wrappers ────────────────────────────────────────
        if 'wipe' in env_lower:
            if getattr(args, 'use_domain_rand', False):
                env = WipeDomainRandomizationWrapper(
                    env,
                    tilt_min_deg=38.0,
                    tilt_max_deg=52.0,
                    size_scale_min=0.7,
                    randomize_friction=True,
                    is_eval=is_eval,
                )
            if getattr(args, 'primitive_init', 'none') in ('teleport', 'both'):
                env = WipeTeleportWrapper(
                    env,
                    tilt_angle_deg=45.0,
                    hover_dist=0.15,
                    is_eval=is_eval,
                    randomize_pose=getattr(args, 'use_domain_rand', False),
                )
        
        # ✅ Auto-select profile: VLM overrides, then env-specific defaults
        llm_profile_path = args.llm_profile
        if args.use_llm_prior:
            if args.use_vlm:
                llm_profile_path = "configs/nutassembly_vlm_impedance_profile.yaml"
            elif 'door' in env_lower and args.llm_profile == "configs/nutassembly_robosuite_impedance_profile.yaml":
                llm_profile_path = "configs/door_impedance_profile.yaml"
            elif 'wipe' in env_lower and args.llm_profile == "configs/nutassembly_robosuite_impedance_profile.yaml":
                # Auto-select Wipe profile (was missing here — caused NAS profile to be used in eval!)
                llm_profile_path = "configs/wipe_impedance_profile_HQ.yaml"

        if task_type == 'wipe':
            stiffness_penalty = 0.002
            success_bonus = 0.0      # Wipe uses continuous % reward, no binary ET
        elif task_type == 'nutassembly':
            stiffness_penalty = 0.01
            success_bonus = 0.0      # Removed per PI recommendation (was 5.0)
        elif task_type == 'door':
            stiffness_penalty = 0.001
            success_bonus = 0.0      # Removed per PI recommendation (was 5.0)
        else:
            stiffness_penalty = 0.0
            success_bonus = 0.0
        print(f'Using stiffness penalty of {stiffness_penalty}, success bonus of {success_bonus}')

        env = GeometricWrapper(
            env=env,
            stiffness_penalty=stiffness_penalty,
            success_bonus=success_bonus,
            early_terminate_on_success=getattr(args, 'early_terminate', False),
            use_spd_manifold=args.use_spd,
            use_lie_group=args.use_lie,
            use_diag_manifold=args.use_diag,
            use_fixed=args.use_fixed,
            is_eval=is_eval,
            use_llm_prior=args.use_llm_prior,
            use_ema=args.use_ema,
            llm_backend=args.llm_backend,
            llm_model=args.vlm_model if args.use_vlm else args.llm_model,
            llm_query_interval=args.llm_query_interval,
            llm_prior_weight=args.llm_prior_weight,
            llm_profile_path=llm_profile_path if args.use_llm_prior else None,
            llm_anneal_steps=getattr(args, 'llm_anneal_steps', 0),
            llm_anneal_floor=getattr(args, 'llm_anneal_floor', 0.05),
            llm_anneal_schedule=getattr(args, 'llm_anneal_schedule', 'linear'),
            task_type=task_type,
            task_metrics_fn=task_metrics,
            use_vision=args.use_vlm,
            add_prior_obs=getattr(args, 'add_prior_obs', False),
            use_quality_reward=getattr(args, 'use_quality_reward', False),
            use_sequential_waypoints=getattr(args, 'use_sequential_waypoints', True),
            quality_f_target=getattr(args, 'quality_f_target', 15.0),
            quality_sigma=getattr(args, 'quality_sigma', 15.0),
            quality_r_checkpoint=getattr(args, 'quality_r_checkpoint', 0.08),
            quality_w_con=getattr(args, 'quality_w_con', 1.5),
            quality_w_force=getattr(args, 'quality_w_force', 2.0),
            quality_w_guide=getattr(args, 'quality_w_guide', 1.5),
            quality_guide_scale=getattr(args, 'quality_guide_scale', 0.35),
        )

        # ── Seeding ──────────────────────────────────────────────────────────
        # NOTE: Do NOT call env.reset(seed=...) here.
        # SubprocVecEnv already calls reset() internally during __init__, so
        # an explicit reset() here would trigger a second full episode setup
        # (e.g. the 90-step teleport sequence for NutAssembly), doubling frames
        # in rollout videos and wasting wall time.
        #
        # Instead, we store the seed on the GeometricWrapper so it is applied
        # the next time reset() is naturally called by SB3 (which is the one
        # SubprocVecEnv already triggers during __init__ or between episodes).
        # GeometricWrapper.reset() already calls np.random.seed(seed) when
        # self._pending_seed is set (see geometric.py).
        if hasattr(env, '_pending_seed'):
            env._pending_seed = seed + rank  # GeometricWrapper picks this up on next reset()
        env.action_space.seed(seed + rank)
        env.observation_space.seed(seed + rank)
        np.random.seed(seed + rank)
        random.seed(seed + rank)

        return env
    
    return _init

class SyncEvalCallback(EvalCallback):
    def _on_step(self) -> bool:
        try:
            # Get the first (and only) eval environment from the VecMonitor -> DummyVecEnv
            inner_env = self.eval_env.venv.envs[0]
            
            # Unwrap until we find the GeometricWrapper with the llm_planner
            while hasattr(inner_env, "env") and not hasattr(inner_env, "llm_planner"):
                inner_env = inner_env.env
            
            if hasattr(inner_env, "llm_planner") and inner_env.llm_planner is not None:
                # Sync eval LLM planner's step counter with the training progress
                train_n_envs = self.model.get_env().num_envs
                local_steps = self.num_timesteps // train_n_envs
                inner_env.llm_planner._global_step = local_steps
        except Exception as e:
            print(f"Warning: Failed to sync eval LLM step: {e}")
            
        # 1. Let the parent class run the evaluation and (optionally) save the reward-based best model
        result = super()._on_step()

        # 2. Add our own logic to track and save the best success-rate model
        if self.eval_freq > 0 and self.n_calls % self.eval_freq == 0:
            if len(self.evaluations_successes) > 0 and len(self.evaluations_successes[-1]) > 0:
                current_success_rate = np.mean(self.evaluations_successes[-1])
                
                if not hasattr(self, 'best_success_rate'):
                    self.best_success_rate = -np.inf
                
                if current_success_rate > self.best_success_rate:
                    self.best_success_rate = current_success_rate
                    if self.verbose > 0:
                        print(f"New best success rate: {current_success_rate:.2f}! Saving to best_success_model.zip")
                    
                    import os
                    save_path = os.path.join(self.best_model_save_path, "best_success_model")
                    self.model.save(save_path)

        return result

def setup_evaluation_callback(args, run_name):
    # Evaluation 
    eval_env_fn = make_env(args, is_eval=True, rank=0, seed=42)
    eval_env = DummyVecEnv([eval_env_fn])
    eval_env = VecMonitor(eval_env)
    
    # Evaluate every ~10% of total training budget, but at least every 50k env steps
    eval_freq_steps = max(args.total_timesteps // (args.n_envs * 10), 50_000 // args.n_envs)
    # eval_freq_steps = 1000
    
    eval_callback = SyncEvalCallback(
        eval_env,
        best_model_save_path=f"./logs/best_models/{run_name}/",
        log_path=f"./logs/eval/{run_name}/",
        eval_freq=eval_freq_steps,
        n_eval_episodes=10,  # 10 deterministic episodes per eval
        deterministic=True,
        render=False
    )

    return eval_callback

def main():
    args = parse_args()
    
    # ✅ Global Auto-select profile: VLM overrides, then env-specific defaults
    # This ensures the callback, WandB, and envs all use the exact same profile.
    if args.use_llm_prior:
        env_lower = args.env.lower()
        if args.use_vlm:
            args.llm_profile = "configs/nutassembly_vlm_impedance_profile.yaml"
        elif 'door' in env_lower and args.llm_profile == "configs/nutassembly_robosuite_impedance_profile.yaml":
            args.llm_profile = "configs/door_impedance_profile.yaml"
        elif 'wipe' in env_lower and args.llm_profile == "configs/nutassembly_robosuite_impedance_profile.yaml":
            args.llm_profile = "configs/wipe_impedance_profile_HQ.yaml"

    set_random_seed(args.seed)

    device = "cuda" if torch.cuda.is_available() else "cpu"
    print(f"Training on {device.upper()} with {args.n_envs} parallel environments.")
    print(f"🚀 Starting training for {args.env} with SEED: {args.seed} and gamma: {args.gamma}")

    # wandb_run_name = f'{args.algorithm}_{args.env.upper()}_{args.run_name}' 
    wandb_run_name = f'{args.algorithm}_{args.run_name}' 
    run_name = f'{wandb_run_name}_SEED_{args.seed}'
    
    run = wandb.init(
        project="HiRes-VIC",
        name=wandb_run_name,
        config=vars(args),
        sync_tensorboard=True,
        monitor_gym=True,
        settings=wandb.Settings(init_timeout=300)
    )

    env_fns = [make_env(args, is_eval=False, rank=i) for i in range(args.n_envs)]
    env = SubprocVecEnv(env_fns)
    env = VecMonitor(env)

    if args.load_model:
        print(f"Loading existing model from {args.load_model}")
        
        # Use command-line learning rate if specified, otherwise default to a safe 2e-5
        custom_objects = {
            "learning_rate": args.lr if getattr(args, "lr", None) is not None else 2e-5,
        }
        
        model = SAC.load(
            args.load_model, 
            env=env,
            tensorboard_log=f"./outputs/logs/{run_name}",
            device=device,
            buffer_size=1_000_000,
            custom_objects=custom_objects
        )
        
        # --- CRITICAL FIX: Buffer Warmup ---
        # SAC will start with an empty 1M buffer. If we train immediately, it will overfit
        # to the first 100 transitions in the batch, causing gradients to explode and ruining
        # the weights. We must run the current policy for a while with NO updates to fill the buffer.
        print("Warming up fresh replay buffer with 50,000 steps of current policy...")
        original_grad_steps = model.gradient_steps
        model.gradient_steps = 0  # Disable learning
        model.learn(total_timesteps=50_000, reset_num_timesteps=False)
        print(f"Buffer warmed up. Buffer size: {model.replay_buffer.pos}")
        
        # Re-enable learning
        model.gradient_steps = original_grad_steps
    else:
        model = SAC(
            "MlpPolicy",
            env,
            verbose=1,
            tensorboard_log=f"./outputs/logs/{run_name}",
            learning_rate=args.lr,
            batch_size=args.batch_size,
            buffer_size=1_000_000,
            tau=0.002,                  # For soft updates of the target network
            target_entropy="auto",      # Encourage exploration (tune based on action space)
            gamma=args.gamma,           # default 0.99
            train_freq=1,               # Train every step
            gradient_steps=args.n_envs, # 1 update per new sample (matches n_envs parallel envs)
            use_sde=False,              # Smooth robotic noise
            # sde_sample_freq=8,
            seed=args.seed,
            device=device
        )

    eval_callback = setup_evaluation_callback(args, run_name)
    
    modes = None
    if args.use_llm_prior:
        try:
            import yaml
            with open(args.llm_profile, 'r') as f:
                profile = yaml.safe_load(f)
                modes = list(profile.get("phases", {}).keys())
                print(f"Loaded LLM impedance profile for logging from {args.llm_profile} with modes: {modes}")
        except Exception as e:
            print(f"Failed to load LLM profile for logging callback: {e}")
    logging_callback = RobosuiteLoggingCallback(modes=modes)

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

    from hires_vic.utils.callbacks import GammaCurriculumCallback
    gamma_callback = None
    
    # ── Dynamic Gamma Curriculum ──────────────────────────────────────────────
    if getattr(args, 'gamma_start', None) is not None and not args.load_model:
        # Standard curriculum for from-scratch training
        anneal_end = args.total_timesteps // 2   # reach gamma_end by 50% of training
        gamma_callback = GammaCurriculumCallback(
            gamma_start=args.gamma_start,
            gamma_end=args.gamma,
            anneal_start_steps=0,
            anneal_end_steps=anneal_end,
            verbose=1,
        )
        print(f"Γ Gamma curriculum: {args.gamma_start} → {args.gamma} over {anneal_end:,} steps")
    elif args.load_model and model.gamma < args.gamma:
        # Fine-tuning curriculum: The loaded checkpoint might have been saved early
        # during the original curriculum (e.g. gamma=0.96). Instantly assigning 0.9933
        # would shock the Q-values. Instead, we smoothly anneal from its saved gamma!
        loaded_gamma = model.gamma
        anneal_end = args.total_timesteps // 2
        gamma_callback = GammaCurriculumCallback(
            gamma_start=loaded_gamma,
            gamma_end=args.gamma,
            anneal_start_steps=0,
            anneal_end_steps=anneal_end,
            verbose=1,
        )
        print(f"Γ Resuming Gamma curriculum: loaded {loaded_gamma:.4f} → {args.gamma:.4f} over {anneal_end:,} steps")

    # 6. Train!
    print(f"Starting training for {args.total_timesteps} steps...")
    callbacks = [logging_callback, eval_callback, wandb_callback]
    if video_callback is not None:
        callbacks.append(video_callback)
    if gamma_callback is not None:
        callbacks.append(gamma_callback)

    if args.load_model:
        # Subtract the warmup steps from the total
        remaining_timesteps = max(0, args.total_timesteps - 50_000)
        print(f"Starting fine-tuning for remaining {remaining_timesteps} steps...")
        model.learn(
            total_timesteps=remaining_timesteps,
            callback=callbacks,
            reset_num_timesteps=False
        )
    else:
        model.learn(
            total_timesteps=args.total_timesteps,
            callback=callbacks,
            reset_num_timesteps=True
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