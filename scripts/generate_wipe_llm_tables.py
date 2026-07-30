import os
import sys
import re
import glob
import json
import torch
import numpy as np
import pandas as pd
import wandb

# Disable wandb to prevent errors in wrappers during evaluation
wandb.init(mode="disabled")

sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))
import robosuite as suite
import robosuite.controllers.parts.controller_factory as factory
from robosuite import load_composite_controller_config
from robosuite.wrappers import GymWrapper
from stable_baselines3 import SAC
import zipfile
import io
import torch
import gymnasium as gym

from hires_vic import envs
from hires_vic.envs.riemannian_controller import RiemannianController
from hires_vic.wrappers import GeometricWrapper, RobosuiteTeleportWrapper

factory.arm_controllers.OperationalSpaceController = RiemannianController

WIPE_TASK_CONFIG = {
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
    "num_markers": 50,
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

def get_expected_obs_dim(model_path):
    try:
        with zipfile.ZipFile(model_path, "r") as archive:
            if "data" in archive.namelist():
                with archive.open("data") as f:
                    data = f.read()
                    try:
                        import json
                        data_dict = json.loads(data.decode("utf-8"))
                        return data_dict.get("observation_space").get("_shape")[0]
                    except Exception:
                        import io
                        data_dict = torch.load(io.BytesIO(data), map_location="cpu", weights_only=False)
                        return data_dict.get("observation_space").shape[0]
    except Exception as e:
        print(f"Error reading expected obs dim from {model_path}: {e}")
        return None

def get_expected_action_dim(model_path):
    try:
        with zipfile.ZipFile(model_path, "r") as archive:
            if "data" in archive.namelist():
                with archive.open("data") as f:
                    data = f.read()
                    try:
                        import json
                        data_dict = json.loads(data.decode("utf-8"))
                        return data_dict.get("action_space").get("_shape")[0]
                    except Exception:
                        import io
                        data_dict = torch.load(io.BytesIO(data), map_location="cpu", weights_only=False)
                        return data_dict.get("action_space").shape[0]
    except Exception as e:
        print(f"Error reading expected action dim from {model_path}: {e}")
        return None

class ObservationAlignWrapper(gym.ObservationWrapper):
    def __init__(self, env, expected_dim):
        super().__init__(env)
        self.expected_dim = expected_dim
        self.observation_space = gym.spaces.Box(
            low=-np.inf, high=np.inf, shape=(expected_dim,), dtype=np.float32
        )
        # Calculate how many extra_obs_dims there are in the current env
        # GeometricWrapper appends prior_dim + 1
        self.extra_obs_dim = env.unwrapped.extra_obs_dim if hasattr(env.unwrapped, "extra_obs_dim") else 0
        if hasattr(env, "extra_obs_dim"):
             self.extra_obs_dim = env.extra_obs_dim

    def observation(self, observation):
        if observation.shape[0] == self.expected_dim:
            return observation
        
        # If it doesn't match, we assume the mismatch is in the flat_obs part,
        # and that the extra_obs_dim was originally exactly `prior_dim` instead of `prior_dim + 1` 
        # (or some other small difference in the raw robosuite state).
        
        # We know current env has `self.extra_obs_dim` appended at the end.
        current_flat_len = observation.shape[0] - self.extra_obs_dim
        
        # We need to extract the prior_dim part (ignoring current_w which is at the very end)
        prior_dim = self.extra_obs_dim - 1
        prior_state = observation[current_flat_len : current_flat_len + prior_dim]
        
        # The expected flat_obs length is expected_dim - prior_dim
        expected_flat_len = self.expected_dim - prior_dim
        
        # Truncate or pad the base flat_obs to match expected_flat_len
        base_obs = observation[:current_flat_len]
        if len(base_obs) > expected_flat_len:
            base_obs = base_obs[:expected_flat_len]
        else:
            base_obs = np.pad(base_obs, (0, expected_flat_len - len(base_obs)))
            
        aligned_obs = np.concatenate([base_obs, prior_state]).astype(np.float32)
        return aligned_obs

class ActionAlignWrapper(gym.ActionWrapper):
    def __init__(self, env, expected_action_dim):
        super().__init__(env)
        self.expected_action_dim = expected_action_dim
        low = np.full((expected_action_dim,), -1.0, dtype=np.float32)
        high = np.full((expected_action_dim,), 1.0, dtype=np.float32)
        self.action_space = gym.spaces.Box(low=low, high=high, dtype=np.float32)
        
    def action(self, action):
        env_action_dim = self.env.action_space.shape[0]
        if len(action) == env_action_dim:
            return action
        elif len(action) > env_action_dim:
            return action[:env_action_dim]
        else:
            return np.pad(action, (0, env_action_dim - len(action)))

def make_env(env_name, config_name, model_path=None):
    controller_config = load_composite_controller_config(controller="BASIC", robot="panda")
    for part in ["left", "torso", "head", "base", "legs"]:
        controller_config["body_parts"].pop(part, None)
    
    arm_config = controller_config["body_parts"]["right"]
    arm_config["type"] = "OSC_POSE"

    use_spd = False
    use_lie = False
    use_diag = False
    use_fixed = False
    use_llm = False
    profile_path = None

    if "LLM" in config_name:
        use_llm = True
        if "NutAssemblySquare" in env_name:
            profile_path = "configs/nutassembly_robosuite_impedance_profile.yaml"
        elif "Door" in env_name:
            profile_path = "configs/door_impedance_profile.yaml"
        elif "TiltedWipe" in env_name:
            profile_path = "configs/wipe_impedance_profile.yaml"

    if "BASELINE" in config_name or "VARIABLE_KP" in config_name:
        arm_config["impedance_mode"] = "variable_kp"
        use_fixed = False
    elif "SPD_ONLY" in config_name:
        arm_config["impedance_mode"] = "riemannian_kp"
        use_spd = True
    elif "LIE_ONLY" in config_name:
        arm_config["impedance_mode"] = "variable_kp"
        use_lie = True
    elif "DIAG" in config_name:
        arm_config["impedance_mode"] = "variable_kp"
        use_diag = True
    elif "FULL_GRL" in config_name:
        arm_config["impedance_mode"] = "riemannian_kp"
        use_spd = True
        use_lie = True
    else:
        arm_config["impedance_mode"] = "variable_kp"

    arm_config["kp_limits"] = [1, 300]
    arm_config["damping_ratio_limits"] = [1.0, 1.0]

    kwargs = {
        "env_name": env_name,
        "robots": "Panda",
        "controller_configs": controller_config,
        "has_renderer": False,
        "use_object_obs": True,
        "has_offscreen_renderer": True,
        "use_camera_obs": False,
        "reward_shaping": True,
        "horizon": 230 if env_name == "NutAssemblySquare" else (50 if env_name == "Door" else 150),
    }

    if "TILTED" in env_name.upper():
        kwargs["task_config"] = WIPE_TASK_CONFIG

    env = suite.make(**kwargs)
    env = GymWrapper(env)
    
    if "NutAssemblySquare" in env_name:
        env = RobosuiteTeleportWrapper(env, setup_steps=150, is_eval=True)
        # Monkey-patch _capture_frame so it correctly fetches frames without use_camera_obs=True
        def custom_capture_frame(self_instance):
            try:
                scene = self_instance.env.unwrapped.sim.render(camera_name="frontview", width=512, height=512, depth=False)
                return np.flipud(scene)
            except Exception:
                return None
        env._capture_frame = custom_capture_frame.__get__(env, type(env))
        
    task_type_str = env_name.lower()
    if 'nutassembly' in task_type_str:
        task_type_str = 'nutassembly'
    elif 'wipe' in task_type_str:
        task_type_str = 'wipe'
    elif 'door' in task_type_str:
        task_type_str = 'door'

    env = GeometricWrapper(
        env=env,
        use_spd_manifold=use_spd,
        use_lie_group=use_lie,
        use_diag_manifold=use_diag,
        use_fixed=use_fixed,
        is_eval=True,
        task_type=task_type_str,
        use_llm_prior=use_llm,
        llm_prior_weight=0.05 if "NutAssembly" in env_name else 0.4,
        llm_anneal_floor=0.05 if "NutAssembly" in env_name else 0.4,
        llm_model="llama3.2",
        llm_profile_path=profile_path
    )
    
    if model_path:
        expected_dim = get_expected_obs_dim(model_path)
        if expected_dim and expected_dim != env.observation_space.shape[0]:
            print(f"Aligning observation space from {env.observation_space.shape[0]} to {expected_dim}")
            env = ObservationAlignWrapper(env, expected_dim)

        expected_action_dim = get_expected_action_dim(model_path)
        if expected_action_dim and expected_action_dim != env.action_space.shape[0]:
            print(f"Aligning action space from {env.action_space.shape[0]} to {expected_action_dim}")
            env = ActionAlignWrapper(env, expected_action_dim)
            
    return env

def compute_auc_from_log(log_path):
    # Parse success_rate from standard sb3 log
    if not os.path.exists(log_path):
        return 0.0
    sr_vals = []
    with open(log_path, 'r') as f:
        for line in f:
            if "success_rate" in line and "|" in line:
                try:
                    val = float(line.split("|")[2].strip())
                    sr_vals.append(val)
                except:
                    pass
    if not sr_vals: return 0.0
    # Normalize AUC to 1.0 max (AUC of 1.0 over N steps is just the mean)
    return np.mean(sr_vals)

def evaluate_model(model_path, env_name, config_name, log_path, num_episodes=100, video_writer=None):
    env = make_env(env_name, config_name, model_path=model_path)
    model = SAC.load(model_path, env=env, device="cuda" if torch.cuda.is_available() else "cpu")

    
    results = {
        "success": [],
        "reward": [],
        "peak_force": [],
        "force_exceedance_rate": [],
        "joint_violations": [],
        "avg_cond_num": [],
        "avg_eucl_jerk": [],
        "avg_riem_jerk": [],
        "wipe_percentage": [],
        "auc_sr": compute_auc_from_log(log_path)
    }

    for ep in range(num_episodes):
        obs, info = env.reset()
        done = False
        truncated = False
        ep_reward = 0.0
        
        record_this_ep = (ep % 20 == 0) and (video_writer is not None)
        
        if record_this_ep:
            import cv2
            ptr = env
            while hasattr(ptr, 'env'):
                if "Teleport" in type(ptr).__name__:
                    if hasattr(ptr, 'frames'):
                        for i, f in enumerate(ptr.frames):
                            if f is not None and isinstance(f, np.ndarray):
                                f_copy = np.ascontiguousarray(f.copy())
                                text = f"Config: {config_name} | Ep: {ep}/100 | Setup {i}/{len(ptr.frames)}"
                                cv2.putText(f_copy, text, (10, 30), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (255, 255, 255), 1)
                                try: video_writer.append_data(f_copy)
                                except Exception: pass
                    break
                ptr = ptr.env
                
        step_idx = 0
                
        while not (done or truncated):
            action, _ = model.predict(obs, deterministic=True)
            obs, reward, terminated, truncated, info = env.step(action)
            ep_reward += reward
            done = bool(terminated)
            step_idx += 1
            
            if record_this_ep:
                try:
                    frame = env.unwrapped.sim.render(camera_name="frontview", width=512, height=512, depth=False)
                    frame = np.ascontiguousarray(np.flipud(frame))
                    import cv2
                    success = info.get("is_success", False)
                    text = f"Config: {config_name} | Ep: {ep}/100 | Step: {step_idx} | R: {ep_reward:.1f} | Success: {success}"
                    cv2.putText(frame, text, (10, 30), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (255, 255, 255), 1)
                    video_writer.append_data(frame)
                except Exception as e:
                    print(f"Video render error: {e}")

        results["success"].append(1.0 if info.get("is_success", False) else 0.0)
        results["reward"].append(ep_reward)
        results["peak_force"].append(info.get("safety/peak_force", 0.0))
        results["force_exceedance_rate"].append(info.get("safety/force_exceedance_rate", 0.0))
        results["joint_violations"].append(info.get("physics/joint_violation_count", 0.0))
        results["avg_cond_num"].append(info.get("smoothness/avg_cond_num", 0.0))
        results["avg_eucl_jerk"].append(info.get("smoothness/avg_euclidean_jerk", 0.0))
        results["avg_riem_jerk"].append(info.get("smoothness/avg_riemannian_jerk", 0.0))
        
        if env_name == "TiltedWipe":
            unwrapped_env = env.unwrapped
            if hasattr(unwrapped_env, "wiped_markers"):
                num_wiped = len(unwrapped_env.wiped_markers)
                total_markers = unwrapped_env.num_markers
                results["wipe_percentage"].append(num_wiped / total_markers)

    env.close()
    
    return {k: np.mean(v) if isinstance(v, list) else v for k, v in results.items()}

def parse_run_name(run_name):
    parts = run_name.split("_")
    
    if "SAC" not in parts: return None
    env_str = parts[1]
    seed_match = re.search(r"SEED_(\d+)", run_name)
    gamma_match = re.search(r"G(\d+\.\d+)", run_name) or re.search(r"GAMMA(\d+\.\d+)", run_name, re.IGNORECASE)
    
    if "BASELINE" in run_name: config = "BASELINE"
    elif "FULL_GRL" in run_name or "FULLGRL" in run_name: config = "FULL_GRL"
    elif "SPD_ONLY" in run_name: config = "SPD_ONLY"
    elif "LIE_ONLY" in run_name: config = "LIE_ONLY"
    elif "DIAG" in run_name: config = "DIAG"
    else: return None
    
    is_llm = "LLM" in run_name
    if is_llm: config += "_LLM"
        
    if env_str == "DOOR":
        env_real = "Door"
        if "_FINAL_LR3e-4_" not in run_name: return None
        if "_H50_" not in run_name: return None
        if "_G0.95_" not in run_name: return None
        if is_llm and "_LLM_W0.8_" not in run_name: return None
        
    elif env_str == "TILTEDWIPE":
        env_real = "TiltedWipe"
        if is_llm:
            if "_FINAL_LR3e-4_" not in run_name: return None
            if "_H150_" not in run_name: return None
            if "_LLM_W0.8_" not in run_name: return None
            if "_G0.95_" not in run_name: return None
        else:
            if "_FINAL_H150_" not in run_name: return None
            if "_G0.95_" not in run_name: return None
            
    elif env_str == "NUTASSEMBLYSQUARE" or "NUT_SQ" in run_name:
        env_real = "NutAssemblySquare"
        if config == "FULL_GRL" and not is_llm:
            if "FULLGRL_" not in run_name: return None
        if "_H80_" not in run_name: return None
        if "_G0.90_" not in run_name: return None
        if is_llm and "_LLM_W0.8_" not in run_name: return None
    else:
        return None
        
    s_match = re.search(r"_SEED_(\d+)", run_name)
    if not s_match: return None
    seed = int(s_match.group(1))
    
    g_match = re.search(r"_G(\d+\.\d+)_", run_name)
    gamma = float(g_match.group(1)) if g_match else 0.95
    
    return env_real, config, gamma, seed

def main():
    import argparse
    from datetime import datetime
    parser = argparse.ArgumentParser()
    parser.add_argument("--test", action="store_true", help="Run 1 episode of 1 model for fast testing")
    args = parser.parse_args()

    # Timestamped output directory so re-runs never overwrite previous videos/CSVs
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    output_dir = f"/home/cjimenez/projects/HiRes-VIC/outputs/eval_{timestamp}"
    os.makedirs(output_dir, exist_ok=True)
    print(f"Output directory: {output_dir}")

    log_dir = "/home/cjimenez/projects/HiRes-VIC/logs/best_models"
    base_log_dir = "/home/cjimenez/projects/HiRes-VIC/logs"
    
    models_found = []
    for entry in os.listdir(log_dir):
        if os.path.isdir(os.path.join(log_dir, entry)):
            parsed = parse_run_name(entry)
            if parsed:
                env_real, config, gamma, seed = parsed
                model_path = os.path.join(log_dir, entry, "best_model.zip")
                if os.path.exists(model_path):
                    models_found.append({
                        "run_name": entry,
                        "env": env_real,
                        "config": config,
                        "gamma": gamma,
                        "seed": seed,
                        "path": model_path
                    })
    
    df = pd.DataFrame(models_found)
    final_results = []
    
    config_list = [
        "BASELINE", "DIAG", "LIE_ONLY", "SPD_ONLY", "FULL_GRL",
        "BASELINE_LLM", "SPD_ONLY_LLM", "FULL_GRL_LLM"
    ]
    
    for env in ["NutAssemblySquare", "Door", "TiltedWipe"]:
        df_env = df[df["env"] == env]
        if df_env.empty: continue
        
        for config in config_list:
            if env in ["Door", "NutAssemblySquare"] and "_LLM" not in config:
                continue
                
            df_conf = df_env[df_env["config"] == config]
            if df_conf.empty: continue
            
            import imageio
            video_path = os.path.join(output_dir, f"{env}_{config}_recap.mp4")
            try:
                video_writer = imageio.get_writer(video_path, fps=20)
            except Exception as e:
                print(f"Could not create video writer for {env} {config}: {e}")
                video_writer = None

            
            latest_gamma = df_conf["gamma"].max()
            df_target = df_conf[df_conf["gamma"] == latest_gamma]
            
            # If testing, only evaluate the very first target model we find
            if args.test:
                df_target = df_target.head(1)
            
            print(f"Evaluating {env} | {config} | Gamma: {latest_gamma} | Seeds: {df_target['seed'].tolist()}")
            
            config_metrics = []
            for _, row in df_target.iterrows():
                base_name = row['run_name'].replace('SAC_', '')
                log_names = [f"{base_name}.log", f"{env}_{base_name}.log", f"{row['run_name']}.log"]
                
                log_path = ""
                for n in log_names:
                    p = os.path.join(base_log_dir, env.lower(), n)
                    if os.path.exists(p): log_path = p; break
                if not log_path:
                    for n in log_names:
                        p = os.path.join(base_log_dir, n)
                        if os.path.exists(p): log_path = p; break

                print(f"  -> Seed {row['seed']}...")
                try:
                    num_ep = 1 if args.test else 100
                    res = evaluate_model(row["path"], env, config, log_path, num_episodes=num_ep, video_writer=video_writer)
                    config_metrics.append(res)
                except Exception as e:
                    import traceback
                    print(f"     [Error] Failed to evaluate {row['run_name']}: {e}")
                    traceback.print_exc()
                    
            if config_metrics:
                agg = { "env": env, "config": config, "gamma": latest_gamma, "seeds_count": len(config_metrics) }
                for k in config_metrics[0].keys():
                    vals = [m[k] for m in config_metrics]
                    agg[f"{k}_mean"] = np.mean(vals)
                    agg[f"{k}_std"] = np.std(vals)
                final_results.append(agg)
                
            if args.test and final_results:
                break
                
        if video_writer is not None:
            video_writer.close()
            
        if args.test and final_results:
            break

                
    # Generate LaTeX
    if not final_results:
        print("No models evaluated successfully.")
        return

    new_res_df = pd.DataFrame(final_results)
    
    # Load existing metrics to preserve previously evaluated runs (like Door/NAS base configs)
    stable_csv = "/home/cjimenez/projects/HiRes-VIC/outputs/chapter6_metrics.csv"
    if os.path.exists(stable_csv):
        existing_df = pd.read_csv(stable_csv)
        # Update existing records with new ones, append new rows
        # We match by 'env' and 'config'
        merged_df = pd.concat([existing_df, new_res_df]).drop_duplicates(subset=['env', 'config'], keep='last').reset_index(drop=True)
        res_df = merged_df
    else:
        res_df = new_res_df

    res_df.to_csv(os.path.join(output_dir, "chapter6_metrics.csv"), index=False)
    # Also write a stable fixed-path copy for quick reference
    res_df.to_csv(stable_csv, index=False)
    
    print("\n" + "="*50)
    print("LaTeX Tables (Copy & Paste to ch6_experiments.tex)")
    print("="*50 + "\n")
    
    def get_best_val(group_df, metric_col, higher_is_better):
        if group_df.empty or metric_col not in group_df.columns: return None
        vals = group_df[metric_col].dropna()
        if vals.empty: return None
        return vals.max() if higher_is_better else vals.min()

    def fmt_cell(val, std, best_val, fmt=":.1f", mult=1.0):
        if pd.isna(val): return "-"
        is_best = False
        if best_val is not None and abs(val - best_val) < 1e-6:
            is_best = True
            
        mean_str = ("{" + fmt + "}").format(val * mult)
        std_str = ("{" + fmt + "}").format(std * mult)
        
        if is_best:
            return f"\\textbf{{{mean_str}}} $\\pm$ {std_str}"
        return f"{mean_str} $\\pm$ {std_str}"

    # -------------------------------------------------------------------------
    # 1. Performance Table (Horizontal Combined)
    # -------------------------------------------------------------------------
    print("% LaTeX Table for Performance (SR & Reward) across all environments")
    print("\\begin{table*}[h]")
    print("\\centering")
    print("\\resizebox{\\textwidth}{!}{")
    print("\\setlength{\\tabcolsep}{5pt}")
    print("\\begin{tabular}{@{}l cc cc cc@{}}")
    print("\\toprule")
    print("& \\multicolumn{2}{c}{\\textbf{Door}} & \\multicolumn{2}{c}{\\textbf{TiltedWipe}} & \\multicolumn{2}{c}{\\textbf{NutAssembly}} \\\\")
    print("\\cmidrule(lr){2-3}\\cmidrule(lr){4-5}\\cmidrule(lr){6-7}")
    print("\\textbf{Config} & SR (\\%) & Reward & SR (\\%) & Reward & SR (\\%) & Reward \\\\")
    print("\\midrule")
    
    prev_is_llm = False
    for config in config_list:
        is_llm = "_LLM" in config
        
        # Check if we have any data for this config across any env
        if res_df[res_df["config"] == config].empty:
            continue
            
        if is_llm and not prev_is_llm:
            print("\\midrule")
        prev_is_llm = is_llm
        
        c_fmt = config.replace('_', '\\_')
        row_str = f"\\texttt{{{c_fmt}}}"
        
        for env in ["Door", "TiltedWipe", "NutAssemblySquare"]:
            match = res_df[(res_df["env"] == env) & (res_df["config"] == config)]
            if not match.empty:
                r = match.iloc[0]
                
                env_group = res_df[res_df["env"] == env]
                sub_group = env_group[env_group['config'].str.contains('_LLM')] if is_llm else env_group[~env_group['config'].str.contains('_LLM')]
                
                best_sr = get_best_val(sub_group, 'success_mean', True)
                best_rw = get_best_val(sub_group, 'reward_mean', True)
                
                sr = fmt_cell(r['success_mean'], r['success_std'], best_sr, ":.1f", 100.0)
                rw = fmt_cell(r['reward_mean'], r['reward_std'], best_rw, ":.1f")
                row_str += f" & {sr} & {rw}"
            else:
                row_str += f" & - & -"
                
        row_str += " \\\\"
        print(row_str)

    print("\\bottomrule")
    print("\\end{tabular}")
    print("}")
    print("\\caption{Task performance metrics across all environments. Results averaged over 100 episodes.}")
    print("\\label{tab:performance_combined}")
    print("\\end{table*}\n")

    # -------------------------------------------------------------------------
    # 1.5. TiltedWipe Specific Table
    # -------------------------------------------------------------------------
    print("% LaTeX Table for TiltedWipe specific metrics")
    print("\\begin{table}[h]")
    print("\\centering")
    print("\\begin{tabular}{lccc}")
    print("\\toprule")
    print("\\textbf{Configuration} & \\textbf{SR (\\%)} $\\uparrow$ & \\textbf{Wipe (\\%)} $\\uparrow$ & \\textbf{Reward} $\\uparrow$ \\\\")
    print("\\midrule")
    
    tw_df = res_df[res_df["env"] == "TiltedWipe"]
    prev_is_llm = False
    for config in config_list:
        match = tw_df[tw_df["config"] == config]
        if not match.empty:
            r = match.iloc[0]
            is_llm = "_LLM" in config
            if is_llm and not prev_is_llm:
                print("\\midrule")
            prev_is_llm = is_llm
            
            c_fmt = config.replace('_', '\\_')
            sub_group = tw_df[tw_df['config'].str.contains('_LLM')] if is_llm else tw_df[~tw_df['config'].str.contains('_LLM')]
            
            best_sr = get_best_val(sub_group, 'success_mean', True)
            best_wp = get_best_val(sub_group, 'wipe_percentage_mean', True)
            best_rw = get_best_val(sub_group, 'reward_mean', True)
            
            sr = fmt_cell(r['success_mean'], r['success_std'], best_sr, ":.1f", 100.0)
            rw = fmt_cell(r['reward_mean'], r['reward_std'], best_rw, ":.1f")
            if 'wipe_percentage_mean' in r and not pd.isna(r['wipe_percentage_mean']):
                wp = fmt_cell(r['wipe_percentage_mean'], r['wipe_percentage_std'], best_wp, ":.1f", 100.0)
            else:
                wp = "-"
            
            print(f"\\texttt{{{c_fmt}}} & {sr} & {wp} & {rw} \\\\")
            
    print("\\bottomrule")
    print("\\end{tabular}")
    print("\\caption{TiltedWipe specific performance metrics.}")
    print("\\label{tab:tiltedwipe_performance}")
    print("\\end{table}\n")


    # -------------------------------------------------------------------------
    # 2. Safety & Smoothness Table (Stacked by Environment)
    # -------------------------------------------------------------------------
    print("% LaTeX Table for Safety & Smoothness (Stacked Environments)")
    print("\\begin{table*}[h]")
    print("\\centering")
    print("\\resizebox{\\textwidth}{!}{")
    print("\\begin{tabular}{@{}l cccccc@{}}")
    print("\\toprule")
    print("\\textbf{Config} & \\textbf{Peak force}$\\downarrow$ & \\textbf{Force exc.}$\\downarrow$ & \\textbf{Joint viol.}$\\downarrow$ & \\textbf{Eucl.\\ jerk}$\\downarrow$ & \\textbf{Riem.\\ jerk}$\\downarrow$ & \\textbf{Cond Num}$\\downarrow$ \\\\")
    print("& (N) & (per ep.) & (per ep.) & (mean) & (mean) & (mean) \\\\")
    print("\\midrule")
    
    for env, env_title in [("Door", "Door Opening"), ("TiltedWipe", "TiltedWipe"), ("NutAssemblySquare", "NutAssembly")]:
        print(f"\\multicolumn{{7}}{{c}}{{\\textit{{{env_title}}}}} \\\\")
        print("\\midrule")
        
        env_df = res_df[res_df["env"] == env]
        prev_is_llm = False
        
        for config in config_list:
            match = env_df[env_df["config"] == config]
            if match.empty:
                continue
                
            is_llm = "_LLM" in config
            if is_llm and not prev_is_llm:
                print("\\midrule")
            prev_is_llm = is_llm
            
            r = match.iloc[0]
            c_fmt = config.replace('_', '\\_')
            
            sub_group = env_df[env_df['config'].str.contains('_LLM')] if is_llm else env_df[~env_df['config'].str.contains('_LLM')]
            
            best_pk = get_best_val(sub_group, 'peak_force_mean', False)
            best_fe = get_best_val(sub_group, 'force_exceedance_rate_mean', False)
            best_jv = get_best_val(sub_group, 'joint_violations_mean', False)
            best_ej = get_best_val(sub_group, 'avg_eucl_jerk_mean', False)
            best_rj = get_best_val(sub_group, 'avg_riem_jerk_mean', False)
            best_cn = get_best_val(sub_group, 'avg_cond_num_mean', False)
            
            pk = fmt_cell(r['peak_force_mean'], r['peak_force_std'], best_pk, ":.1f")
            fe = fmt_cell(r['force_exceedance_rate_mean'], r['force_exceedance_rate_std'], best_fe, ":.1f")
            jv = fmt_cell(r['joint_violations_mean'], r['joint_violations_std'], best_jv, ":.1f")
            ej = fmt_cell(r['avg_eucl_jerk_mean'], r['avg_eucl_jerk_std'], best_ej, ":.2f")
            rj = fmt_cell(r['avg_riem_jerk_mean'], r['avg_riem_jerk_std'], best_rj, ":.2f")
            cn = fmt_cell(r['avg_cond_num_mean'], r['avg_cond_num_std'], best_cn, ":.1f")
            
            print(f"\\texttt{{{c_fmt}}} & {pk} & {fe} & {jv} & {ej} & {rj} & {cn} \\\\")
            
        if env != "NutAssemblySquare":
            print("\\midrule")

    print("\\bottomrule")
    print("\\end{tabular}")
    print("}")
    print("\\caption{Safety and smoothness metrics across all environments. Results averaged over 100 episodes.}")
    print("\\label{tab:safety_combined}")
    print("\\end{table*}\n")

if __name__ == "__main__":
    main()
