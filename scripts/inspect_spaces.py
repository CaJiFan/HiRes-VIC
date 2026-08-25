import sys
import os
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))

import torch
import numpy as np
from stable_baselines3 import SAC
from scripts.evaluate_runs import make_env

def run_inspection(config_name, seed, run_type):
    model_path = f"logs/best_models/{config_name}_SEED_{seed}/best_model.zip"
    print(f"Loading {run_type} from {model_path}...")
    
    env = make_env("TiltedWipe", config_name, model_path=model_path)
    model = SAC.load(model_path, env=env, device="cpu")
    
    # We must access the underlying environment carefully due to VecEnv
    obs, info = env.reset()
    
    # Grab the robosuite env to extract raw observations later
    # evaluate_runs `make_env` returns a DummyVecEnv or SubprocVecEnv?
    # No, it returns a plain `gym.Env` wrapper. Let's see. 
    # Ah, evaluate_runs `make_env` returns `GymWrapper(env)`.
    # Wait, evaluate_runs.py make_env wraps in TiltedWipe + RobosuiteTeleportWrapper + GymWrapper.
    
    done = False
    
    output_lines = []
    output_lines.append(f"=======================================")
    output_lines.append(f"RUN: {run_type} (Seed {seed})")
    output_lines.append(f"CONFIG: {config_name}")
    output_lines.append(f"=======================================\n")
    
    step = 0
    while not done:
        # Action is a single step. Since model was given an unvectorized obs, 
        # it might return a batched action.
        action, _ = model.predict(obs, deterministic=True)
        # Flatten action for single env step
        action_np = np.array(action).flatten()
        
        # Get semantic observation (Robosuite native dictionary)
        # We access the unwrapped Robosuite env to get the raw unflattened dict
        obs_dict = env.unwrapped._get_observations()
        
        output_lines.append(f"--- STEP {step} ---")
        output_lines.append("OBSERVATION:")
        for k, v in obs_dict.items():
            # Format nicely
            if isinstance(v, np.ndarray):
                v_str = np.array2string(v, precision=4, separator=', ', max_line_width=200)
                output_lines.append(f"  {k}: {v_str} (shape: {v.shape})")
            else:
                output_lines.append(f"  {k}: {v}")
                
        output_lines.append("\nACTION (RL Output):")
        action_str = np.array2string(action_np, precision=4, separator=', ')
        output_lines.append(f"  Raw Network Output: {action_str} (shape: {action_np.shape})")
        
        # Semantic breakdown based on wrapper logic
        output_lines.append("ACTION (Semantic Breakdown):")
        if run_type == "BASELINE":
            # Baseline is 13D: 6D action -> 3D trans kp, 3D rot kp, 7D pose
            kp_trans_raw = action_np[:3]
            kp_rot_raw = action_np[3:6]
            pose_raw = action_np[6:]
            output_lines.append(f"  Translational Stiffness (Raw): {np.array2string(kp_trans_raw, precision=4)}")
            output_lines.append(f"  Rotational Stiffness (Raw): {np.array2string(kp_rot_raw, precision=4)}")
            output_lines.append(f"  Pose/Gripper Command (Raw): {np.array2string(pose_raw, precision=4)}")
        elif run_type == "SPD":
            # SPD is 16D: 9D action -> 6D Mandel, 3D rot kp, 7D pose
            kp_mandel_raw = action_np[:6]
            kp_rot_raw = action_np[6:9]
            pose_raw = action_np[9:]
            output_lines.append(f"  SPD Mandel Basis (Raw): {np.array2string(kp_mandel_raw, precision=4)}")
            output_lines.append(f"  Rotational Stiffness (Raw): {np.array2string(kp_rot_raw, precision=4)}")
            output_lines.append(f"  Pose/Gripper Command (Raw): {np.array2string(pose_raw, precision=4)}")
            
        output_lines.append("\n")
        
        obs, reward, terminated, truncated, info = env.step(action_np)
        done = terminated or truncated
        step += 1
            
    with open(f"logs/wipe/{run_type}_space_inspection.txt", "w") as f:
        f.write("\n".join(output_lines))
        
    print(f"Saved inspection to logs/wipe/{run_type}_space_inspection.txt\n")

if __name__ == "__main__":
    os.makedirs("logs/wipe", exist_ok=True)
    run_inspection("SAC_WIPE_ICRA_BASELINE_LR1e-4_H150_CURRICULUM16", 4, "BASELINE")
    run_inspection("SAC_WIPE_ICRA_SPD_LR1e-4_H150_EIGENCLAMP16", 2, "SPD")
    print("Done!")
