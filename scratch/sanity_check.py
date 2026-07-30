import os
import sys
import pandas as pd
import re

def parse_run_name(run_name):
    parts = run_name.split("_")
    if "SAC" not in parts: return None
    env_str = parts[1]
    
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
    log_dir = "/home/cjimenez/projects/HiRes-VIC/logs/best_models"
    
    models_found = []
    for entry in os.listdir(log_dir):
        if os.path.isdir(os.path.join(log_dir, entry)):
            parsed = parse_run_name(entry)
            if parsed:
                env_real, config, gamma, seed = parsed
                models_found.append({
                    "run_name": entry,
                    "env": env_real,
                    "config": config,
                    "gamma": gamma,
                    "seed": seed
                })
    
    df = pd.DataFrame(models_found)
    
    config_list = [
        "BASELINE", "DIAG", "LIE_ONLY", "SPD_ONLY", "FULL_GRL",
        "BASELINE_LLM", "SPD_ONLY_LLM", "FULL_GRL_LLM"
    ]
    
    for env in ["Door", "TiltedWipe", "NutAssemblySquare"]:
        print(f"\n{'='*60}\n{env.upper()} RUNS IDENTIFIED FOR EVALUATION\n{'='*60}")
        df_env = df[df["env"] == env]
        if df_env.empty:
            print("  No runs found.")
            continue
            
        for config in config_list:
            df_conf = df_env[df_env["config"] == config]
            if df_conf.empty:
                print(f"\n[ {config} ] - No runs found.")
                continue
                
            latest_gamma = df_conf["gamma"].max()
            df_target = df_conf[df_conf["gamma"] == latest_gamma]
            
            print(f"\n[ {config} ] - Gamma selected: {latest_gamma}")
            for _, row in df_target.sort_values(by="seed").iterrows():
                print(f"  -> Seed {row['seed']}: {row['run_name']}")

if __name__ == "__main__":
    main()
