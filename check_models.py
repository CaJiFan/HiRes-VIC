import os
import sys
import pandas as pd
sys.path.append(os.path.abspath('.'))
from scripts.generate_door_tables import parse_run_name

log_dir = "/home/cjimenez/projects/HiRes-VIC/logs/best_models"
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
                    "seed": seed
                })

df = pd.DataFrame(models_found)
door_df = df[df["env"] == "Door"]
door_df = door_df.sort_values(by=["config", "gamma", "seed"])

print(f"Total valid Door models found: {len(door_df)}")
for _, row in door_df.iterrows():
    print(f"  Config: {row['config']:<15} | Seed: {row['seed']} | Gamma: {row['gamma']} | Dir: {row['run_name']}")
