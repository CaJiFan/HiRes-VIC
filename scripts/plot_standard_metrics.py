import os
import pandas as pd
import seaborn as sns
import matplotlib.pyplot as plt
import matplotlib.ticker as ticker
from tensorboard.backend.event_processing.event_accumulator import EventAccumulator

# --- CONFIGURATION ---
LOG_DIR = "./outputs/logs/"
OUTPUT_DIR = "./plots/grouped/"
os.makedirs(OUTPUT_DIR, exist_ok=True)

# Fixed Colors for Consistency
CUSTOM_PALETTE = {
    "PPO": "#1f77b4",       # Blue
    "RecurrentPPO": "#1f77b4", 
    "SAC": "#ff7f0e",       # Orange
    "TD3": "#2ca02c",       # Green
    "TQC": "#d62728"        # Red
}

METRICS_MAP = {
    "rollout/ep_rew_mean": "Average Reward",
    "rollout/ep_len_mean": "Episode Length"
}

# Smoothing Factor (0.95 = Very smooth "Thesis" look)
SMOOTH_FACTOR = 0.95 

def parse_metadata(folder_name):
    parts = folder_name.split('_')
    algo = parts[0]
    raw_env = parts[1].lower()
    
    if "door" in raw_env: env_name = "Door Opening"
    elif "wipe" in raw_env: env_name = "Surface Wiping"
    elif "nutassemblyround" in raw_env: env_name = "Nut Assembly (Round)"
    elif "nutassemblysquare" in raw_env: env_name = "Nut Assembly (Square)"
    else: env_name = raw_env.capitalize()
    return algo, env_name

def load_data(root_dir):
    all_data = []
    print(f"Scanning {root_dir}...")
    
    for root, dirs, files in os.walk(root_dir):
        for file in files:
            if "events.out.tfevents" in file:
                full_path = os.path.join(root, file)
                path_parts = full_path.split(os.sep)
                experiment_folder = next((p for p in path_parts if "VIC_5M" in p), None)
                
                if not experiment_folder: continue
                algo, env_name = parse_metadata(experiment_folder)
                
                try:
                    ea = EventAccumulator(full_path)
                    ea.Reload()
                    tags = ea.Tags()['scalars']
                    
                    for tag, readable_name in METRICS_MAP.items():
                        if tag in tags:
                            events = ea.Scalars(tag)
                            steps = [e.step for e in events]
                            values = [e.value for e in events]
                            
                            for s, v in zip(steps, values):
                                all_data.append({
                                    "Timesteps": s,
                                    "Value": v,
                                    "Metric": readable_name,
                                    "Algorithm": algo,
                                    "Environment": env_name
                                })
                except Exception as e:
                    print(f"    Error reading {file}: {e}")
    return pd.DataFrame(all_data)

def plot_custom(df):
    if df.empty:
        print("No data found!")
        return

    environments = df["Environment"].unique()
    sns.set_theme(style="whitegrid", context="paper", font_scale=1.5)
    
    for env in environments:
        print(f"Plotting {env}...")
        env_data = df[df["Environment"] == env]
        
        # Layout Logic
        is_wiping = "Wiping" in env or "Wipe" in env
        if is_wiping:
            fig, axes = plt.subplots(1, 2, figsize=(16, 6))
            plot_configs = [(axes[0], "Average Reward"), (axes[1], "Episode Length")]
            fig.suptitle(f"{env} (VIC)", fontsize=20, weight='bold')
        else:
            fig, ax = plt.subplots(1, 1, figsize=(10, 6))
            plot_configs = [(ax, "Average Reward")]
            fig.suptitle(f"{env} (VIC)", fontsize=18, weight='bold')

        for ax, metric in plot_configs:
            metric_data = env_data[env_data["Metric"] == metric]
            algorithms = metric_data["Algorithm"].unique()
            
            for algo in sorted(algorithms):
                algo_df = metric_data[metric_data["Algorithm"] == algo].sort_values("Timesteps")
                if algo_df.empty: continue
                
                # Calculate Statistics
                smoothed_mean = algo_df["Value"].ewm(alpha=(1-SMOOTH_FACTOR)).mean()
                smoothed_std = algo_df["Value"].ewm(alpha=(1-SMOOTH_FACTOR)).std().fillna(0)
                
                color = CUSTOM_PALETTE.get(algo, "#333333")
                
                # Plot the Line
                ax.plot(algo_df["Timesteps"], smoothed_mean, color=color, label=algo, linewidth=2.5)
                
                # Plot Shadow ONLY if it's NOT Episode Length
                if metric != "Episode Length":
                    ax.fill_between(
                        algo_df["Timesteps"],
                        smoothed_mean - smoothed_std,
                        smoothed_mean + smoothed_std,
                        color=color, alpha=0.2, linewidth=0
                    )

            # Formatting
            ax.set_title(metric, fontsize=16)
            ax.set_xlabel("Timesteps")
            if metric == "Average Reward":
                ax.set_ylabel("Reward")
            else:
                ax.set_ylabel("Steps")
                
            ylabel = lambda x, pos: f'{x//1e3:.1f}K' if 'Door' in env else f'{x//1e6:.1f}M'
            ax.xaxis.set_major_formatter(ticker.FuncFormatter(ylabel))
            ax.legend(title="Algorithm", loc="best", frameon=True)
            ax.grid(True, which='major', linestyle='--', alpha=0.7)

        plt.tight_layout()
        safe_name = env.replace(" ", "_").replace("(", "").replace(")", "").lower()
        save_path = os.path.join(OUTPUT_DIR, f"{safe_name}_metrics.png")
        plt.savefig(save_path, dpi=300)
        print(f"  Saved: {save_path}")
        plt.close()

if __name__ == "__main__":
    df = load_data(LOG_DIR)
    plot_custom(df)
    print("\nDone! Check './plots/grouped/'")