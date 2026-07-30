import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
import os

# ==============================================================================
# Configuration
# ==============================================================================
OFFLINE_MODE = True

OFFLINE_CSVS = {
    "rollout/success_rate|wipe": "manuscript/data/wandb_export_2026-06-22T10_36_10.355+02_00_SUCCESS_RATE.csv",
    "physics/raw_wipe_percentage|wipe": "manuscript/data/wandb_export_2026-06-30T18_56_15.331+02_00_RAW_WIPE_FINAL2.csv",
    "rollout/success_rate|nas": "manuscript/data/wandb_export_2026-06-24T21_19_56.482+02_00_NAS_FINAL.csv",
    "rollout/success_rate|door": "manuscript/data/wandb_export_2026-06-24T21_10_48.877+02_00_DOOR_FINAL.csv"
}

METRICS_TO_PLOT = [
    "rollout/success_rate|door",
    "rollout/success_rate|wipe",
    "physics/raw_wipe_percentage|wipe",
    "rollout/success_rate|nas"
]

MAX_STEPS = 1000000

# Define the paired plots we want to generate
PAIRED_PLOTS = {
    "A_Orientation": ["BASELINE", "LIE_ONLY"],
    "B_Stiffness": ["BASELINE", "DIAG", "SPD_ONLY"],
    "C_Interference": ["LIE_ONLY", "SPD_ONLY", "FULL_GRL"],
    "D_LLM_Prior": ["SPD_ONLY", "SPD_ONLY_LLM", "BASELINE_LLM", "FULL_GRL_LLM"]
}

# Define a consistent color scheme for each configuration
CONFIG_COLORS = {
    "BASELINE": "#1f77b4",       # Blue
    "BASELINE_LLM": "#ff7f0e",   # Orange
    "FULL_GRL": "#2ca02c",       # Green
    "FULL_GRL_LLM": "#d62728",   # Red
    "LIE_ONLY": "#9467bd",       # Purple
    "SPD_ONLY": "#8c564b",       # Brown
    "SPD_ONLY_LLM": "#e377c2",   # Pink
    "DIAG": "#17becf"            # Cyan
}
# ==============================================================================

def get_clean_name(raw_name):
    """Extracts a standardized, clean config name from the W&B column name."""
    # Remove the generic parts
    n = raw_name.split(" - ")[0].replace("Name: ", "")
    n = n.replace("SAC_TILTEDWIPE_", "").replace("SAC_NUTASSEMBLYSQUARE_", "").replace("SAC_DOOR_", "").replace("_FINAL", "").strip()
    
    # Check for LLM flags
    is_llm = "_LLM" in n
    
    # Identify base geometry
    if "BASELINE" in n: base = "BASELINE"
    elif "DIAG" in n: base = "DIAG"
    elif "LIE_ONLY" in n: base = "LIE_ONLY"
    elif "SPD_ONLY" in n: base = "SPD_ONLY"
    elif "FULL_GRL" in n or "FULLGRL" in n: base = "FULL_GRL"
    else: base = "UNKNOWN"
    
    if is_llm:
        return f"{base}_LLM"
    return base

def setup_matplotlib():
    sns.set_theme(style="whitegrid", context="paper")
    plt.rcParams.update({
        "font.family": "serif",
        "font.size": 12,
        "axes.labelsize": 14,
        "axes.titlesize": 16,
        "legend.fontsize": 11,
        "xtick.labelsize": 12,
        "ytick.labelsize": 12,
        "axes.edgecolor": "black",
        "axes.linewidth": 1.2,
        "legend.frameon": True,
        "legend.edgecolor": "black",
        "figure.figsize": (7, 4.5), # Slightly more compact for paired plots
    })

def get_smoothed_and_bounds(y, span=100):
    import pandas as pd
    series = pd.Series(y).ffill().bfill()
    # EWM (Exponential Weighted Mean) exactly replicates WandB/Tensorboard smoothing
    y_smooth = series.ewm(span=span, adjust=False).mean().values
    
    # Reconstruct true standard deviation across unaligned seeds using rolling window
    y_std = series.rolling(window=span, min_periods=1, center=True).std().fillna(0).values
    
    y_min = y_smooth - y_std
    y_max = y_smooth + y_std
    return y_smooth, y_min, y_max

def format_plot(metric_name):
    base_metric = metric_name.split('|')[0]
    plt.xlabel("Global Step")
    clean_title = base_metric.split('/')[-1].replace('_', ' ').title()
    plt.ylabel(clean_title)
    plt.xlim(0, MAX_STEPS)
    
    ax = plt.gca()
    ax.xaxis.set_major_formatter(plt.FuncFormatter(lambda x, pos: f'{int(x/1000)}k' if x < 1e6 else f'{x/1e6:g}M'))
    
    if "percentage" in metric_name or "success_rate" in metric_name:
        plt.ylim(0, 1.0)
        
    plt.legend(
        loc='upper center',
        bbox_to_anchor=(0.5, -0.15),
        ncol=4,
        frameon=False,
        fontsize=11,
        handlelength=1.5,
        columnspacing=1.0
    )
    plt.tight_layout()

def plot_offline(metric_name, csv_path):
    print(f"\n[OFFLINE] Reading CSV for {metric_name}...")
    if not os.path.exists(csv_path):
        print(f"File not found: {csv_path}")
        return
        
    df = pd.read_csv(csv_path)
    step_col = df.columns[0]
    common_x = df[step_col].values
    
    base_metric = metric_name.split("|")[0]
    mean_cols = [c for c in df.columns if base_metric in c and "__MIN" not in c and "__MAX" not in c and "Name:" in c]
    
    if not mean_cols:
        print(f"Could not find valid columns for {base_metric} in the CSV.")
        return

    # Extract all data into a clean dictionary
    parsed_data = {}
    for col in mean_cols:
        clean_name = get_clean_name(col)
        y_mean = df[col].values
        
        min_col = col + "__MIN"
        max_col = col + "__MAX"
        y_min = df[min_col].values if min_col in df.columns else None
        y_max = df[max_col].values if max_col in df.columns else None
        
        parsed_data[clean_name] = {
            'mean': y_mean,
            'min': y_min,
            'max': y_max
        }

    # Generate the single combined plot
    setup_matplotlib()
    plt.figure()
    
    # We want a consistent coloring scheme for the single plots
    valid_single_configs = list(parsed_data.keys())
    if valid_single_configs:
        for config_name in valid_single_configs:
            color = CONFIG_COLORS.get(config_name, "#000000")
            data = parsed_data[config_name]
            y_mean = data['mean']
            
            mask = ~np.isnan(y_mean)
            if not np.any(mask): continue
                
            x_clean = common_x[mask]
            y_clean = y_mean[mask]
            y_smooth, y_min, y_max = get_smoothed_and_bounds(y_clean, span=100)
            
            label = config_name.replace("_", " ")
            
            plt.plot(x_clean, y_smooth, label=label, color=color, linewidth=2.0, alpha=0.9)
            
            # For the single plot, we might want to drop fill_between to avoid too much clutter,
            # but let's keep a very faint fill just in case.
            plt.fill_between(x_clean, y_min, y_max, color=color, alpha=0.15, linewidth=0)

        format_plot(metric_name)
        safe_filename_single = f"{metric_name.split('/')[-1].replace('|', '_')}.pdf"
        os.makedirs("plots", exist_ok=True)
        plt.savefig(f"plots/{safe_filename_single}", format='pdf', dpi=300, bbox_inches='tight')
        plt.close()
        print(f"✅ Saved single plot to plots/{safe_filename_single}")

    # Generate the paired plots
    for plot_group_name, config_list in PAIRED_PLOTS.items():
        setup_matplotlib()
        plt.figure()
        
        # We only want to plot configs that actually exist in this parsed data
        valid_configs = [c for c in config_list if c in parsed_data]
        if not valid_configs:
            plt.close()
            continue
            
        for config_name in valid_configs:
            color = CONFIG_COLORS.get(config_name, "#000000")
            data = parsed_data[config_name]
            y_mean = data['mean']
            
            mask = ~np.isnan(y_mean)
            if not np.any(mask): continue
                
            x_clean = common_x[mask]
            y_clean = y_mean[mask]
            y_smooth, y_min, y_max = get_smoothed_and_bounds(y_clean, span=100)
            
            # Map clean name back to readable labels for the paper
            label = config_name.replace("_", " ")
            
            plt.plot(x_clean, y_smooth, label=label, color=color, linewidth=2.5)
            plt.fill_between(x_clean, y_min, y_max, color=color, alpha=0.15, linewidth=0)

        format_plot(metric_name)
        
        # Save plot
        safe_filename = f"{metric_name.split('/')[-1].replace('|', '_')}_{plot_group_name}.pdf"
        os.makedirs("plots/paired", exist_ok=True)
        plt.savefig(f"plots/paired/{safe_filename}", format='pdf', dpi=300, bbox_inches='tight')
        plt.close()
        print(f"✅ Saved {plot_group_name} plot to plots/paired/{safe_filename}")

if __name__ == "__main__":
    for metric in METRICS_TO_PLOT:
        csv_path = OFFLINE_CSVS.get(metric)
        if csv_path:
            plot_offline(metric, csv_path)
