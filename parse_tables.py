import pandas as pd

df = pd.read_csv('outputs/chapter6_metrics.csv')
# Filter out LLM configs
df = df[~df['config'].str.contains('LLM')]

configs = ['BASELINE', 'DIAG', 'LIE_ONLY', 'SPD_ONLY', 'FULL_GRL']
tasks = ['Door', 'TiltedWipe', 'NutAssemblySquare']

def fmt(mean, std, mult=1.0):
    if pd.isna(mean): return "---"
    return f"{mean*mult:.1f} \\pm {std*mult:.1f}"

print("=== Task Performance Table ===")
for conf in configs:
    row = [f"\\texttt{{{conf}}}"]
    for task in tasks:
        sub = df[(df['env'] == task) & (df['config'] == conf)]
        if len(sub) == 0:
            row.extend(["---", "---"])
            continue
        s = sub.iloc[0]
        if task == 'TiltedWipe':
            # Use wipe_percentage for SR column
            sr_str = fmt(s['wipe_percentage_mean'], s['wipe_percentage_std'], 100.0)
        else:
            sr_str = fmt(s['success_mean'], s['success_std'], 100.0)
        rew_str = fmt(s['reward_mean'], s['reward_std'], 1.0)
        row.extend([sr_str, rew_str])
    print(" & ".join(row) + " \\\\")

print("\n=== Safety Metrics Table ===")
for task in tasks:
    print(f"\\midrule\n\\multicolumn{{6}}{{c}}{{\\textit{{{task}}}}} \\\\\n\\midrule")
    for conf in configs:
        sub = df[(df['env'] == task) & (df['config'] == conf)]
        if len(sub) == 0: continue
        s = sub.iloc[0]
        row = [f"\\texttt{{{conf}}}"]
        # Joint viol, Force exc (rate * 100? or just rate? The prompt says "force exceedance rate". Let's output raw or *100. Force exc. in table is "per ep." or rate? In CSV it is rate. Let's leave as rate * maybe horizon? Let's just print the values)
        # Horizon for door=50, wipe=300, nas=150. If force_exceedance_rate is per step, * H gives per ep.
        H = 50 if task == 'Door' else (300 if task == 'TiltedWipe' else 150)
        jv = fmt(s['joint_violations_mean'], s['joint_violations_std'])
        fe = fmt(s['force_exceedance_rate_mean'] * H, s['force_exceedance_rate_std'] * H)
        pf = fmt(s['peak_force_mean'], s['peak_force_std'])
        ej = fmt(s['avg_eucl_jerk_mean'], s['avg_eucl_jerk_std'])
        rj = fmt(s['avg_riem_jerk_mean'], s['avg_riem_jerk_std']) if conf in ['SPD_ONLY', 'FULL_GRL'] else "---"
        
        row.extend([jv, fe, pf, ej, rj])
        print(" & ".join(row) + " \\\\")
