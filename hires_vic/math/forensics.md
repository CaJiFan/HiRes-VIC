Listed directory wipe
Ran command: `tail -n 100 /home/cjimenez/projects/HiRes-VIC/logs/wipe/TiltedWipe_WIPE_ICRA_SPD_LLMv4_LLAMA_W0.50_LR3e-4_H150_LOGEUC_SEED_0.log`
Ran command: `python3 -c "
import glob

for f in sorted(glob.glob('logs/wipe/*SPD_LLMv4*.log')):
    lines = open(f).readlines()
    last_sr = 'N/A'
    last_step = 'N/A'
    for line in lines[-200:]:
        if 'success_rate' in line:
            last_sr = line.split('|')[2].strip()
        if 'total_timesteps' in line:
            last_step = line.split('|')[2].strip()
    print(f'{f}: step={last_step}, sr={last_sr}')
"`
Ran command: `python3 -c "
import glob

for f in sorted(glob.glob('logs/wipe/*LLMv4*.log')):
    lines = open(f).readlines()
    last_sr = 'N/A'
    last_step = 'N/A'
    for line in lines[-200:]:
        if 'success_rate' in line:
            last_sr = line.split('|')[2].strip()
        if 'total_timesteps' in line:
            last_step = line.split('|')[2].strip()
    print(f'{f}: step={last_step}, sr={last_sr}')
"`
Viewed task-369.log
Ran command: `python3 -c "
import glob, os

for f in sorted(glob.glob('logs/wipe/*LLMv4*.log')):
    with open(f, 'rb') as fp:
        try:
            fp.seek(-20000, os.SEEK_END)
        except Exception:
            pass
        chunk = fp.read().decode('utf-8', errors='ignore')
    lines = chunk.splitlines()
    last_sr = 'N/A'
    last_step = 'N/A'
    for line in lines:
        if 'success_rate' in line:
            parts = line.split('|')
            if len(parts) > 2:
                last_sr = parts[2].strip()
        if 'total_timesteps' in line:
            parts = line.split('|')
            if len(parts) > 2:
                last_step = parts[2].strip()
    print(f'{os.path.basename(f)}: step={last_step}, sr={last_sr}')
"`
Ran command: `python3 -c "
import os

def parse_log(path):
    with open(path, 'r') as f:
        content = f.read()
    blocks = content.split('-------------------------------------------')
    records = []
    for b in blocks:
        lines = b.strip().splitlines()
        rec = {}
        for l in lines:
            if '|' in l:
                parts = [p.strip() for p in l.split('|') if p.strip()]
                if len(parts) == 2:
                    k, v = parts
                    rec[k] = v
        if 'total_timesteps' in rec:
            records.append(rec)
    return records

rec0 = parse_log('logs/wipe/TiltedWipe_WIPE_ICRA_SPD_LLMv4_LLAMA_W0.50_LR3e-4_H150_LOGEUC_SEED_0.log')
rec1 = parse_log('logs/wipe/TiltedWipe_WIPE_ICRA_SPD_LLMv4_LLAMA_W0.50_LR3e-4_H150_LOGEUC_SEED_1.log')
rec2 = parse_log('logs/wipe/TiltedWipe_WIPE_ICRA_SPD_LLMv4_LLAMA_W0.50_LR3e-4_H150_LOGEUC_SEED_2.log')

print('--- SEED 0 (Success) ---')
for r in rec0[::max(1, len(rec0)//5)]:
    print(f\"step={r.get('total_timesteps')}, sr={r.get('success_rate')}, raw_wipe={r.get('raw_wipe_percentage')}, ent={r.get('ent_coef')}, kx={r.get('kp_trans_x_avg')}, ky={r.get('kp_trans_y_avg')}, kz={r.get('kp_trans_z_avg')}, rot_y={r.get('kp_rot_y_avg')}, eigmax={r.get('spd_pre_clamp_eigmax_avg')}, peak_f={r.get('peak_force')}, modes=(app:{r.get('pct_approach')}, con:{r.get('pct_contact')}, wipe:{r.get('pct_wipe')})\")

print('\n--- SEED 1 (Failed) ---')
for r in rec1[::max(1, len(rec1)//5)]:
    print(f\"step={r.get('total_timesteps')}, sr={r.get('success_rate')}, raw_wipe={r.get('raw_wipe_percentage')}, ent={r.get('ent_coef')}, kx={r.get('kp_trans_x_avg')}, ky={r.get('kp_trans_y_avg')}, kz={r.get('kp_trans_z_avg')}, rot_y={r.get('kp_rot_y_avg')}, eigmax={r.get('spd_pre_clamp_eigmax_avg')}, peak_f={r.get('peak_force')}, modes=(app:{r.get('pct_approach')}, con:{r.get('pct_contact')}, wipe:{r.get('pct_wipe')})\")

print('\n--- SEED 2 (Learning) ---')
for r in rec2[::max(1, len(rec2)//5)]:
    print(f\"step={r.get('total_timesteps')}, sr={r.get('success_rate')}, raw_wipe={r.get('raw_wipe_percentage')}, ent={r.get('ent_coef')}, kx={r.get('kp_trans_x_avg')}, ky={r.get('kp_trans_y_avg')}, kz={r.get('kp_trans_z_avg')}, rot_y={r.get('kp_rot_y_avg')}, eigmax={r.get('spd_pre_clamp_eigmax_avg')}, peak_f={r.get('peak_force')}, modes=(app:{r.get('pct_approach')}, con:{r.get('pct_contact')}, wipe:{r.get('pct_wipe')})\")
"`
Viewed task-375.log
Ran command: `python3 -c "
import os

def parse_log_fast(path):
    # read last 2MB
    with open(path, 'rb') as f:
        f.seek(0, os.SEEK_END)
        size = f.tell()
        f.seek(max(0, size - 2*1024*1024))
        content = f.read().decode('utf-8', errors='ignore')
    blocks = content.split('-------------------------------------------')
    for b in blocks[-10:]:
        lines = b.strip().splitlines()
        rec = {}
        for l in lines:
            if '|' in l:
                parts = [p.strip() for p in l.split('|') if p.strip()]
                if len(parts) == 2:
                    k, v = parts
                    rec[k] = v
        if 'total_timesteps' in rec:
            print(f\"{os.path.basename(path)} | step={rec.get('total_timesteps')} | sr={rec.get('success_rate')} | raw_wipe={rec.get('raw_wipe_percentage')} | ent={rec.get('ent_coef')} | kx={rec.get('kp_trans_x_avg')} | ky={rec.get('kp_trans_y_avg')} | kz={rec.get('kp_trans_z_avg')} | rot_y={rec.get('kp_rot_y_avg')} | eigmax={rec.get('spd_pre_clamp_eigmax_avg')}\")

for s in [0, 1, 2]:
    p = f'logs/wipe/TiltedWipe_WIPE_ICRA_SPD_LLMv4_LLAMA_W0.50_LR3e-4_H150_LOGEUC_SEED_{s}.log'
    if os.path.exists(p):
        print(f'=== SEED {s} ===')
        parse_log_fast(p)
"`
Viewed task-379.log
Used tool: manage_task
Used tool: manage_task
Used tool: manage_task
Used tool: manage_task
Used tool: manage_task
Ran command: `tail -n 80 logs/wipe/TiltedWipe_WIPE_ICRA_SPD_LLMv4_LLAMA_W0.50_LR3e-4_H150_LOGEUC_SEED_1.log`
Ran command: `grep -E "critic_loss|total_timesteps|success_rate" logs/wipe/TiltedWipe_WIPE_ICRA_SPD_LLMv4_LLAMA_W0.50_LR3e-4_H150_LOGEUC_SEED_1.log | paste - - - | head -n 40`
Ran command: `grep -E "critic_loss|total_timesteps|success_rate|raw_wipe_percentage|ep_rew_mean|peak_force" logs/wipe/TiltedWipe_WIPE_ICRA_SPD_LLMv4_LLAMA_W0.50_LR3e-4_H150_LOGEUC_SEED_1.log | grep -A 5 "100[0-9]\{3\}" | head -n 30`
Ran command: `python3 -c "
import glob

def check_steps(path):
    with open(path) as f:
        content = f.read()
    blocks = content.split('-------------------------------------------')
    for b in blocks:
        if any(f'|    total_timesteps           | {s} ' in b or f'|    total_timesteps           | {s}' in b for s in [50000, 100000, 200000, 300000, 400000, 500000, 600000, 700000]):
            step = '?'
            sr = '?'
            rw = '?'
            eig = '?'
            peak_f = '?'
            closs = '?'
            kx = ky = kz = '?'
            for l in b.splitlines():
                if 'total_timesteps' in l: step = l.split('|')[2].strip()
                if 'success_rate' in l: sr = l.split('|')[2].strip()
                if 'raw_wipe_percentage' in l: rw = l.split('|')[2].strip()
                if 'spd_pre_clamp_eigmax_avg' in l: eig = l.split('|')[2].strip()
                if 'peak_force' in l: peak_f = l.split('|')[2].strip()
                if 'critic_loss' in l: closs = l.split('|')[2].strip()
                if 'kp_trans_x_avg' in l: kx = l.split('|')[2].strip()
                if 'kp_trans_y_avg' in l: ky = l.split('|')[2].strip()
                if 'kp_trans_z_avg' in l: kz = l.split('|')[2].strip()
            print(f'step={step} | sr={sr} | wipe={rw} | eigmax={eig} | peak_f={peak_f} | closs={closs} | k=({kx}, {ky}, {kz})')

print('=== SEED 0 ===')
check_steps('logs/wipe/TiltedWipe_WIPE_ICRA_SPD_LLMv4_LLAMA_W0.50_LR3e-4_H150_LOGEUC_SEED_0.log')
print('\n=== SEED 1 ===')
check_steps('logs/wipe/TiltedWipe_WIPE_ICRA_SPD_LLMv4_LLAMA_W0.50_LR3e-4_H150_LOGEUC_SEED_1.log')
print('\n=== SEED 2 ===')
check_steps('logs/wipe/TiltedWipe_WIPE_ICRA_SPD_LLMv4_LLAMA_W0.50_LR3e-4_H150_LOGEUC_SEED_2.log')
"`
Viewed task-403.log
Used tool: manage_task
Ran command: `grep -E "total_timesteps|success_rate|raw_wipe_percentage|spd_pre_clamp_eigmax_avg|critic_loss" logs/wipe/TiltedWipe_WIPE_ICRA_SPD_LLMv4_LLAMA_W0.50_LR3e-4_H150_LOGEUC_SEED_0.log | paste - - - - - | awk 'NR % 100 == 1'`
Ran command: `grep -E "total_timesteps|success_rate|raw_wipe_percentage|spd_pre_clamp_eigmax_avg|critic_loss" logs/wipe/TiltedWipe_WIPE_ICRA_SPD_LLMv4_LLAMA_W0.50_LR3e-4_H150_LOGEUC_SEED_1.log | paste - - - - - | awk 'NR % 100 == 1'`
Ran command: `grep -E "total_timesteps|success_rate|raw_wipe_percentage|spd_pre_clamp_eigmax_avg|critic_loss" logs/wipe/TiltedWipe_WIPE_ICRA_SPD_LLMv4_LLAMA_W0.50_LR3e-4_H150_LOGEUC_SEED_2.log | paste - - - - - | awk 'NR % 100 == 1'`

Here is the full forensic breakdown of what is happening in the WandB curves and why the SPD run is behaving this way:

---

### 1. What the Logs Actually Reveal Across Seeds

Looking at the individual seed logs for `SAC_WIPE_ICRA_SPD_LLMv4...`:

- **Seed 0**: **Converged to 100% Success Rate** (completed successfully at 580k steps with $1.0$ SR, low jerk, and stable loss).
- **Seed 2**: **Converged to 85% Success Rate** (currently at 738k steps with $0.85$ SR and rising).
- **Seed 1**: **Collapsing at 0% Success Rate** after initially reaching $40\%$ SR at 330k steps.

The green curve in WandB is hovering around $\sim 55-60\%$ because it plots the mean across all 3 seeds:
$$\text{Mean SR} = \frac{1.00 + 0.85 + 0.00}{3} \approx 61\%$$

---

### 2. Forensic Diagnosis: Why Did Seed 1 Collapse in SPD while Baseline Succeeded in All Seeds?

Inspecting the exact metrics in `Seed 1` right when it collapsed between steps 330k and 400k reveals:

```
Step 332k: critic_loss = 0.331 | success_rate = 0.40 | peak_force = 157 N
Step 402k: critic_loss = 3.160 | success_rate = 0.10 | peak_force = 474 N
Step 465k: critic_loss = 13.00 | success_rate = 0.02 | peak_force = 441 N
Step 716k: critic_loss = 46.60 | success_rate = 0.00 | peak_force = 344 N
```

#### Why did the Critic Loss explode to $46.6$ in SPD?

1. **The Log-Euclidean Exponentiation Overshoot**:
   In `contact` / `wipe` mode:
   - $K_{\text{prior}} = [250, 250, 50]\text{ N/m}$
   - $S_{\text{prior}} = \ln(250) \approx 5.52$
   - When the SAC policy outputs $a_x = +1.0$ with $w = 0.50$ and `tangent_scale_diag = 2.5`:
     $$\Delta S_x = 1.0 \times (1 - 0.5) \times 2.5 = +1.25$$
     $$S_{\text{total}} = 5.52 + 1.25 = 6.77$$
     $$K_x = \exp(6.77) = \mathbf{872\text{ N/m}}!$$

2. **The Steep Gradient & Contact Force Spikes**:
   - Look at the log metric `spd_pre_clamp_eigmax_peak`: it reached **$919\text{ N/m}$** and **$868\text{ N/m}$**!
   - Because the derivative of the exponential map $\frac{d}{dS}\exp(S) \approx 900$ is extremely steep near $S = 6.77$, small policy actions created massive stiffness swings ($\Delta K \approx 100\text{ N/m}$).
   - This triggered high contact force spikes (**$474\text{ N}$** peak force in Seed 1), generating extreme reward penalties and Bellman TD error spikes in SAC.
   - In high-variance seeds like Seed 1, this caused the Q-critic loss to blow up ($46.6$), destabilizing the policy.

3. **Why Baseline (Euclidean RPL) Did NOT Suffer from This**:
   In Baseline:
   $$K = \operatorname{clip}(K_{\text{prior}} + (1 - w) \Delta K_{\max} a,\, 1,\, 300)$$
   - The derivative $\frac{dK}{da} = 75\text{ N/m}$ is strictly constant and linear.
   - It never overshoots $300\text{ N/m}$, produces no exponential force spikes, and all seeds converge smoothly to $100\%$.

4. **Why Pure RL SPD Did NOT Suffer from This**:
   In pure RL SPD (without LLM):
   $$m_{\text{params}}[0:3] = \ln(\text{target\_physical}) \quad \text{where } \text{target\_physical} \in [1, 300]$$
   - The log coordinates are strictly bounded in $[0, \ln(300)] = [0, 5.70]$. The pre-clamp eigenvalues never exceed $300\text{ N/m}$.

---

### 3. The Mathematical Fix: Geodesic Bounded Tangent Residual

To make Log-Euclidean SPD as stable and consistent as Baseline across all seeds, we should bound the tangent displacement so it **cannot exponentiate beyond $[K_{\min}, K_{\max}]$**.

Instead of a fixed unconstrained multiplier $\Delta S = a \cdot 2.5$, we scale the tangent step based on the remaining geodesic distance to the manifold boundary:

$$\Delta S_i = \begin{cases} 
a_i \cdot (1 - w) \cdot \Big(\ln(K_{\max}) - \ln(K_{\text{prior}, i})\Big) & \text{if } a_i \ge 0 \\[8pt]
a_i \cdot (1 - w) \cdot \Big(\ln(K_{\text{prior}, i}) - \ln(K_{\min})\Big) & \text{if } a_i < 0 
\end{cases}$$

And for off-diagonal terms:
$$\Delta S_{3:6} = a_{3:6} \cdot (1 - w) \cdot 0.2 \quad (\text{matching pure RL SPD})$$

#### Why this solves the problem:
1. **Exact Prior**: When $a = 0$, $\Delta S = 0 \implies K = K_{\text{prior}}$ (zero residual error).
2. **Strictly Bounded on the Manifold Cone**: For any $a \in [-1, 1]$ and any $w \in [0, 1]$, $K \in [K_{\min}, K_{\max}]$ **guaranteed**.
3. **No Exponential Overshoot**: $K$ never exceeds $300\text{ N/m}$ (pre-clamp eigenvalues stay $\le 300$), eliminating the $900\text{ N/m}$ force spikes and critic explosions.
4. **Scale-Invariant & Isometric**: Preserves the Riemannian geometry of $\mathcal{S}_{++}^3$ while guaranteeing numerical stability in off-policy SAC.

Would you like me to implement this bounded Log-Euclidean residual mapping in `hires_vic/wrappers/geometric.py`?