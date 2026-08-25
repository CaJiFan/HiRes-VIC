Viewed wipe_impedance_profile_v2.yaml:1-67
Listed directory configs
Viewed wipe_impedance_profile.yaml:1-80
Searched for "wipe_impedance_profile"
Viewed impedance_planner.py:110-250
Viewed impedance_planner.py:50-110
Viewed riemannian.py:1-60
Listed directory wipe
Viewed TiltedWipe_WIPE_ICRA_SPD_LR3e-4_H150_HQSEQ_NM5_G0.95_NORMv1_SEED_0.log:150-300
Viewed geometric.py:430-480
Searched for "def _flatten_obs"
Viewed geometric.py:270-325
Viewed geometric.py:325-370
Viewed impedance_planner.py:360-430
Viewed impedance_planner.py:460-520
Viewed TiltedWipe_WIPE_ICRA_SPD_LLMv3_LLAMA_W0.50_LR3e-4_H150_LOGEUC_SEED_0.log:1-150
Viewed TiltedWipe_WIPE_ICRA_SPD_LLMv3_LLAMA_W0.50_LR3e-4_H150_LOGEUC_SEED_0.log:5000-5100
Searched for "pct_wipe"

Based on the WandB curves and an inspection of your training logs ([`TiltedWipe_WIPE_ICRA_SPD_LLMv3_...log`](file:///home/cjimenez/projects/HiRes-VIC/logs/wipe/TiltedWipe_WIPE_ICRA_SPD_LLMv3_LLAMA_W0.50_LR3e-4_H150_LOGEUC_SEED_0.log)), we have identified the **exact root causes** of why Baseline+LLM was slower than pure Baseline, and why SPD+LLM underperformed.

---

### Root Cause 1: The Closed-Loop LLM Deadlock (`pct_wipe = 0`)

In the log files across hundreds of episodes:
```
| pct_approach   | 0.545 |
| pct_transition | 0.455 |
| pct_wipe       | 0.000 |  <-- The LLM NEVER predicted 'wipe'
```

#### Why did this happen?
1. The prompt in [`configs/wipe_impedance_profile_v2.yaml`](file:///home/cjimenez/projects/HiRes-VIC/configs/wipe_impedance_profile_v2.yaml#L54-L65) defines:
   - `transition`: *"EEF has just made contact (contact flag just flipped True)."*
   - `wipe`: *"EEF is in sustained contact and actively sweeping across markers."*
2. When the robot touches the table, `In contact: True`, but `Wipe completion: 0.0%`.
3. Llama-3.2 sees `Wipe completion: 0.0%` and classifies the state as `transition`.
4. In `transition`, the profile commands $K_{\text{trans}} = [10, 100, 10]\text{ N/m}$ (an extremely soft $10\text{ N/m}$ along the normal).
5. With normal stiffness of only $10\text{ N/m}$, the contact force is only $\sim 0.4\text{ N}$ (far below the $15\text{ N}$ target), which is too weak to wipe markers.
6. Because no markers are wiped, `Wipe completion` stays $0.0\%$, trapping the LLM in `transition` **forever**.

---

### Root Cause 2: Why SPD Suffered Much More Than Baseline (Logarithmic Bottleneck)

In [`configs/wipe_impedance_profile_v2.yaml`](file:///home/cjimenez/projects/HiRes-VIC/configs/wipe_impedance_profile_v2.yaml), translational stiffness was set to $10\text{ N/m}$:

1. **In Baseline (Linear RPL)**:
   $$K = \operatorname{clip}(K_{\text{prior}} + (1 - w) \Delta K_{\max} a)$$
   With $w = 0.5$, $(1 - w)\Delta K_{\max} = 0.5 \times 149.5 \approx 75\text{ N/m}$.
   The baseline agent could add $+75\text{ N/m}$ linearly: $10 + 75 = \mathbf{85\text{ N/m}}$. This allowed Baseline to slowly overcome the limp prior and reach 100% SR.

2. **In SPD (Log-Euclidean RPL)**:
   $$S_{\text{total}} = \ln(10) + (1 - w) \cdot 2.5 \cdot a = 2.3026 + 1.25 \cdot a$$
   The **maximum stiffness** the SPD agent could reach even with $a = +1.0$ was:
   $$K_{\max} = \exp(2.3026 + 1.25) = \exp(3.55) = \mathbf{34.9\text{ N/m}}$$
   Because of the logarithmic map, an anchor of $10\text{ N/m}$ **physically capped the SPD policy to $\le 35\text{ N/m}$**.
   The pure RL agent learns that it needs $K \approx 120-170\text{ N/m}$ to wipe markers cleanly and track the $15\text{ N}$ force reward. The SPD agent was physically starved of stiffness.

---

### Root Cause 3: Underestimated Rotational Stiffness ($K_{\text{rot}}$)

- The profile set $K_{\text{rot}} = [30, 30, 30]\text{ N}\cdot\text{m/rad}$.
- For a flat wiper on a $45^\circ$ tilted board, $30\text{ N}\cdot\text{m/rad}$ is too compliant. When touching the board, the wiper twists away from being parallel to the board, creating corner/edge contact instead of full pad contact.
- Pure RL learns $K_{\text{rot}} \approx 160-200\text{ N}\cdot\text{m/rad}$ to lock the flat orientation against the inclined surface.

---

### Recommended Solutions to Fix and Boost Both Runs

#### 1. Fix the Wipe Profile ([`configs/wipe_impedance_profile_v2.yaml`](file:///home/cjimenez/projects/HiRes-VIC/configs/wipe_impedance_profile_v2.yaml))

We should update the phases and stiffness values to match the physics of the $45^\circ$ tilted board:

- **`approach`**:
  - `kp_trans: [80.0, 80.0, 80.0]` (stable free-space motion toward table)
  - `kp_rot:   [140.0, 140.0, 140.0]` (locks $+45^\circ$ pitch parallel to board)
- **`wipe`**:
  - `kp_trans: [120.0, 150.0, 120.0]` (provides the $\sim 120\text{ N/m}$ normal stiffness needed for $15\text{ N}$ force tracking)
  - `kp_rot:   [180.0, 180.0, 180.0]` (keeps the wiper pad flat against the board during sweeping)

#### 2. Simplify the State-to-Phase Mapping in the Prompt

Eliminate the ambiguous `transition` phase so the LLM has a crisp, unambiguous binary trigger:
- `In contact: False` $\to$ **`approach`**
- `In contact: True` $\to$ **`wipe`**

#### 3. Log-Space Exploration Span for SPD

With a nominal prior of $K_{\text{prior}} \approx 120\text{ N/m}$ ($S = \ln(120) \approx 4.79$), the Log-Euclidean exploration span becomes:
$$S_{\text{total}} = 4.79 \pm 1.25 \implies K \in [34.5\text{ N/m},\, 300\text{ N/m}]$$
This allows the policy full authority to explore both compliant wiping ($35\text{ N/m}$) and stiff sweeping ($300\text{ N/m}$).

#### 4. Prior Annealing ($w: 0.50 \to 0.05$)

In [`scripts/run_WipeTeleportLLM.slurm`](file:///home/cjimenez/projects/HiRes-VIC/scripts/run_WipeTeleportLLM.slurm#L62-L66), enabling cosine annealing (`LLM_ANNEAL=500000`, `LLM_W_FLOOR=0.05`) ensures the LLM provides strong guidance early on to reach 80% SR fast, while fading so the RL agent can fine-tune in late training.

---

Would you like me to update [`configs/wipe_impedance_profile_v2.yaml`](file:///home/cjimenez/projects/HiRes-VIC/configs/wipe_impedance_profile_v2.yaml) with these optimized parameters?