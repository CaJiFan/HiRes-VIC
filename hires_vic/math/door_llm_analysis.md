Viewed door_impedance_profile.yaml:1-81
Viewed impedance_planner.py:450-520
Searched for "llm/avg_phase_duration"
Ran command: `ls -lt logs/door/*LLM*.log | head -n 10`
Viewed task-961.log
Searched for "LLMv1"
Viewed Door_DOOR_BASELINE_ICRA_LR3e-4_H100_LLMv1_W0.50_G0.98_HCET_SEED_0.log:100-200
Viewed impedance_planner.py:480-510

Here is the exact diagnostic of why the LLM prior is creating a **250k–350k step delay in sample efficiency** on `Door`, and what we can do to fix it:

---

### 1. The Core Root Cause: The "Limp Wrist" Latency Trap

Look at what the two methods are doing at Step 0:

#### A. Pure Baseline ($W=0.0$, Green Curve — Takes off at 100k):
- Under default Gaussian initialization ($a \sim \mathcal{U}(-1, 1)$), the expected stiffness is:
  $$\mathbb{E}[K_{\text{trans}}] = \frac{1 + 300}{2} \approx \mathbf{150\text{ N/m}}, \quad \mathbb{E}[K_{\text{rot}}] \approx \mathbf{150\text{ N}\cdot\text{m/rad}}$$
- With $150\text{ N}\cdot\text{m/rad}$ stiffness, the Panda arm has the physical rigidity to **overpower the MuJoCo spring-loaded handle latch** and turn it immediately.

#### B. LLM Prior ($W=0.5 - 0.8$, Magenta/Blue/Purple — Delayed to 350k–500k):
- Look at [`configs/door_impedance_profile.yaml`](file:///home/cjimenez/projects/HiRes-VIC/configs/door_impedance_profile.yaml#L29-L30):
  $$\text{grasp\_turn}: \quad K_{\text{trans}} = [80, 60, 50]\text{ N/m}, \quad K_{\text{rot}} = [15, 15, 35]\text{ N}\cdot\text{m/rad}$$
- In MuJoCo, the door latch has a return spring. At $K_{\text{rot}} = 15-20\text{ N}\cdot\text{m/rad}$, the robot's wrist is **too soft / compliant**; when the gripper touches the handle, the handle's spring easily pushes the wrist back!
- The SAC policy has to spend **300,000 steps learning massive positive residual actions ($a \approx +1.0$)** just to fight against the soft prior and boost stiffness back up to $\sim 150\text{ N/m}$.

#### The Proof from your Plot:
The higher the weight $W$, the less authority the RL agent has ($(1-W)$), and the longer it is trapped fighting the low stiffness:
- **$W = 0.0$**: Takes off at **100k**
- **$W = 0.5$**: Takes off at **350k**
- **$W = 0.8$**: Takes off at **500k**

---

### 2. The Information Gap: The LLM is Blind to Latch Angles

In [`hires_vic/llm/impedance_planner.py:492-502`](file:///home/cjimenez/projects/HiRes-VIC/hires_vic/llm/impedance_planner.py#L492-L502), the Door state description passed to the LLM only includes EEF position and Door position.

It was **missing**:
- **`handle_qpos`** (latch rotation angle)
- **`hinge_qpos`** (door opening angle)
- **`handle_to_eef_pos`** (exact distance from gripper to handle)

Because the LLM never receives `handle_qpos`, it cannot detect when the latch is turned, so it gets stuck in `grasp_turn` ($K=80$) instead of transitioning to `pull_open` ($K_x = 150$)!

---

### 3. The 3 Recommended Tweaks

#### Step 1: Add Latch & Hinge Observables to the LLM State Description
In [`hires_vic/llm/impedance_planner.py`](file:///home/cjimenez/projects/HiRes-VIC/hires_vic/llm/impedance_planner.py#L492-L502):
```python
if "handle_to_eef_pos" in obs_dict:
    he = np.array(obs_dict.get("handle_to_eef_pos", np.zeros(3)))
    lines.append(f"EEF to handle distance (xyz): [{he[0]:.3f}, {he[1]:.3f}, {he[2]:.3f}] (dist: {np.linalg.norm(he):.4f} m)")
if "handle_qpos" in obs_dict:
    hq = float(np.asarray(obs_dict["handle_qpos"]).flatten()[0])
    lines.append(f"Handle rotated angle: {hq:.3f} rad (latch unlatched if > 0.3 rad)")
if "hinge_qpos" in obs_dict:
    dq = float(np.asarray(obs_dict["hinge_qpos"]).flatten()[0])
    lines.append(f"Door open hinge angle: {dq:.3f} rad (door opened if > 0.3 rad)")
```
*This allows the LLM to transition to `pull_open` the instant the latch is rotated.*

#### Step 2: Ground the Profile Stiffness to Overpower the MuJoCo Spring
In [`configs/door_impedance_profile.yaml`](file:///home/cjimenez/projects/HiRes-VIC/configs/door_impedance_profile.yaml), the stiffness values for `grasp_turn` and `pull_open` should reflect the physical force needed for the door mechanism:
- `grasp_turn`: $K_{\text{trans}} = [120, 100, 100]\text{ N/m}$, $K_{\text{rot}} = [80, 80, 100]\text{ N}\cdot\text{m/rad}$ (sufficient grip to turn the latch).
- `pull_open`: $K_{\text{trans}} = [200, 40, 80]\text{ N/m}$, $K_{\text{rot}} = [60, 40, 40]\text{ N}\cdot\text{m/rad}$ (strong outward pull + compliant arc).

#### Step 3: Enable Weight Annealing ($W: 0.6 \to 0.05$)
In `scripts/run_Door.slurm`, setting `LLM_ANNEAL=$((STEPS * 50 / 100 / N_ENVS))` allows the LLM to guide early exploration during the first 250k steps, and then smoothly hands full residual authority to the RL policy.

---

### Would you like me to update `impedance_planner.py` and the Door profile with these fixes?