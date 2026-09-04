# HiRes-VIC Project Guidelines & Architecture Reference

## 1. Project Purpose & Core Philosophy
**HiRes-VIC** aims to advance Reinforcement Learning for Variable Impedance Control (VIC) for contact-rich robotic manipulation. The core approach integrates:
1. **Geometric-Riemannian Priors**: Operating on the Symmetric Positive Definite (SPD) manifold $\mathcal{S}_{++}^3$ for full 3D Cartesian stiffness matrices (including cross-axis coupling). (Note: for this particular iteration the Lie group observation space for orientation is no longer used. Only the Riemannian metric on the SPD manifold is used for the stiffness prior.)
2. **Semantic / LLM Impedance Priors**: Leveraging LLM/VLM reasoning to provide structured, phase-dependent baseline impedance parameters and confidence weightings ($w$).
3. **Residual Reinforcement Learning (SAC)**: Enabling the RL agent to learn corrections and fine-grained stiffness policies around the prior on the manifold.

---

## SPD Manifold Action Space Processing

**CRITICAL RULE:** Whenever editing `geometric.py` (especially the `step()` function), you MUST ensure that SPD runs (`use_spd_manifold=True`) always learn and process the FULL 6D Mandel basis for translational stiffness, regardless of whether `use_llm_prior` is enabled or disabled. 

Specifically:
- The action space must allocate 9 dimensions for stiffness (6 for translational SPD Mandel basis, 3 for rotational stiffness).
- Both the LLM SPD path and the pure RL (no-LLM) SPD path must construct a 6D tensor for the GRL mapping (`spd_grl_map`), where `action[0:3]` maps to diagonal terms and `action[3:6]` maps to off-diagonal (coupling) terms.
- NEVER reduce the pure RL SPD path to a 3D diagonal-only learning space.

---


## 2. Numerical Stability & Manifold Guarantees
- **Eigenvalue Clamping**: Eigenvalues of $K_p \in \mathcal{S}_{++}^3$ are clamped to $[\min(K_p), \max(K_p)]$ via spectral decomposition to preserve positive-definiteness without violating torque limits.
- **Riemannian Jerk Penalty**: Scale-invariant stiffness smoothness penalty computed using the affine-invariant Riemannian metric on $\mathcal{S}_{++}^3$:
  $$\delta_R(K_{t-1}, K_t) = \|\log(K_{t-1}^{-1/2} K_t K_{t-1}^{-1/2})\|_F$$

---

## 3. Environment Architectures & Current Status

### A. TiltedWipe (`TiltedWipe`)
* **Task Objective**: Clean sequential dirt markers on a whiteboard tilted by $45^\circ$ around the Y-axis.
* **Paper Reference**: Adapted from [arXiv:2502.12599](https://arxiv.org/abs/2502.12599).
* **Kinematic Reset (`WipeTeleportWrapper`)**:
  - Teleports EEF to hover $15\text{ cm}$ directly above the board normal.
  - Applies a $+45^\circ$ positive pitch rotation ($+\theta_{\text{tilt}}$ around $Y$) so the wiper pad's flat bottom is parallel to the tilted surface ($\mathbf{z}_{\text{eef}} \cdot \hat{\mathbf{n}}_{\text{table}} = -1.0$).
* **Observation Space**: `use_condensed_obj_obs: False` providing explicit per-marker positions, binary wiped status, and relative vectors.
  - **Waypoint guidance mode (default: nearest-unwiped)**: The quality reward `r_guide` uses **nearest-unwiped-marker** mode by default (`--use_sequential_waypoints False`). In this mode the agent can always derive its target from `min(|gripper_to_markerX|)` for unwiped markers — no extra obs feature required. This is faithful to arXiv:2502.12599.
  - **Sequential mode (optional, `--use_sequential_waypoints`)**: If Y-sorted sequential guidance is enabled, the direction vector `gripper_to_active_waypoint` (3D) is **automatically appended** in `GeometricWrapper._flatten_obs`. Do NOT enable sequential mode without this vector — empirically SR collapses to ~10% because the MLP cannot solve the relational Y-sort lookup from obs alone.
* **Reward Structure**:
  - Native centroid-reaching (`distance_multiplier`) and un-gated contact (`wipe_contact_reward`) are set to `0.0` in `configs/wipe_task_config.yaml`.
  - Checkpoint-gated quality reward (`r_guide`, `r_con_q`, `r_force_q`) with Gaussian force tracking ($F_{\text{target}} = 15\text{ N}$).
  - Quality reward is scaled by `reward_scale * reward_normalization_factor` ($\approx 0.0375$) to prevent dominating ground-truth wipe events ($50.0 \to 1.875$).
* **Domain Randomization (`--use_domain_rand`)**:
  - **Pose DR (`WipeTeleportWrapper`)**: $d_{\text{hover}} \in [10, 20]\text{ cm}$, $\Delta y \in [-6, +6]\text{ cm}$, $\Delta x \in [-3, +3]\text{ cm}$, orientation jitter $\pm 5^\circ$.
  - **Environment DR (`WipeDomainRandomizationWrapper`)**: Tilt angle $\theta_{\text{tilt}} \in [38^\circ, 52^\circ]$, scale $\in [0.7, 1.0]$, friction $\pm 50\%$.
  - **Evaluation Behavior**: Evaluates on the DR distribution with a fixed random seed (`seed=42`) across checkpoints for reproducible, representative generalization benchmarking.
* **Hyperparameters**: Fixed $\gamma = 0.95$, Horizon $= 150$, Batch Size $= 1024$, LR $= 3\times 10^{-4}$.
## High-Quality Wiping Paper Reference
* **arXiv Citation**: [arXiv:2502.12599](https://arxiv.org/abs/2502.12599)
* **Purpose in TiltedWipe**: Provides the checkpoint-gated quality reward formulation:
  - Checkpoint-gated contact ($r_{\text{con\_q}}$)
  - Gaussian normal force tracking ($r_{\text{force\_q}}$ with $F_{\text{target}} = 15\text{ N}$)
  - Continuous waypoint guidance ($r_{\text{guide}}$)
  - Smooth Gaussian checkpoint gating ($I_{\text{checkpoint}}$ with $\sigma_c = 0.15\text{ m}$)

---

### B. Door (`Door`)
* **Task Objective**: Approach handle, rotate/unlatch handle, and pull door open beyond target angle.
* **Controller & Prior Profile**: Auto-selects `configs/door_impedance_profile.yaml` with phase-dependent stiffness (e.g. low radial/tangential stiffness during handle rotation, stiff normal pulling).
* **Success & Termination**: Early termination triggered on `_check_success()` with `success_bonus` to avoid post-completion drift.

### C. NutAssembly (`NutAssemblySquare` / `NutAssemblyRound`)
* **Task Objective**: Grasp nut from table, transfer to peg, align, and insert without jamming.
* **Controller & Prior Profile**: Auto-selects `configs/nutassembly_robosuite_impedance_profile.yaml` (or VLM config).
* **Anti-Jamming Mechanism**: Monitors consecutive stuck steps (`dz > -0.5mm` over >10 steps) to penalize resting on the peg lip and encourage compliant insertion.
* **Success & Termination**: Early termination triggered on peg insertion success.

---

## 4. Key Lessons Learned & Anti-Patterns (What NOT to Do)

1. **❌ Long-Horizon Gamma Annealing ($\gamma \to 0.9933$)**:
   - *Why it fails*: Accumulating dense guidance rewards over 150 steps creates a large hovering attractor ($Q_{\text{hover}} \approx 27.0$), dwarfing sparse completion rewards ($1.875$) and causing the policy to decline after $\sim 150\text{k}$ steps.
   - *Rule*: Keep $\gamma = 0.95$ fixed for dense-reward contact manipulation tasks.

2. **❌ Unscaled Wrapper Rewards**:
   - *Why it fails*: Robosuite normalizes native rewards by $\approx 0.0375$. Adding raw unscaled wrapper rewards ($5.0/\text{step}$) completely overshadows task completion.
   - *Rule*: Always scale wrapper-level dense rewards by `raw_env.reward_scale * raw_env.reward_normalization_factor`.

3. **❌ Competing Dense Reward Signals**:
   - *Why it fails*: Combining native centroid reaching with custom waypoint guidance pulls the agent in two opposing directions.
   - *Rule*: Zero-out native `distance_multiplier` when using custom waypoint guidance rewards.

4. **❌ Large Teleport Hover Distances ($> 30\text{ cm}$) or Inverted Pitch Signs**:
   - *Why it fails*: Stretching the Panda arm $>70\text{ cm}$ forward with a $-45^\circ$ wrist bend exceeds physical kinematic limits and causes IK self-collisions.
   - *Rule*: Keep hover initialization at $15\text{ cm}$ with $+45^\circ$ pitch.

5. **❌ Direct Euclidean Blending of SPD Matrices**:
   - *Why it fails*: Linear combinations of arbitrary symmetric matrices do not stay on the SPD cone, leading to negative eigenvalues and simulation instability.
   - *Rule*: Blend via tangent space logarithmic mappings (`spd_grl_map`) or bi-linear Euclidean scaling of physical eigenvalue parameters.
