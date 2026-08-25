Yes! Beyond the YAML profile, there are **4 mathematical aspects and hyperparameters** of the Log-Euclidean residual mapping that we can examine, tune, or ablate in future iterations:

---

### 1. Tangent Space Exploration Span ($\sigma_{\text{diag}}$)

In our SPD mapping, the diagonal residual is:
$$\Delta S_{\text{diag}} = a_{0:3} \cdot (1 - w) \cdot \sigma_{\text{diag}}$$

Currently, $\sigma_{\text{diag}} = 2.5$.
- With $w = 0.50$, the maximum tangent displacement is $\pm 1.25$.
- In physical stiffness, an exploration of $\Delta S = \pm 1.25$ represents a multiplicative span of:
  $$K \in [K_{\text{prior}} \cdot e^{-1.25},\, K_{\text{prior}} \cdot e^{+1.25}] = [0.28 \cdot K_{\text{prior}},\, 3.5 \cdot K_{\text{prior}}]$$
- For a prior $K_{\text{prior}} = 150\text{ N/m}$, this gives $K \in [42\text{ N/m},\, 300\text{ N/m}]$.
- **Potential tuning**:
  If a task requires dipping into ultra-soft compliance (e.g. $<20\text{ N/m}$), increasing $\sigma_{\text{diag}}$ from $2.5 \to 3.0$ or $3.5$ expands the reachable dynamic range to $[0.17 \cdot K_{\text{prior}},\, 5.8 \cdot K_{\text{prior}}]$ without breaking positive definiteness.

---

### 2. Off-Diagonal Mandel Coupling Authority ($\sigma_{\text{off}}$)

Currently, the Mandel off-diagonal terms explore with:
$$\Delta S_{\text{off}} = a_{3:6} \cdot (1 - w) \cdot \sigma_{\text{off}} \quad (\text{with } \sigma_{\text{off}} = 0.5)$$

- On a $45^\circ$ tilted board, aligning the stiffness ellipsoid with the surface normal requires a non-zero coupling:
  $$K_{xz} = \frac{K_n - K_t}{2} \approx \frac{150 - 250}{2} = -50\text{ N/m}$$
- **Potential tuning**:
  If the agent needs stronger authority to rotate the principal axes of the stiffness ellipsoid on steep angles ($45^\circ$ or $60^\circ$ tilts), scaling $\sigma_{\text{off}}$ from $0.5 \to 0.8$ or $1.0$ gives the RL policy more geometric flexibility to introduce cross-axis compliance.

---

### 3. Log-Euclidean Metric vs. Affine-Invariant Riemannian Metric (AIRM)

In Riemannian robotics literature, there are two ways to apply a residual around a prior point $K_{\text{prior}} \in \mathcal{S}_{++}^3$:

1. **Log-Euclidean Tangent Addition (Arsigny et al., 2006)** *(What we implemented)*:
   $$S_{\text{total}} = \log(K_{\text{prior}}) + \Delta S, \quad K = \exp(S_{\text{total}})$$
   - *Pros*: Commutative, forms a vector space on $\operatorname{Sym}(3)$, compute-efficient (single matrix exponential, no matrix square roots), and preserves scale-invariance.
2. **Affine-Invariant Exponential Map from Base Point (Jaquier et al., 2020)**:
   $$K = K_{\text{prior}}^{1/2} \exp(K_{\text{prior}}^{-1/2} \Delta X K_{\text{prior}}^{-1/2}) K_{\text{prior}}^{1/2}$$
   - *Pros*: Geometrically canonical for the invariant metric.
   - *Cons*: Requires two matrix square roots and multiple matrix multiplications per step (more expensive in a $20\text{ Hz}$ control loop).
   - Log-Euclidean is standard in residual RL for its numerical stability and speed.

---

### 4. Baseline Linear RPL vs. Logarithmic Baseline

In our Baseline, we use standard additive Euclidean RPL (Johannink et al., 2019):
$$K_{\text{baseline}} = \operatorname{clip}(K_{\text{prior}} + (1 - w) \Delta K_{\max} a,\, 1,\, 300)$$

- **Why this is fair**:
  At $a = 0$, both Baseline and SPD evaluate to **exactly $K_{\text{prior}}$** ($0$ error).
  The baseline explores in linear Euclidean space ($\Delta K$ in $\text{N/m}$), while SPD explores in Riemannian tangent space ($\Delta S$ on $\operatorname{Sym}(3)$). This provides a scientifically clean ablation between Euclidean and Riemannian residual policy learning.

---

### Summary for the Upcoming Runs

The current mathematical formulation is theoretically sound and clean. Once these new training runs finish, we can inspect whether the agent utilizes the full tangent space and whether tuning $\sigma_{\text{diag}}$ or $\sigma_{\text{off}}$ is warranted for even faster convergence.