

To give you a direct, honest answer: **Your current implementation is a working empirical heuristic, but it is *not* standard Residual Policy Learning (RPL), and it has a few mathematical quirks that deviate from literature.**

Here is a detailed breakdown of where your current technique comes from, how it compares to the literature, its mathematical limitations, and what a mathematically grounded Riemannian Residual RL formulation looks like.

---

### 1. What does the Literature do for Residual RL?

In reinforcement learning literature, there are two standard paradigms:

#### A. Classical Residual Policy Learning (Johannink et al., 2019 / Silver et al., 2018)
In standard Euclidean action spaces ($\mathbb{R}^n$):
$$a_t = a_{\text{prior}}(s_t) + \Delta a_{\text{RL}}(s_t)$$
- The prior $a_{\text{prior}}$ provides the nominal baseline action.
- The RL policy $\pi_\theta(s_t)$ outputs an additive correction $\Delta a$.
- At initialization (when the actor's final weights output $\approx 0$), $\Delta a \approx 0 \implies a_t \approx a_{\text{prior}}$.

#### B. Riemannian / Geometric Residual RL (on $\mathcal{S}_{++}^3$)
On a Riemannian manifold, you cannot add matrices in Euclidean space ($K \neq K_{\text{prior}} + \Delta K$) because simple addition can easily break positive-definiteness or distort the geometry.
Instead, residual addition is performed in the **tangent space** (which is an unconstrained vector space $\mathbb{R}^6$ in Mandel basis):
- Under the **Log-Euclidean Metric (LEM)**:
  $$S_t = \log(K_{\text{prior}}) + (1 - w)\Delta S_{\text{RL}}$$
  $$K_t = \operatorname{matrix\_exp}(S_t) \in \mathcal{S}_{++}^3$$
- Under the **Affine-Invariant Metric (AIRM / Riemannian Exponential)**:
  $$K_t = K_{\text{prior}}^{1/2} \exp\left( (1-w) K_{\text{prior}}^{-1/2} \Delta S_{\text{RL}} K_{\text{prior}}^{-1/2} \right) K_{\text{prior}}^{1/2}$$

In both manifold formulations:
1. $K_{\text{prior}}$ sets the anchor point on the manifold.
2. The RL agent outputs a tangent vector $\Delta S \in \operatorname{Sym}(3)$.
3. The exponential map guarantees that $K_t$ remains strictly SPD without clipping or projection artifacts.

---

### 2. How your Current Code Works (and where it deviates)

Your current code in [`geometric.py`](file:///home/cjimenez/projects/HiRes-VIC/hires_vic/wrappers/geometric.py#L570-L628) uses a custom **piecewise bi-linear warping**:

```python
a_rl = action * (1.0 - w)
a_final = np.where(a_rl > 0, 
                   a_prior + a_rl * (1.0 - a_prior), 
                   a_prior + a_rl * (a_prior + 1.0))
```

#### Why was this written?
It was designed to solve a practical problem: if you simply add $a_{\text{prior}} + a_{\text{rl}}$ in a bounded box $[-1, 1]$, you can easily overshoot $\pm 1$. This formula linearly stretches the positive exploration from $[a_{\text{prior}}, +1]$ and the negative exploration from $[-1, a_{\text{prior}}]$.

#### The 3 Mathematical Weaknesses of this approach:

1. **Gradient Discontinuity (The "Kink" at $a_{\text{RL}} = 0$):**
   The derivative of $a_{\text{final}}$ with respect to $a_{\text{rl}}$ is:
   $$\frac{\partial a_{\text{final}}}{\partial a_{\text{rl}}} = \begin{cases} 1 - a_{\text{prior}} & \text{if } a_{\text{rl}} > 0 \\ a_{\text{prior}} + 1 & \text{if } a_{\text{rl}} < 0 \end{cases}$$
   Unless $a_{\text{prior}} = 0$, the slope **jumps discontinuously** across $a_{\text{rl}} = 0$ by $\Delta = |2 a_{\text{prior}}|$. For actor-critic algorithms like SAC (which compute policy gradients $\nabla_\theta Q(s, \pi_\theta(s))$ through the action), this creates an artificial non-smooth barrier at the zero-correction point.

2. **Inconsistent Handling of Diagonals vs Off-Diagonals:**
   - **Diagonals** ($K_{xx}, K_{yy}, K_{zz}$): undergo the bi-linear Euclidean warping in $[-1, 1]$, then linear scaling to $[1, 300]$, then $\log()$, then `matrix_exp`.
   - **Off-Diagonals** ($K_{xy}, K_{xz}, K_{yz}$): bypass the bi-linear formula completely and are directly multiplied by $0.2 \cdot (1 - w)$ in tangent space.
   This means the diagonal stiffness is parameterized differently from the cross-axis coupling.

3. **Partially Observable Residual State:**
   If the LLM changes its impedance phase (e.g. from `align` to `contact`), $a_{\text{prior}}$ changes. If `add_prior_obs` is `False`, the agent's observation does not contain $a_{\text{prior}}$ or $w$, making the environment transition dynamics non-stationary from the agent's perspective.

---

### 3. What would a mathematically grounded Riemannian Residual approach look like?

The cleanest, mathematically principled formulation that unifies the LLM prior, Riemannian geometry, and Residual RL is **Log-Euclidean Residual Mapping**:

$$\begin{aligned}
S_{\text{prior}} &= \operatorname{diag}\Big(\log(K_{\text{prior}, x}),\, \log(K_{\text{prior}, y}),\, \log(K_{\text{prior}, z})\Big) \in \operatorname{Sym}(3) \\
\Delta S_{\text{RL}} &= \text{MandelBasis}\Big(\Delta v_{\text{diag}},\, \Delta v_{\text{off}}\Big) \in \operatorname{Sym}(3) \\
S_{\text{total}} &= S_{\text{prior}} + (1 - w) \Delta S_{\text{RL}} \\
K &= \operatorname{matrix\_exp}(S_{\text{total}}) \in \mathcal{S}_{++}^3
\end{aligned}$$

#### Why is this theoretically sound?
1. **Smooth everywhere**: No piecewise branches or derivative kinks; SAC gradients flow smoothly through $\operatorname{matrix\_exp}$.
2. **True Residual on the Manifold**: When the RL agent outputs $\Delta v = 0$ (at init), $S_{\text{total}} = S_{\text{prior}} \implies K = K_{\text{prior}}$ identically.
3. **Natural Scaling**: $(1 - w)$ scales the tangent vector directly. When $w=1$, the policy is pinned to $K_{\text{prior}}$; when $w \to 0$, the policy has full tangent-space exploration freedom.
4. **No Limp Initialization**: Because $S_{\text{prior}}$ is set from the physical prior (e.g. $\log(150)$ or the LLM's profile value), the arm starts at the intended stiffness from step 0.

---

### Summary & Recommendation

- **Is your current method valid?** It is a functioning engineering heuristic that prevents boundary overshooting, but it is not grounded in standard RPL literature.
- **Will reviewers notice?** Reviewers familiar with geometric control or Residual RL will likely question the piecewise `np.where(a_rl > 0, ...)` warping and the asymmetric treatment of diagonals vs off-diagonals.
- **Next Step:** Moving to a clean **Log-Euclidean tangent space addition** ($S = S_{\text{prior}} + (1-w)\Delta S$) provides full mathematical grounding, smooth policy gradients, and standard alignment with Riemannian RPL literature.


Summary of Changes:
Planner Prior Consistency (

impedance_planner.py
):

Standardized _mode_to_action_prior() to use linear normalization consistently.
Ensures exact 1-to-1 recovery of physical stiffness from profile YAMLs without logarithmic distortion.
Riemannian Log-Euclidean Residual Mapping on SPD (

geometric.py
):

Tangent Anchor: $S_{\text{prior}} = [\ln(K_{\text{prior},x}), \ln(K_{\text{prior},y}), \ln(K_{\text{prior},z}), 0, 0, 0] \in \operatorname{Sym}(3)$.
Tangent Residual: $\Delta S = (1 - w) \cdot [\sigma_{\text{diag}} \cdot a_{0:3}, \sigma_{\text{off}} \cdot a_{3:6}]^T$.
Manifold Map: $K = \operatorname{spd_grl_map}(S_{\text{prior}} + \Delta S)$ with spectral clamping to $[1, 300]\text{ N/m}$.
Euclidean Residual Policy Learning for Baseline (

geometric.py
):

Additive residual: $K = \operatorname{clip}(K_{\text{prior}} + (1 - w) \Delta K_{\max} \cdot a_{\text{RL}}, 1, 300)$.
Eliminates the previous piecewise derivative discontinuity ("kink" at $\Delta a = 0$).
Testing & Verification:

verify_residual.py: Passed unit tests for exact physical recovery, zero-action invariance ($a = 0 \implies K = K_{\text{prior}}$), and random tangent space exploration across 100 trials.
test_env_residual.py & test_wipe_residual.py: Full end-to-end multi-step rollouts on both Door and TiltedWipe environments passed for SPD and Baseline modes.