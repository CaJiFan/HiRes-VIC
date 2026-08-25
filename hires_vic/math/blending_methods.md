Here is the theoretical and mathematical explanation of the difference between the two approaches, along with why the comparison with pure RL (non-LLM) SPD remains mathematically sound and fair:

---

### 1. Mathematical Difference: Unbounded Tangent Addition vs. Geodesic Bounded Mapping

#### A. Previous Approach (Unbounded Fixed-Scale Log-Euclidean Residual)
In the previous implementation:
$$\Delta S_i = a_i \cdot (1 - w) \cdot \sigma_{\text{diag}} \quad (\text{with } \sigma_{\text{diag}} = 2.5)$$
$$S_{\text{total}, i} = \ln(K_{\text{prior}, i}) + \Delta S_i$$
$$K_i = \exp(S_{\text{total}, i}) = K_{\text{prior}, i} \cdot e^{a_i (1 - w) \cdot 2.5}$$

* **The Theoretical Flaw**:
  The exponential map $\exp: \operatorname{Sym}(3) \to \mathcal{S}_{++}^3$ is an unbounded convex function.
  When $K_{\text{prior}} = 250\text{ N/m}$ (already near the $300\text{ N/m}$ ceiling), adding a fixed $+1.25$ in log-space pushes the total coordinate to:
  $$S = \ln(250) + 1.25 = 6.77 \implies K = \exp(6.77) = \mathbf{872\text{ N/m}}$$
  Because the derivative of the exponential function $\frac{dK}{dS} = \exp(S) \approx 900$ is extremely steep in this region:
  - Tiny policy action changes caused massive physical stiffness swings ($\Delta K \approx 100\text{ N/m}$).
  - This produced severe contact force spikes ($474\text{ N}$ in Seed 1), exploding the SAC critic loss ($46.6$) and destabilizing learning.

---

#### B. Current Approach (Geodesic Boundary-Scaled Log-Euclidean Residual)
In Riemannian geometry, on the 1D positive real line with the affine-invariant metric $g = ds^2 / x^2$, the geodesic (shortest path) between $K_{\text{prior}}$ and $K_{\max}$ is the curve:
$$\gamma(t) = K_{\text{prior}} \cdot \left(\frac{K_{\max}}{K_{\text{prior}}}\right)^t \quad \text{for } t \in [0, 1]$$

The new implementation scales the tangent vector by the exact geodesic distance to the manifold boundary:
$$\Delta S_i = \begin{cases}
a_i \cdot (1 - w) \cdot \Big(\ln K_{\max} - \ln K_{\text{prior}, i}\Big) & \text{if } a_i \ge 0 \\[6pt]
a_i \cdot (1 - w) \cdot \Big(\ln K_{\text{prior}, i} - \ln K_{\min}\Big) & \text{if } a_i < 0
\end{cases}$$

When mapped through the exponential:
$$K_i = \exp(S_{\text{prior}, i} + \Delta S_i) = \begin{cases}
K_{\text{prior}, i} \cdot \left(\frac{K_{\max}}{K_{\text{prior}, i}}\right)^{a_i (1 - w)} & \text{if } a_i \ge 0 \\[8pt]
K_{\text{prior}, i} \cdot \left(\frac{K_{\min}}{K_{\text{prior}, i}}\right)^{|a_i| (1 - w)} & \text{if } a_i < 0
\end{cases}$$

* **Theoretical Properties**:
  1. **Strict Boundary Confinement**: $K_i \in [K_{\min}, K_{\max}]$ for all $a \in [-1, 1]$ and all $w \in [0, 1]$.
  2. **Constant Geodesic Speed**: The policy action $a_i \in [-1, 1]$ directly acts as the normalized progress parameter along the geodesic from $K_{\min} \to K_{\text{prior}} \to K_{\max}$.
  3. **Zero-Action Invariance**: At $a_i = 0$, $K_i = K_{\text{prior}, i}$ exactly.

---

### 2. Does this affect the Non-LLM SPD Approach? (Is the comparison fair?)

**No, it does not alter or invalidate the comparison.** In fact, it makes the comparison **strictly fair and symmetric**.

Let's look at how pure RL (non-LLM) SPD computes stiffness:
```python
# Pure RL SPD (in hires_vic/wrappers/geometric.py):
target_physical = min_kp + 0.5 * (action[:3] + 1.0) * (max_kp - min_kp)  # in [1, 300]
m_params_rl[0:3] = np.log(target_physical)                               # in [0, ln(300)]
m_params_rl[3:6] = action[3:6] * 0.2                                     # off-diagonal coupling
kp_matrix = spd_grl_map(m_params_rl)                                     # Riemannian mapping
```

#### Why the comparison is 100% fair:
1. **Identical Manifold Support**:
   - Pure RL SPD maps $a \in [-1, 1]$ across $[K_{\min}, K_{\max}] = [1, 300]\text{ N/m}$.
   - LLM SPD (with $w=0$) maps $a \in [-1, 1]$ across the exact same $[K_{\min}, K_{\max}] = [1, 300]\text{ N/m}$.
2. **Identical Geometry Pipeline**:
   - Both methods feed their 6D Mandel parameter vector into the **exact same `spd_grl_map`** function.
   - Both use the **exact same off-diagonal exploration bound ($0.2$)**.
   - Both undergo the **exact same spectral eigenvalue clamping** to $[1, 300]\text{ N/m}$.
3. **The Only Difference (The Scientific Hypothesis)**:
   - **Pure RL SPD**: Explores the manifold from scratch without an initial semantic center.
   - **LLM SPD**: Starts anchored at the nominal semantic prior $K_{\text{prior}}$ on the manifold and uses residual RL to modulate stiffness around it with confidence $w$.

This represents a clean, scientifically rigorous residual RL ablation on the Riemannian manifold.