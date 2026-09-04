Here are the exact values based on the `robosuite` simulation model ([wiping_gripper.xml](file:///home/cjimenez/.conda/envs/tfm/lib/python3.12/site-packages/robosuite/models/assets/grippers/wiping_gripper.xml) and [robot.xml](file:///home/cjimenez/.conda/envs/tfm/lib/python3.12/site-packages/robosuite/models/assets/robots/panda/robot.xml)), along with important real-hardware considerations.

---

### 1. Exact Parameters (Simulation Match)

| Parameter | Value in Franka Desk (SI Units) | Value in (mm) |
| :--- | :--- | :--- |
| **Mass ($m$)** | `0.030 kg` | `30 g` |
| **Flange to Center of Mass ($x, y, z$)** | `[0.0, 0.0, 0.015] m` | `[0, 0, 15] mm` |
| **Flange to TCP Translation ($x, y, z$)** | `[0.0, 0.0, 0.030] m` | `[0, 0, 30] mm` |
| **Flange to TCP Orientation** | $\text{Yaw} = -90^\circ$ (or $-1.5708\text{ rad}$) | $R_z(-90^\circ)$ |

#### **Inertia Tensor Matrix ($\mathbf{I}_{\text{com}}$ at the Center of Mass)**
$$\mathbf{I}_{\text{com}} = \begin{bmatrix} 
0.01 & 0.00 & 0.00 \\ 
0.00 & 0.01 & 0.00 \\ 
0.00 & 0.00 & 0.01 
\end{bmatrix} \text{ kg}\cdot\text{m}^2$$

---

### 2. Breakdown of Each Field

1. **Mass ($m = 0.030\text{ kg}$)**:
   * Defined directly as `mass="3e-2"` in `wiping_gripper.xml`.

2. **Flange to Center of Mass ($\mathbf{r}_{\text{CoM}} = [0, 0, 0.015]\text{ m}$)**:
   * The wiping block spans $30\text{ mm}$ in height. Its geometric origin / CoM is located at the center of the block, which is $+15\text{ mm}$ along the $Z$-axis from the Panda flange surface.

3. **Flange to TCP ($\mathbf{r}_{\text{TCP}} = [0, 0, 0.030]\text{ m}, \text{Yaw} = -90^\circ$)**:
   * **Translation ($Z = 30\text{ mm}$)**: The contact pad (bottom face where wiping occurs) is located $15\text{ mm}$ below the CoM, giving a total distance of $15\text{ mm} + 15\text{ mm} = 30\text{ mm}$ from the flange.
   * **Rotation ($\text{quat} = [0.707107, 0, 0, -0.707107]$)**: Represents a $-90^\circ$ rotation about the flange $Z$-axis.

4. **Inertia Tensor ($\mathbf{I}_{\text{diag}} = \text{diag}(0.01, 0.01, 0.01)\text{ kg}\cdot\text{m}^2$)**:
   * Defined directly as `diaginertia="1e-2 1e-2 1e-2"` with all off-diagonal cross terms ($I_{xy}, I_{xz}, I_{yz}$) set to `0.0`.

---

### 3. Important Real Franka Robot Considerations

> [!WARNING]
> **Gravity Compensation & Franka Safety Reflexes on Real Hardware:**
> * Franka's internal controller (FCI / Desk) uses the configured **Mass**, **CoM**, and **Inertia** for real-time gravity compensation and dynamic model feedforward ($\mathbf{M}(\mathbf{q})\ddot{\mathbf{q}} + \mathbf{c} + \mathbf{g}$).
> * If you attach a **physical 3D-printed bracket + real eraser** (which typically weighs between $150\text{ g}$ and $400\text{ g}$ with screws), entering the simulation's $30\text{ g}$ value will cause the Franka to miscalculate gravity compensation, leading to false external torque triggers (`communication_constraints_violation` or reflex stops).
> * **Recommendation for Real Hardware**:
>   - **TCP ($x, y, z, \text{rot}$)**: Keep the exact geometric offset ($Z = 30\text{ mm}$ or your measured tool length, $\text{Yaw} = -90^\circ$) to preserve kinematic equivalence.
>   - **Mass & CoM**: Weigh your actual assembled tool on a precision scale and set the true physical mass and measured center of mass so that Franka's F/T estimator and gravity compensation remain accurate.