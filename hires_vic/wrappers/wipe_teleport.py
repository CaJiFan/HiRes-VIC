import gymnasium as gym
import numpy as np
from scipy.spatial.transform import Rotation as R
import mujoco


class WipeTeleportWrapper(gym.Wrapper):
    """
    Teleports the robot EEF to ~30cm above the centre of the tilted table
    at the start of each episode, then immediately hands control to the RL agent.

    This skips the unproductive free-space approach phase so the agent can
    focus on the contact transition and compliance learning.

    Implementation note
    -------------------
    The teleport is done by directly overwriting the robot's joint qpos in
    MuJoCo data and calling sim.forward(). No scripted steps are run.

    Hover target computation (in order of preference):
      1. Sim-based: read the table body centroid from sim.data.body_xpos and
         compute the hover point along the surface normal of the tilted table.
      2. Config fallback: derive the hover point from task_config constants
         (table_offset + table_full_size) with the same tilt geometry.

    Args
    ----
    env              : wrapped Robosuite/GymWrapper env
    tilt_angle_deg   : table tilt angle in degrees (should match TiltedWipe's
                       tilt_angle_degrees kwarg, default 45.0)
    hover_dist       : how far above the table surface to place the EEF (metres)
    table_body_name  : MuJoCo body name for the table (default 'table_body')
    """

    def __init__(
        self,
        env,
        tilt_angle_deg: float = 45.0,
        hover_dist: float = 0.30,
        table_body_name: str = "table",
    ):
        super().__init__(env)
        self.tilt_angle_deg = tilt_angle_deg
        self.hover_dist = hover_dist
        self.table_body_name = table_body_name

    # ------------------------------------------------------------------
    # Internal helpers
    # ------------------------------------------------------------------

    def _get_sim(self):
        """Walk the wrapper stack to find the MuJoCo sim object."""
        return self.env.unwrapped.sim

    def _compute_hover_target(self, sim) -> np.ndarray:
        """
        Return a world-frame 3-D hover target ~hover_dist above the
        tilted table's upper surface centre.

        Strategy
        --------
        1. Try to read the table body centroid directly from the sim.
        2. Fall back to the hard-coded task_config geometry.
        """
        tilt_rad = np.radians(self.tilt_angle_deg)

        # Surface normal for a table pitched around the Y axis by tilt_rad:
        # normal = R_y(tilt_rad) @ [0, 0, 1] = [sin(tilt), 0, cos(tilt)]
        normal = np.array([np.sin(tilt_rad), 0.0, np.cos(tilt_rad)])

        # --- 1. Sim-based: read table body centroid --------------------
        try:
            body_id = sim.model.body_name2id(self.table_body_name)
            table_centre = np.array(sim.data.body_xpos[body_id], dtype=float)
            # The table centroid is at half-thickness below the surface;
            # add half the table thickness along the (untilted) Z to reach surface.
            # We keep it simple and trust hover_dist to cover the slack.
            hover_target = table_centre + self.hover_dist * normal
            print(
                f"[WipeTeleport] Sim-based hover target: {hover_target} "
                f"(table centre: {table_centre})"
            )
            return hover_target
        except Exception as e:
            print(f"[WipeTeleport] Sim-based table lookup failed ({e}), using config fallback.")

        # --- 2. Config fallback: use known table_offset from task_config -
        # DEFAULT: table_offset=[0.15, 0, 0.9], table_full_size=[0.5, 0.8, 0.05]
        # Surface centre z = 0.9 + 0.025 = 0.925
        table_centre_fallback = np.array([0.15, 0.0, 0.925])
        hover_target = table_centre_fallback + self.hover_dist * normal
        print(f"[WipeTeleport] Config-fallback hover target: {hover_target}")
        return hover_target

    def _teleport_eef_to(self, sim, target_pos: np.ndarray) -> bool:
        """
        Instantaneously move the robot so that its EEF is at target_pos by
        solving a simple IK approximation: nudge joint qpos using the Jacobian
        pseudo-inverse, then zero out all joint velocities.

        For Robosuite / MuJoCo, the most reliable approach without a full IK
        solver is to directly set the arm joint positions to a neutral/home
        configuration that is known to place the EEF at roughly the right
        height, then let the agent refine from there.

        We use a two-step approach:
          a) Read the current joint qpos.
          b) Apply a Jacobian-based delta (one step) to move toward target_pos.
          c) Zero velocities and call sim.forward().

        Returns True on success, False on failure (env will start from default
        reset pose which is still a valid—if unoptimised—starting point).
        """
        try:
            robot = sim.model.robot_name  # e.g. 'robot0'
        except Exception:
            robot = "robot0"

        # Get current EEF position from observations
        try:
            raw_obs = self.env.unwrapped._get_observations()
            current_eef = np.array(raw_obs["robot0_eef_pos"], dtype=float)
        except Exception as e:
            print(f"[WipeTeleport] Could not read EEF pos: {e}. Skipping teleport.")
            return False

        # Get joint addresses for the arm (7 DOF Panda)
        try:
            # Panda joint names in Robosuite
            joint_names = [f"robot0_joint{i}" for i in range(1, 8)]
            qpos_addrs = [sim.model.get_joint_qpos_addr(j) for j in joint_names]
            # Flatten: each free joint returns a range; revolute joints return (idx, idx+1)
            qpos_indices = []
            for addr in qpos_addrs:
                if isinstance(addr, (list, tuple, range)):
                    qpos_indices.extend(range(addr[0], addr[1]))
                else:
                    qpos_indices.append(int(addr))
        except Exception as e:
            print(f"[WipeTeleport] Joint address lookup failed: {e}. Skipping teleport.")
            return False

        # Jacobian-based one-step IK
        try:
            # Get Jacobian: shape (3, n_dof) for translational part
            # MuJoCo sim.data.get_body_jacp returns a flat (3*nv,) array
            nv = sim.model.nv
            jacp = np.zeros((3, nv))
            jacr = np.zeros((3, nv))

            # Find EEF body id (Robosuite names it robot0_right_hand or similar)
            eef_body_candidates = [
                "robot0_right_hand", "robot0_eef", "right_hand", "eef"
            ]
            eef_body_id = None
            for name in eef_body_candidates:
                try:
                    eef_body_id = sim.model.body_name2id(name)
                    break
                except Exception:
                    continue

            if eef_body_id is None:
                # Last resort: scan all bodies for one containing 'hand'
                for i in range(sim.model.nbody):
                    bname = sim.model.body_id2name(i)
                    if "hand" in bname.lower() and "robot" in bname.lower():
                        eef_body_id = i
                        break

            try:
                # Newer mujoco versions (e.g. >= 3)
                mujoco.mj_jacBody(sim.model._model, sim.data._data, jacp, jacr, eef_body_id)
            except AttributeError:
                try:
                    # Older mujoco versions (e.g. 2.x via mujoco_py)
                    sim.data.get_body_jacp(sim.model._model, eef_body_id, jacp)
                    sim.data.get_body_jacr(sim.model._model, eef_body_id, jacr)
                except TypeError:
                    sim.data.get_body_jacp(eef_body_id, jacp.ravel())
                    sim.data.get_body_jacr(eef_body_id, jacr.ravel())

            # Extract columns corresponding to our arm joints
            J = jacp[:, qpos_indices]  # (3, 7)

            delta_pos = target_pos - current_eef
            # Damped least-squares pseudo-inverse
            damping = 0.05
            JJT = J @ J.T + damping ** 2 * np.eye(3)
            dq = J.T @ np.linalg.solve(JJT, delta_pos)

            # Clamp the delta to avoid huge jumps (max 0.5 rad per joint)
            dq = np.clip(dq, -0.5, 0.5)

            # Apply delta to qpos
            current_qpos = np.array(sim.data.qpos[qpos_indices])
            new_qpos = current_qpos + dq

            sim.data.qpos[qpos_indices] = new_qpos

        except Exception as e:
            print(f"[WipeTeleport] Jacobian IK failed: {e}. Skipping teleport.")
            return False

        # Zero all joint velocities for a clean start
        try:
            qvel_addrs = [sim.model.get_joint_qvel_addr(j) for j in joint_names]
            qvel_indices = []
            for addr in qvel_addrs:
                if isinstance(addr, (list, tuple, range)):
                    qvel_indices.extend(range(addr[0], addr[1]))
                else:
                    qvel_indices.append(int(addr))
            sim.data.qvel[qvel_indices] = 0.0
        except Exception as e:
            print(f"[WipeTeleport] Could not zero velocities: {e}")

        sim.forward()
        return True

    # ------------------------------------------------------------------
    # Gym API
    # ------------------------------------------------------------------

    def reset(self, **kwargs):
        obs, info = self.env.reset(**kwargs)

        sim = self._get_sim()
        hover_target = self._compute_hover_target(sim)
        success = self._teleport_eef_to(sim, hover_target)

        if success:
            # Re-read obs after teleport so the agent starts from the new pose
            try:
                raw_obs = self.env.unwrapped._get_observations()
                # The GymWrapper flattens obs; replicate that here
                obs = self.env._flatten_obs(raw_obs)
            except Exception:
                # If re-reading fails, the obs from before teleport is still
                # a valid starting point — agent will self-correct immediately.
                pass

        return obs, info


# ---------------------------------------------------------------------------


class WipeDomainRandomizationWrapper(gym.Wrapper):
    """
    Per-episode Domain Randomization for the TiltedWipe task.

    Randomizes at each reset():
      - Table tilt angle (pitch around Y axis): Uniform(tilt_min, tilt_max)
      - Effective table size (scale applied to X and Y): Uniform(size_scale_min, 1.0)
      - Table friction (if enabled): log-Uniform around the nominal value

    The tilt is applied by directly overwriting the table body's quaternion in
    sim.data.body_xquat and calling sim.forward(). Table geometry (half-extents)
    is patched via sim.model.geom_size for the table geom(s).

    Recommended usage
    -----------------
    Wrap AFTER WipeTeleportWrapper so the hover target is computed with the
    already-randomized tilt:

        env = GymWrapper(suite.make(...))
        env = WipeDomainRandomizationWrapper(env, ...)
        env = WipeTeleportWrapper(env, tilt_angle_deg=45.0)
        env = GeometricWrapper(env, ...)

    Note: WipeTeleportWrapper accepts `tilt_angle_deg` which should match the
    *nominal* angle; the actual tilt each episode is determined by this wrapper.
    For correct hover computation, pass tilt_angle_deg=45.0 and rely on the
    sim-based lookup (which reads the body's actual post-DR quaternion).

    Args
    ----
    env              : wrapped env
    tilt_min_deg     : minimum tilt angle (degrees), default 38
    tilt_max_deg     : maximum tilt angle (degrees), default 52
    size_scale_min   : minimum scale factor for table X and Y, default 0.7
                       (1.0 = no size change, 0.7 = 30% smaller → matches
                       a real whiteboard smaller than the sim default)
    randomize_friction : if True, also randomize table friction ±50%
    table_body_name  : MuJoCo body name for the table
    table_geom_name  : MuJoCo geom name for the table top surface
    """

    def __init__(
        self,
        env,
        tilt_min_deg: float = 38.0,
        tilt_max_deg: float = 52.0,
        size_scale_min: float = 0.7,
        randomize_friction: bool = False,
        table_body_name: str = "table",
        table_geom_name: str = "table_collision",
    ):
        super().__init__(env)
        self.tilt_min_deg = tilt_min_deg
        self.tilt_max_deg = tilt_max_deg
        self.size_scale_min = size_scale_min
        self.randomize_friction = randomize_friction
        self.table_body_name = table_body_name
        self.table_geom_name = table_geom_name

        # Store original geom sizes to scale relative to nominal each episode
        self._nominal_geom_sizes: dict = {}
        self._nominal_frictions: dict = {}
        self._dr_initialised = False

    # ------------------------------------------------------------------

    def _get_sim(self):
        return self.env.unwrapped.sim

    def _initialise_nominal_values(self, sim):
        """Cache the original geom sizes/frictions on the first reset."""
        if self._dr_initialised:
            return
        # Find all geoms whose name contains the table geom name
        for i in range(sim.model.ngeom):
            gname = sim.model.geom_id2name(i)
            if gname and (
                self.table_geom_name in gname or self.table_body_name in gname.lower()
            ):
                self._nominal_geom_sizes[i] = sim.model.geom_size[i].copy()
                self._nominal_frictions[i] = sim.model.geom_friction[i].copy()
        self._dr_initialised = True

    def _apply_tilt(self, sim, tilt_deg: float):
        """Overwrite the table body orientation with the new tilt angle."""
        try:
            body_id = sim.model.body_name2id(self.table_body_name)
            tilt_rad = np.radians(tilt_deg)
            # Pitch around Y axis: R_y(tilt_rad)
            r = R.from_euler("y", tilt_rad)
            # MuJoCo uses [w, x, y, z]
            q = r.as_quat()  # [x, y, z, w]
            sim.model.body_quat[body_id] = np.array([q[3], q[0], q[1], q[2]])
        except Exception as e:
            print(f"[WipeDR] Tilt randomization failed: {e}")

    def _apply_size_scale(self, sim, scale: float):
        """Scale table geom X and Y half-extents by `scale`."""
        for geom_id, nominal_size in self._nominal_geom_sizes.items():
            new_size = nominal_size.copy()
            new_size[0] *= scale  # X half-extent
            new_size[1] *= scale  # Y half-extent
            sim.model.geom_size[geom_id] = new_size

    def _apply_friction(self, sim, friction_scale: float):
        """Multiply table geom friction by `friction_scale`."""
        for geom_id, nominal_fric in self._nominal_frictions.items():
            sim.model.geom_friction[geom_id] = nominal_fric * friction_scale

    # ------------------------------------------------------------------

    def reset(self, **kwargs):
        obs, info = self.env.reset(**kwargs)

        sim = self._get_sim()
        self._initialise_nominal_values(sim)

        # --- Sample randomization parameters ---------------------------
        tilt_deg = float(np.random.uniform(self.tilt_min_deg, self.tilt_max_deg))
        size_scale = float(np.random.uniform(self.size_scale_min, 1.0))

        # --- Apply -------------------------------------------------------
        self._apply_tilt(sim, tilt_deg)
        self._apply_size_scale(sim, size_scale)

        if self.randomize_friction:
            friction_scale = float(np.random.uniform(0.5, 1.5))
            self._apply_friction(sim, friction_scale)
        else:
            friction_scale = 1.0

        sim.forward()

        print(
            f"[WipeDR] Episode DR: tilt={tilt_deg:.1f}°, "
            f"size_scale={size_scale:.2f}, "
            f"friction_scale={friction_scale:.2f}"
        )

        info["dr/tilt_deg"] = tilt_deg
        info["dr/size_scale"] = size_scale
        info["dr/friction_scale"] = friction_scale

        return obs, info