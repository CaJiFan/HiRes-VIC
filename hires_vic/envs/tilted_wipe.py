from robosuite.environments.manipulation.wipe import Wipe
from robosuite.models.tasks import ManipulationTask
from .scattered_wipe_arena import ScatteredWipeArena
import numpy as np

class TiltedWipe(Wipe):
    """
    Custom Wipe environment with a tilted table and scattered markers.
    """
    def __init__(self, tilt_angle_degrees=45.0, **kwargs):
        print(f"Initializing TiltedWipe with a tilt angle of {tilt_angle_degrees} degrees.")
        self.tilt_angle_rad = np.radians(tilt_angle_degrees)
        super().__init__(**kwargs)

    def _load_model(self):
        # Call the parent of Wipe (_load_model in ManipulationEnv)
        super(Wipe, self)._load_model()
        
        # Adjust base pose accordingly
        xpos = self.robots[0].robot_model.base_xpos_offset["table"](self.table_full_size[0])
        self.robots[0].robot_model.set_base_xpos(xpos)

        # Get robot's contact geoms
        self.robot_contact_geoms = self.robots[0].robot_model.contact_geoms
        if self.delta_height is None:
            self.delta_height = self.rng.normal(self.table_height, self.table_height_std)
        self.table_offset[2] += self.delta_height
        
        # Use ScatteredWipeArena instead of WipeArena
        mujoco_arena = ScatteredWipeArena(
            table_full_size=self.table_full_size,
            table_friction=self.table_friction,
            table_offset=self.table_offset,
            table_friction_std=self.table_friction_std,
            coverage_factor=self.coverage_factor,
            num_markers=self.num_markers,
            line_width=self.line_width,
            two_clusters=self.two_clusters,
            rng=self.rng,
        )

        # Arena always gets set to zero origin
        mujoco_arena.set_origin([0, 0, 0])

        # task includes arena, robot, and objects of interest
        self.model = ManipulationTask(
            mujoco_arena=mujoco_arena,
            mujoco_robots=[robot.robot_model for robot in self.robots],
        )
        
        # Apply the tilt (pitch rotation around the Y axis) to table body
        table = self.model.mujoco_arena.table_body
        table.set("euler", f"0 {self.tilt_angle_rad:.4f} 0")

    def _get_active_markers(self, c_geoms):
        """
        Marker wiping detection for tilted tables.
        Uses Robosuite native corner projection, with strict gripper tool contact fallback
        (3.5 cm radius ONLY when tool face is physically touching table).
        """
        active_markers = super()._get_active_markers(c_geoms)
        
        # Strict fallback for tilted surfaces: ONLY trigger if tool face is physically touching table
        # AND marker is within 3.5 cm of tool center (prevents false hovering wipes!)
        if self._has_gripper_contact:
            try:
                site_id = self.sim.model.site_name2id("gripper0_right_grip_site")
                eef_pos = self.sim.data.site_xpos[site_id]
                for marker in self.model.mujoco_arena.markers:
                    if marker in self.wiped_markers or marker in active_markers:
                        continue
                    bid = self.sim.model.body_name2id(marker.root_body)
                    m_pos = self.sim.data.body_xpos[bid]
                    if np.linalg.norm(eef_pos - m_pos) < 0.035:  # Strict 3.5 cm wiping radius on table
                        active_markers.append(marker)
            except Exception:
                pass
                
        return active_markers

    def check_contact(self, geoms_1, geoms_2=None):
        """
        Collision check for TiltedWipe:
        Excludes wiper/tool contact geoms from arm collision checks
        unless geoms_2 is explicitly specified (e.g. checking contact with table).
        """
        if geoms_2 is None and hasattr(self, 'robots') and len(self.robots) > 0 and hasattr(self.robots[0], 'gripper'):
            tool_geoms = set(self.robots[0].gripper['right'].contact_geoms)
            if isinstance(geoms_1, list):
                arm_geoms = [g for g in geoms_1 if g not in tool_geoms]
                for g in arm_geoms:
                    if super().check_contact(g, geoms_2):
                        return True
                return False
            elif hasattr(geoms_1, 'contact_geoms'):
                arm_geoms = [g for g in geoms_1.contact_geoms if g not in tool_geoms]
                for g in arm_geoms:
                    if super().check_contact(g, geoms_2):
                        return True
                return False
        return super().check_contact(geoms_1, geoms_2)



