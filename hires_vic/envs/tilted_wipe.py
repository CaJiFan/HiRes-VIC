from robosuite.environments.manipulation.wipe import Wipe
import numpy as np

# 1. Define the custom environment
class TiltedWipe(Wipe):
    """
    Custom Wipe environment with a tilted table.
    """
    def __init__(self, tilt_angle_degrees=15.0, **kwargs):
        # Convert to radians and round it nicely for the XML
        print(f"Initializing TiltedWipe with a tilt angle of {tilt_angle_degrees} degrees.")
        self.tilt_angle_rad = np.radians(tilt_angle_degrees)
        super().__init__(**kwargs)

    def _load_model(self):
        # 1. Let the original Wipe environment load everything
        super()._load_model()
        
        # 2. Grab the table body through the master 'model' attribute
        table = self.model.mujoco_arena.table_body
        
        # 3. Apply the tilt (pitch rotation around the Y axis)
        table.set("euler", f"0 {self.tilt_angle_rad:.4f} 0")


