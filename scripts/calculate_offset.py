import numpy as np
from scipy.spatial.transform import Rotation as R

# From step 0 of inspect_spaces.py
# Robosuite / GymWrapper outputs quaternions in [x, y, z, w] format
eef_quat_sim = np.array([-0.0316, 0.0614, 0.0028, 0.9976]) # x, y, z, w
eef_quat_site_sim = np.array([0.6831, 0.0414, 0.0454, 0.7277]) # x, y, z, w (site)

r_eef = R.from_quat(eef_quat_sim)
r_site = R.from_quat(eef_quat_site_sim)

# Calculate relative rotation: r_site = r_eef * r_offset => r_offset = r_eef.inv() * r_site
r_offset = r_eef.inv() * r_site
offset_quat = r_offset.as_quat()
offset_euler = r_offset.as_euler('xyz', degrees=True)

print("Offset Quat (x, y, z, w):", np.array2string(offset_quat, precision=4))
print("Offset Euler (xyz, deg):", np.array2string(offset_euler, precision=4))
