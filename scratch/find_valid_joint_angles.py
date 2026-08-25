import sys
sys.path.insert(0, '/home/cjimenez/projects/HiRes-VIC')

import numpy as np
import robosuite as suite
from hires_vic.envs.tilted_wipe import TiltedWipe
from scipy.spatial.transform import Rotation as R
import mujoco

env = suite.make('TiltedWipe', robots=['Panda'], has_offscreen_renderer=False, use_camera_obs=False)
env.reset()
sim = env.sim

site_id = sim.model.site_name2id("gripper0_right_grip_site")
table_bid = sim.model.body_name2id('table')

tilt_rad = np.radians(45.0)
r_y = R.from_euler('y', tilt_rad).as_matrix() # +45 deg pitch
base_rot = np.array([[-1.0, 0.0, 0.0], [0.0, 1.0, 0.0], [0.0, 0.0, -1.0]])
target_rot = r_y @ base_rot

table_centre = np.array(sim.data.body_xpos[table_bid], dtype=float)
normal = np.array([np.sin(tilt_rad), 0.0, np.cos(tilt_rad)])
target_pos = table_centre + 0.30 * normal

joint_names = [f"robot0_joint{i}" for i in range(1, 8)]
qpos_indices = [sim.model.get_joint_qpos_addr(j) for j in joint_names]

# Let's try to search from a neutral posture where the shoulder is tilted forward and elbow is up
# Standard neutral posture for Panda:
neutral_qpos = np.array([0.0, -0.3, 0.0, -2.0, 0.0, 1.8, 0.0])

# Copy neutral qpos into sim to start the search from a clean posture
sim.data.qpos[qpos_indices] = neutral_qpos
sim.forward()

print("Starting IK search from neutral posture...")
converged = False
for it in range(100):
    cur_eef = sim.data.site_xpos[site_id].copy()
    cur_rot = sim.data.site_xmat[site_id].reshape(3, 3)
    
    delta_pos = target_pos - cur_eef
    rot_err_mat = target_rot @ cur_rot.T
    r_err = R.from_matrix(rot_err_mat).as_rotvec()
    
    delta = np.hstack([delta_pos, r_err])
    
    if np.linalg.norm(delta_pos) < 0.001 and np.linalg.norm(r_err) < 0.01:
        print(f"IK converged at iteration {it}!")
        converged = True
        break
        
    nv = sim.model.nv
    jacp = np.zeros((3, nv))
    jacr = np.zeros((3, nv))
    mujoco.mj_jacSite(sim.model._model, sim.data._data, jacp, jacr, site_id)
    
    J = np.vstack([jacp[:, qpos_indices], jacr[:, qpos_indices]])
    damping = 0.02
    JJT = J @ J.T + damping ** 2 * np.eye(6)
    dq = J.T @ np.linalg.solve(JJT, delta)
    dq = np.clip(dq, -0.05, 0.05)
    
    sim.data.qpos[qpos_indices] += dq
    sim.forward()

eef_pos = sim.data.site_xpos[site_id]
print("EEF pos reached:  ", np.round(eef_pos, 3))
print("Target pos:        ", np.round(target_pos, 3))
print("Error norm:        ", np.linalg.norm(eef_pos - target_pos))
print("Arm collisions:    ", env.check_contact(env.robots[0].robot_model))
if converged:
    print("Converged Joint Angles:", np.round(sim.data.qpos[qpos_indices], 4))

env.close()
