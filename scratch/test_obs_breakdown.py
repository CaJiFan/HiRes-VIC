import sys
sys.path.insert(0, '/home/cjimenez/projects/HiRes-VIC')

import numpy as np
import robosuite as suite
from robosuite.wrappers import GymWrapper
from hires_vic.envs.tilted_wipe import TiltedWipe
from hires_vic.wrappers import GeometricWrapper
from src.train_fixed import load_wipe_task_config

task_config = load_wipe_task_config()
task_config["num_markers"] = 5
task_config["use_condensed_obj_obs"] = False

raw_env = suite.make(
    'TiltedWipe',
    robots=['Panda'],
    has_offscreen_renderer=False,
    use_camera_obs=False,
    task_config=task_config,
)

gym_env = GymWrapper(raw_env)
wrap_env = GeometricWrapper(
    gym_env,
    use_spd_manifold=False,
    task_type='wipe',
)

obs, _ = wrap_env.reset()
print("Wrapped obs shape:", obs.shape)
print("Last 3 elements of obs (active waypoint rel vector):", obs[-3:])

# Let's get active waypoint directly from env to compare
eef_pos = raw_env._get_observations()['robot0_eef_pos']
all_markers = raw_env.model.mujoco_arena.markers
unwiped = []
for marker in all_markers:
    bid = raw_env.sim.model.body_name2id(marker.root_body)
    pos = np.array(raw_env.sim.data.body_xpos[bid], dtype=float)
    unwiped.append((marker, pos))
unwiped.sort(key=lambda item: item[1][1])
first_marker_pos = unwiped[0][1]
expected_rel = first_marker_pos - eef_pos

print("Expected relative vector from EEF to first marker:", expected_rel)

raw_env.close()
