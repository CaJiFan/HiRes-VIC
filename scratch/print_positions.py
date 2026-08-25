import sys
sys.path.insert(0, '/home/cjimenez/projects/HiRes-VIC')

import robosuite as suite
from hires_vic.envs.tilted_wipe import TiltedWipe

env = suite.make('TiltedWipe', robots=['Panda'], has_offscreen_renderer=False, use_camera_obs=False)
env.reset()
sim = env.sim

table_bid = sim.model.body_name2id('table')
print('Table pos:     ', sim.data.body_xpos[table_bid])

base_bid = sim.model.body_name2id('robot0_base')
print('Robot base pos:', sim.data.body_xpos[base_bid])

hand_bid = sim.model.body_name2id('robot0_right_hand')
print('Robot hand pos:', sim.data.body_xpos[hand_bid])

site_id = sim.model.site_name2id("gripper0_right_grip_site")
print('Grip site pos: ', sim.data.site_xpos[site_id])

env.close()
