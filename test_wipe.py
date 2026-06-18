import robosuite as suite
from robosuite.wrappers import GymWrapper
from hires_vic.wrappers import GeometricWrapper
import hires_vic.envs
WIPE_TASK_CONFIG = {"use_condensed_obj_obs": True, "num_markers": 5}
env = suite.make('TiltedWipe', robots='Panda', has_renderer=False, use_object_obs=True, use_camera_obs=False, task_config=WIPE_TASK_CONFIG)
env = GymWrapper(env)
env = GeometricWrapper(env, use_spd_manifold=False, use_lie_group=False, use_diag_manifold=False, use_fixed=True, is_eval=True, task_type='wipe')
print('GeometricWrapper obs space:', env.observation_space)
