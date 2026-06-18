import robosuite as suite
from robosuite.wrappers import GymWrapper
from hires_vic.wrappers import GeometricWrapper

env = suite.make('Door', robots='Panda', has_renderer=False, use_object_obs=True, use_camera_obs=False)
env = GymWrapper(env)
env = GeometricWrapper(env, use_spd_manifold=False, use_lie_group=False, use_diag_manifold=False, use_fixed=True, is_eval=True, task_type='door')
print('GeometricWrapper obs space:', env.observation_space)
