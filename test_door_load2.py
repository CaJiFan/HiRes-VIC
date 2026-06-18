import torch
import robosuite as suite
from robosuite.wrappers import GymWrapper
from stable_baselines3 import SAC
from hires_vic.wrappers import GeometricWrapper

env = suite.make('Door', robots='Panda', has_renderer=False, use_object_obs=True, use_camera_obs=False)
env = GymWrapper(env)
env = GeometricWrapper(env, use_spd_manifold=False, use_lie_group=False, use_diag_manifold=False, use_fixed=False, is_eval=True, task_type='door')

model_path = '/home/cjimenez/projects/HiRes-VIC/logs/best_models/SAC_DOOR_BASELINE_G0.99_SEED_1/best_model.zip'
print("Loading model...")
try:
    model = SAC.load(model_path, env=env, device='cpu')
    print("Success without custom_objects!")
except Exception as e:
    print("Failed!", str(e))
