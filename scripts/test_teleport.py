import sys
import os
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))
from src.train_fixed import parse_args, make_env
import argparse

class MockArgs:
    def __init__(self):
        self.env = "TiltedWipe"
        self.num_markers = 5
        self.use_spd = False
        self.use_lie = False
        self.use_diag = False
        self.use_fixed = False
        self.horizon = 150
        self.primitive_init = "teleport"
        self.camera_names = "frontview"
        self.use_domain_rand = False
        self.use_llm_prior = False
        self.use_vlm = False
        self.use_cameras = False
        self.llm_profile = None
        self.llm_backend = "ollama"
        self.llm_model = "llama3.2"
        self.llm_query_interval = 50
        self.llm_prior_weight = 0.4
        self.llm_anneal_steps = 0
        self.llm_anneal_floor = 0.05

args = MockArgs()
env_fn = make_env(args)
env = env_fn()
obs, info = env.reset()
print("Reset successful! Shape:", obs.shape)
obs, reward, term, trunc, info = env.step(env.action_space.sample())
print("Step successful! Reward:", reward)
env.close()
