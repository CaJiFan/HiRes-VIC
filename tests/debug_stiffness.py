import numpy as np
import robosuite as suite
import os 
import sys
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))

import gymnasium as gym

from robosuite.controllers import load_composite_controller_config
from hires_vic import envs
from stable_baselines3 import SAC
from robosuite.wrappers import GymWrapper

class ActionNormalizer(gym.ActionWrapper):
    """
    Normalizes actions from [-1, 1] to the environment's action_spec range.
    Assumes the wrapped env has a Box action space.
    """
    def __init__(self, env):
        super().__init__(env)
        self.action_low = self.env.action_space.low
        self.action_high = self.env.action_space.high
        # Normalize to [-1, 1] for the wrapper's action space
        self.action_space = gym.spaces.Box(low=-1, high=1, shape=self.env.action_space.shape, dtype=np.float32)
    
    def action(self, action):
        # Scale from [-1, 1] to [low, high]
        scaled_action = self.action_low + (action + 1) * (self.action_high - self.action_low) / 2
        return scaled_action
    

def inspect_action_space():
    # 1. Setup the exact config you use in training
    controller_config = load_composite_controller_config(controller="BASIC", robot="panda")
    arm_config = controller_config["body_parts"]["right"]
    arm_config["type"] = "OSC_POSE"
    arm_config["impedance_mode"] = "variable_kp"

    phantom_parts = ["left", "torso", "head", "base", "legs"]
    for part in phantom_parts:
        controller_config["body_parts"].pop(part, None)


    env = suite.make(
        env_name="TiltedWipe", # Or Door
        robots="Panda",
        controller_configs=controller_config,
        has_renderer=False,
        use_camera_obs=False,
        control_freq=20,
        horizon=300,
        use_object_obs=True,
    )

    # 3. Extract Action Spec
    low, high = env.action_spec
    print(f"\n{'='*40}")
    print(f"ACTION SPACE INSPECTION for {env.action_dim} Dimensions")
    print(f"{'='*40}")
    
    # 4. Print Ranges per Index
    print(f"{'Idx':<4} | {'Low':<10} | {'High':<10} | {'Guess'}")
    print("-" * 40)
    
    for i in range(len(low)):
        l, h = low[i], high[i]
        
        # Heuristic to identify the part
        if l == 10 and h == 200:
            guess = "STIFFNESS (Kp)"
        elif l == -1 and h == 1:
            guess = "MOTION / GRIPPER"
        else:
            guess = "UNKNOWN"
            
        print(f"{i:<4} | {l:<10.2f} | {h:<10.2f} | {guess}")

    print("-" * 40)
    
    # # 5. Check Controller Internal Name Mapping (The Source of Truth)
    # robot = env.robots[0]
    # # This digs into the controller to find the exact naming order
    # print(robot.composite_controller.__dict__.keys())  # Debug print to check available attributes
    # if hasattr(robot.composite_controller, "part_controller_config"):
    #     print("\nCONTROLLER INTERNAL MAPPING:")
    #     print(robot.composite_controller.part_controller_config)


    dummy_action = np.random.uniform(low=-1, high=1, size=(12,))
    print(f"\nSample Action: {dummy_action}")


    env = GymWrapper(env)
    # env = ActionNormalizer(env)
    env.reset()

    obs, reward, terminated, truncated, info = env.step(dummy_action)



    print(env.action_space)
    print(env.action_spec)

    print()

    # print(env.observation_space)
    print(env.observation_spec().keys())
    for key in env.observation_spec().keys():
        if "eef" in key:
            print(f"  {key}: {env.observation_spec()[key]}")
        
        if "wipe" in key:
            print(f"  {key}: {env.observation_spec()[key]}")

        if "object" in key:
            print(f"  {key}: {env.observation_spec()[key]}")
        
    # print(obs)

    
    model = SAC('MlpPolicy', env, verbose=1)
    model.learn(total_timesteps=10)

    # print(returns)
inspect_action_space()