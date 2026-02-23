import numpy as np
import robosuite as suite
from robosuite.controllers import load_composite_controller_config

def inspect_action_space():
    # 1. Setup the exact config you use in training
    controller_config = load_composite_controller_config(controller="BASIC", robot="panda")
    arm_config = controller_config["body_parts"]["right"]
    arm_config["type"] = "OSC_POSE"
    arm_config["impedance_mode"] = "variable_kp"
    arm_config["kp_limits"] = [10, 200]
    arm_config["damping_ratio_limits"] = [1.0, 1.0]

    # 2. Create the raw Robosuite env
    env = suite.make(
        env_name="Wipe", # Or Door
        robots="Panda",
        controller_configs=controller_config,
        has_renderer=False,
        use_camera_obs=False,
        use_object_obs=False,
        control_freq=20,
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
    
    # 5. Check Controller Internal Name Mapping (The Source of Truth)
    robot = env.robots[0]
    # This digs into the controller to find the exact naming order
    print(robot.composite_controller.__dict__.keys())  # Debug print to check available attributes
    if hasattr(robot.composite_controller, "part_controller_config"):
        print("\nCONTROLLER INTERNAL MAPPING:")
        print(robot.composite_controller.part_controller_config)

inspect_action_space()