import torch
import numpy as np
from stable_baselines3 import PPO, SAC, TD3
from sb3_contrib import TQC
import argparse
import os 
import sys
from time import sleep
import imageio

sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))
from hires_vic.envs.gymnasium_wrapper import RobosuiteGymnasiumWrapper, RobosuitePhysicsWrapper
from scripts.plot_ellipsoids import save_ellipsoid_plot

import robosuite as suite
import robosuite.controllers.parts.controller_factory as factory
from robosuite import load_composite_controller_config

from hires_vic.envs.riemannian_controller import RiemannianController

factory.arm_controllers.OperationalSpaceController = RiemannianController


from robosuite.environments.manipulation.wipe import Wipe
from robosuite.environments.base import register_env
from stable_baselines3.common.monitor import Monitor
from stable_baselines3.common.vec_env import DummyVecEnv

class TiltedWipe(Wipe):
    """Custom Wipe environment with a tilted table."""
    def __init__(self, tilt_angle_degrees=45.0, **kwargs):
        self.tilt_angle_rad = np.radians(tilt_angle_degrees)
        super().__init__(**kwargs)

    def _load_model(self):
        super()._load_model()
        table = self.model.mujoco_arena.table_body
        table.set("euler", f"0 {self.tilt_angle_rad:.4f} 0")

register_env(TiltedWipe)

# =================================================================
# 3. ARGUMENT PARSING
# =================================================================
def parse_args():
    parser = argparse.ArgumentParser(description="Evaluate RL on Robosuite")
    parser.add_argument("--env", type=str, default="Door", help="Name of the Robosuite environment")
    parser.add_argument("--run_name", type=str, default="baseline", help="Name for logging/loading models")
    parser.add_argument("--algorithm", type=str, default="SAC", help="RL Algorithm to use")
    parser.add_argument("--episodes", type=int, default=10, help="Number of episodes to evaluate")
    parser.add_argument("--render", action="store_true", help="Turn on the MuJoCo viewer popup")
    parser.add_argument("--seed", type=int, default=1, help="Random seed (e.g., 1, 2, 3)")
    # Ablation Flags
    parser.add_argument("--use_spd", action="store_true", help="Enable Riemannian SPD stiffness mapping")
    parser.add_argument("--use_lie", action="store_true", help="Enable Lie Group orientation prior")
    
    return parser.parse_args()

# =================================================================
# 4. ENVIRONMENT CREATION
# =================================================================
def make_env(args):
    controller_config = None
    is_vic = "VIC" in args.run_name
    kp_limits = [20, 200] # [50, 300]
    env_name = args.env

    if is_vic:
        controller_config = load_composite_controller_config(controller="BASIC", robot="panda")

        phantom_parts = ["left", "torso", "head", "base", "legs"]
        for part in phantom_parts:
            controller_config["body_parts"].pop(part, None)
        
        arm_config = controller_config["body_parts"]["right"]
        arm_config["type"] = "OSC_POSE"

        arm_config["impedance_mode"] = "riemannian_kp" if args.use_spd else "variable_kp"
        
        # 0 = Completely limp (gravity comp only), 300 = Very stiff
        arm_config["kp_limits"] = kp_limits 
        arm_config["damping_ratio_limits"] = [1.0, 1.0] # Force critical damping
    
    task_kwargs = {
        "has_renderer": False,
        "has_offscreen_renderer": True, # for saving rollout videos later
        "use_camera_obs": False,
        "reward_shaping": True,
        "horizon": 300 if env_name == "Door" else 1000, # Shorter episodes for Door
        "control_freq": 20,              #  50ms per step
        "kp_limits": kp_limits if is_vic else None,  # Only pass kp_limits if using VIC
    }

    if args.env == "TiltedWipe":
        task_kwargs["tilt_angle_degrees"] = 35.0

    env = RobosuiteGymnasiumWrapper(
        env_name=env_name,
        robots="Panda",
        controller_configs=controller_config,
        task_kwargs=task_kwargs,
        use_spd_manifold=args.use_spd, 
        use_lie_group=args.use_lie
    )
        
    stiff_penalty = 0.01 if is_vic else 0.0
    
    phy_env = RobosuitePhysicsWrapper(
        env, 
        stiffness_penalty=stiff_penalty, 
        force_penalty=0.02, 
        max_force_threshold=35.0
    )

    # env = Monitor(env)
    return phy_env, env

# =================================================================
# 5. MAIN EVALUATION LOOP
# =================================================================
def main():
    args = parse_args()
    device = "cuda" if torch.cuda.is_available() else "cpu"
    
    env_name = args.env
    run_name = f'{args.algorithm}_{env_name.upper()}_{args.run_name}_SPD_{str(args.use_spd).upper()}_LG_{str(args.use_lie).upper()}_SEED_{args.seed}'

    model_path = f"./outputs/models/{run_name}_final"
    video_dir = f"./outputs/videos/{run_name}"
    os.makedirs(video_dir, exist_ok=True)
    
    env, wrapper = make_env(args)

    print(f"Loading {args.algorithm} model from {model_path}...")
    if args.algorithm == "PPO":
        model = PPO.load(model_path, env=env, device=device)
    elif args.algorithm == "SAC":
        model = SAC.load(model_path, env=env, device=device)
    elif args.algorithm == "TD3":
        model = TD3.load(model_path, env=env, device=device)
    elif args.algorithm == "TQC":
        model = TQC.load(model_path, env=env, device=device)
    
    print(f"\nStarting Formal Evaluation over {args.episodes} episodes...")
    successes = 0
    
    for ep in range(args.episodes):
        obs, info = env.reset()
        done = False
        truncated = False
        ep_reward = 0.0
        step_counter = 0

        video_path = os.path.join(video_dir, f"eval_ep_{ep+1:02d}.mp4")
        video_writer = imageio.get_writer(video_path, fps=20)
        
        while not (done or truncated):
            action, _states = model.predict(obs, deterministic=True)
            obs, reward, done, truncated, info = env.step(action)
            ep_reward += reward
            step_counter += 1

            frame = wrapper.env.sim.render(camera_name="frontview", width=1024, height=1024)[::-1]
            
            # 3. Append the frame to the video
            video_writer.append_data(frame)
            
            if args.render:
                wrapper.render()

            # if args.use_spd and step_counter % 50 == 0:
            #     # Navigate through the Robosuite hierarchy to your RiemannianController
            #     custom_controller = wrapper.env.robots[0].composite_controller.part_controllers['right']
                
            #     # Extract the matrix you defined in set_goal()!
            #     Kp_pos_3x3 = custom_controller.kp_pos_matrix
                
            #     # Save the plot
            #     save_ellipsoid_plot(Kp_pos_3x3, episode=ep, step=step_counter)

            print(reward)
            sleep(0.01)  # Slow down for better visualization

        video_writer.close()
        print(f"🎥 Saved video for Episode {ep+1} to {video_path}")

        is_success = wrapper.env._check_success()
        if is_success:
            successes += 1
            
        print(f"Episode {ep + 1}/{args.episodes} | Reward: {ep_reward:.2f} | Success: {is_success}")

    print("=====================================================")
    print(f"Evaluation Complete for {args.algorithm} on {args.env}")
    print(f"Ablations -> SPD: {args.use_spd} | Lie Group: {args.use_lie}")
    print(f"Overall Success Rate: {(successes / args.episodes) * 100:.1f}%")
    print("=====================================================")

    env.close()

if __name__ == "__main__":
    main()