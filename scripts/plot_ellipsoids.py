import os
import sys
import argparse
import numpy as np

# Add the project root to the path so it can find hires_vic
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), "..")))

import pandas as pd
import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d import Axes3D
from stable_baselines3 import SAC
import torch

from hires_vic import envs
import robosuite as suite
from robosuite.controllers import load_composite_controller_config
from robosuite.wrappers import GymWrapper
from hires_vic.wrappers.pih_curriculum import RobosuiteTeleportWrapper

from generate_tables import make_env

def get_kp_matrix(env):
    """Extracts the 3x3 translational stiffness matrix from the controller."""
    # In newer Robosuite with composite controllers, the arm controller is under part_controllers
    robot = env.unwrapped.robots[0]
    if hasattr(robot, "composite_controller"):
        controller = robot.composite_controller.part_controllers["right"]
    else:
        controller = robot.controller
    
    kp = controller.kp
    
    # Depending on the impedance mode, kp might be a 6D vector or a 6x6 matrix
    if kp.ndim == 1:
        # Diagonal stiffness (variable_kp)
        return np.diag(kp[:3])
    elif kp.ndim == 2:
        # Full stiffness matrix (riemannian_kp)
        return kp[:3, :3]
    return np.eye(3) * 100.0

def collect_trajectory(model_path, env_name, config_name):
    env = make_env(env_name, config_name, model_path=model_path)
    model = SAC.load(model_path, env=env, device="cuda" if torch.cuda.is_available() else "cpu")
    
    obs, _ = env.reset()
    done = truncated = False
    
    positions = []
    stiffness_matrices = []
    
    while not (done or truncated):
        # Extract end-effector position from raw Robosuite observations
        raw_obs = env.unwrapped._get_observations()
        eef_pos = raw_obs.get('robot0_eef_pos', np.zeros(3))
        
        positions.append(eef_pos.copy())
        stiffness_matrices.append(get_kp_matrix(env).copy())
        
        action, _ = model.predict(obs, deterministic=True)
        obs, reward, terminated, truncated, _ = env.step(action)
        done = bool(terminated)
        
    env.close()
    return np.array(positions), stiffness_matrices

def plot_ellipsoid(ax, center, matrix, scale=0.0005, color='b', alpha=0.3):
    """Plots a 3D ellipsoid based on a stiffness matrix."""
    if np.isnan(matrix).any() or np.isinf(matrix).any():
        return

    # Eigendecomposition to find axes and radii
    eigenvalues, eigenvectors = np.linalg.eigh(matrix)
    
    # We plot the stiffness ellipsoid (larger eigenvalue = larger radius)
    # Scale down the radii so they fit visually on the trajectory
    # Clip to prevent 0 radius (which causes 0 area faces and crashes matplotlib shading)
    radii = np.clip(np.abs(eigenvalues), 1e-6, np.inf) * scale
    
    # Create a base sphere
    u = np.linspace(0.0, 2.0 * np.pi, 20)
    v = np.linspace(0.0, np.pi, 20)
    x = np.outer(np.cos(u), np.sin(v))
    y = np.outer(np.sin(u), np.sin(v))
    z = np.outer(np.ones_like(u), np.cos(v))
    
    # Vectorized transformation
    points = np.stack([x.flatten(), y.flatten(), z.flatten()], axis=0) # (3, N)
    scaled_points = radii[:, np.newaxis] * points # (3, N)
    transformed_points = eigenvectors @ scaled_points # (3, N)
    
    x = transformed_points[0, :].reshape(x.shape) + center[0]
    y = transformed_points[1, :].reshape(y.shape) + center[1]
    z = transformed_points[2, :].reshape(z.shape) + center[2]
    
    if np.isnan(x).any():
        return
        
    import matplotlib.colors as mcolors
    rgba = mcolors.to_rgba(color, alpha=alpha)
    # shade=False prevents the buggy face-normals calculation from crashing on flat ellipsoids
    ax.plot_surface(x, y, z, color=rgba, edgecolors='none', shade=False)

def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--env", type=str, default="TiltedWipe", choices=["NutAssemblySquare", "Door", "TiltedWipe"])
    parser.add_argument("--baseline_path", type=str, required=True, help="Path to best_model.zip for BASELINE")
    parser.add_argument("--spd_path", type=str, required=True, help="Path to best_model.zip for FULL_GRL (SPD)")
    parser.add_argument("--step_interval", type=int, default=20, help="Plot an ellipsoid every N steps")
    parser.add_argument("--scale", type=float, default=0.00015, help="Visual scaling factor for ellipsoids")
    args = parser.parse_args()

    print(f"Running Baseline policy ({args.baseline_path})...")
    pos_base, K_base = collect_trajectory(args.baseline_path, args.env, "BASELINE")
    
    print(f"Running SPD/GRL policy ({args.spd_path})...")
    pos_spd, K_spd = collect_trajectory(args.spd_path, args.env, "FULL_GRL")

    # Plotting side by side
    fig = plt.figure(figsize=(14, 7))
    
    for idx, (title, pos, K_mats, color) in enumerate([
        ("Standard RL (Baseline)", pos_base, K_base, 'r'),
        ("Manifold-Aware RL (FULL_GRL)", pos_spd, K_spd, 'g')
    ]):
        ax = fig.add_subplot(1, 2, idx + 1, projection='3d')
        ax.set_title(title)
        
        # Plot trajectory line
        ax.plot(pos[:, 0], pos[:, 1], pos[:, 2], color='k', linewidth=1.5, label='Trajectory')
        
        # Plot ellipsoids at intervals
        for step in range(0, len(pos), args.step_interval):
            plot_ellipsoid(ax, pos[step], K_mats[step], scale=args.scale, color=color, alpha=0.4)
            
        ax.set_xlabel('X')
        ax.set_ylabel('Y')
        ax.set_zlabel('Z')
        
        # Ensure equal aspect ratio for accurate 3D representation
        max_range = np.array([pos[:,0].max()-pos[:,0].min(), pos[:,1].max()-pos[:,1].min(), pos[:,2].max()-pos[:,2].min()]).max() / 2.0
        mid_x = (pos[:,0].max()+pos[:,0].min()) * 0.5
        mid_y = (pos[:,1].max()+pos[:,1].min()) * 0.5
        mid_z = (pos[:,2].max()+pos[:,2].min()) * 0.5
        ax.set_xlim(mid_x - max_range, mid_x + max_range)
        ax.set_ylim(mid_y - max_range, mid_y + max_range)
        ax.set_zlim(mid_z - max_range, mid_z + max_range)

    plt.tight_layout()
    output_path = f"outputs/{args.env}_ellipsoids.png"
    plt.savefig(output_path, dpi=300)
    print(f"\nSaved beautiful 3D ellipsoid comparison to: {output_path}")

if __name__ == "__main__":
    main()
