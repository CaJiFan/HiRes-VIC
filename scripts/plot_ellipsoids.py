import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d import Axes3D
import numpy as np
import os

import os

def save_ellipsoid_plot(Kp_matrix, episode, step, save_dir="./plots/ellipsoids"):
    """Saves a 3D plot of the stiffness ellipsoid."""
    os.makedirs(save_dir, exist_ok=True)
    
    fig = plt.figure(figsize=(6, 6))
    ax = fig.add_subplot(111, projection='3d')
    
    # 1. Eigen decomposition to find axes and radii
    eigenvalues, eigenvectors = np.linalg.eigh(Kp_matrix)
    
    # 2. Generate a base unit sphere
    u = np.linspace(0.0, 2.0 * np.pi, 30)
    v = np.linspace(0.0, np.pi, 30)
    x = np.outer(np.cos(u), np.sin(v))
    y = np.outer(np.sin(u), np.sin(v))
    z = np.outer(np.ones_like(u), np.cos(v))
    
    # 3. Deform the sphere using the SPD matrix
    # Scale it down visually so it fits nicely on a plot
    scale = 0.05 
    radii = np.sqrt(eigenvalues) * scale
    
    # Matrix multiplication to stretch and rotate the sphere
    sphere_coords = np.vstack((x.flatten(), y.flatten(), z.flatten()))
    ellipsoid = (eigenvectors @ np.diag(radii) @ sphere_coords)
    
    x_ell = ellipsoid[0, :].reshape(30, 30)
    y_ell = ellipsoid[1, :].reshape(30, 30)
    z_ell = ellipsoid[2, :].reshape(30, 30)
    
    # 4. Plot it
    ax.plot_surface(x_ell, y_ell, z_ell, color='c', alpha=0.6, edgecolor='k', linewidth=0.1)
    
    # 5. Formatting
    ax.set_title(f"Translational Stiffness Ellipsoid | Step: {step}")
    ax.set_xlabel("X Stiffness")
    ax.set_ylabel("Y Stiffness")
    ax.set_zlabel("Z Stiffness")
    
    # Keep axes perfectly square so the ellipsoid's rotation isn't visually distorted
    ax.set_box_aspect([1, 1, 1])
    max_range = np.max(np.sqrt(eigenvalues) * scale)
    ax.set_xlim([-max_range, max_range])
    ax.set_ylim([-max_range, max_range])
    ax.set_zlim([-max_range, max_range])
    
    plt.tight_layout()
    plt.savefig(f"{save_dir}/ellipsoid_ep_{episode:02d}_step_{step:04d}.png")
    plt.close(fig)