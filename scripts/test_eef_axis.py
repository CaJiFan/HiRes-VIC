#!/usr/bin/env python3
"""
Quick standalone script to detect which robot-local axis corresponds to world vertical.
Usage: python scripts/test_eef_axis.py --env TiltedWipe
"""
import os
import sys
import argparse
import numpy as np

# Ensure project root is on PYTHONPATH so custom envs register
ROOT = os.path.abspath(os.path.join(os.path.dirname(__file__), '..'))
if ROOT not in sys.path:
    sys.path.insert(0, ROOT)

import robosuite as suite
from robosuite import load_composite_controller_config
from robosuite.utils import transform_utils as T

# Import project envs to register any custom Robosuite environments
try:
    import hires_vic.envs  # noqa: F401
except Exception:
    # not fatal; custom envs may not be present in some contexts
    pass


def detect_axes(env_name: str):
    # Build a basic controller config similar to training scripts
    controller_config = load_composite_controller_config(controller="BASIC", robot="panda")
    phantom_parts = ["left", "torso", "head", "base", "legs"]
    for part in phantom_parts:
        controller_config["body_parts"].pop(part, None)
    arm_config = controller_config["body_parts"]["right"]
    arm_config["type"] = "OSC_POSE"
    arm_config["impedance_mode"] = "variable_kp"

    env = suite.make(
        env_name=env_name,
        robots="Panda",
        controller_configs=controller_config,
        has_renderer=False,
        use_object_obs=True,
        has_offscreen_renderer=False,
        use_camera_obs=False,
        reward_shaping=True,
        horizon=500,
    )

    try:
        env.reset()
    except Exception:
        # some envs may require a step; ignore failures here
        pass

    # Try to get the raw robosuite observation dict
    raw_obs = None
    try:
        raw_obs = env._get_observations()
    except Exception:
        try:
            # wrapped envs might expose unwrapped
            raw_obs = env.unwrapped._get_observations()
        except Exception:
            raw_obs = None

    if not isinstance(raw_obs, dict):
        print("Failed to obtain raw observations from the environment.")
        env.close()
        return 2

    # Find an EEF quaternion key
    eef_quat = None
    eef_key = None
    for k in raw_obs.keys():
        if 'eef_quat' in k.lower():
            eef_key = k
            eef_quat = np.asarray(raw_obs[k], dtype=np.float32)
            break

    if eef_quat is None:
        print("No end-effector quaternion key found in raw observations. Keys:\n", list(raw_obs.keys()))
        env.close()
        return 3

    # Normalize quaternion
    q = eef_quat
    n = np.linalg.norm(q)
    if n < 1e-12:
        q = np.array([1.0, 0.0, 0.0, 0.0], dtype=np.float32)
    else:
        q = (q / n).astype(np.float32)

    # Convert to rotation matrix and extract local axes in world coordinates
    R = T.quat2mat(q)
    x_world = R[:, 0]
    y_world = R[:, 1]
    z_world = R[:, 2]

    world_z = np.array([0.0, 0.0, 1.0], dtype=np.float32)

    def angle_to_world(a):
        dot = float(np.dot(a, world_z))
        dot = max(-1.0, min(1.0, dot))
        ang = np.degrees(np.arccos(dot))
        return ang, dot

    ax = angle_to_world(x_world)
    ay = angle_to_world(y_world)
    az = angle_to_world(z_world)

    print(f"Detected EEF quaternion key: {eef_key}")
    print("EEF local +X in world:", np.round(x_world, 4), f"angle_to_world_z={ax[0]:.2f}deg dot={ax[1]:.3f}")
    print("EEF local +Y in world:", np.round(y_world, 4), f"angle_to_world_z={ay[0]:.2f}deg dot={ay[1]:.3f}")
    print("EEF local +Z in world:", np.round(z_world, 4), f"angle_to_world_z={az[0]:.2f}deg dot={az[1]:.3f}")

    # Determine best (signed) axis: choose axis with largest absolute alignment
    axes = {'+X': x_world, '+Y': y_world, '+Z': z_world}
    dots = {k: float(np.dot(v, [0, 0, 1])) for k, v in axes.items()}
    best_axis, best_dot = max(dots.items(), key=lambda kv: abs(kv[1]))
    # If dot < 0, the local -axis aligns with world+Z
    direction = best_axis if best_dot > 0 else '-' + best_axis
    # angle to world Z (use absolute value of dot to report small angle)
    best_angle = np.degrees(np.arccos(max(-1.0, min(1.0, abs(best_dot)))))
    print(f"\nBest signed axis: local {direction} — abs_angle_to_world_z={best_angle:.2f} deg (dot={best_dot:.3f})")
    env.close()
    return 0


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--env', type=str, default='TiltedWipe', help='Robosuite env name (default: TiltedWipe)')
    args = parser.parse_args()

    rc = detect_axes(args.env)
    sys.exit(rc)


if __name__ == '__main__':
    main()
