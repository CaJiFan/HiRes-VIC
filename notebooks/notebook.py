# %% [markdown]
# # NutAssembly (Robosuite) — Interactive Notebook
# 
# Create a Robosuite `NutAssembly` environment (e.g., `NutAssemblySquare`), compute a handle target from the nut quaternion, run a simple scripted primitive to line up/grasp the handle, and record an inline video.
# 
# This is a best-effort notebook — adapt camera names, controller config, and action indices to your local setup.

# %%
import os
import sys
sys.path.append(os.path.abspath(os.path.join(os.getcwd(), '..')))

import io
import base64
import numpy as np
import imageio
from IPython.display import HTML, display
from scipy.spatial.transform import Rotation as R
from hires_vic.wrappers import WipeMetricWrapper, GeometricWrapper

from hires_vic.envs.riemannian_controller import RiemannianController
import robosuite.controllers.parts.controller_factory as factory
factory.arm_controllers.OperationalSpaceController = RiemannianController

import logging

from robosuite.utils.log_utils import ROBOSUITE_DEFAULT_LOGGER

# Suppress all robosuite warnings (like joint limits and macro files)
ROBOSUITE_DEFAULT_LOGGER.setLevel(logging.ERROR)

def display_video(path, width=640):
    mp4 = open(path,'rb').read()
    data_url = "data:video/mp4;base64," + base64.b64encode(mp4).decode()
    html = f'<video width="{width}" controls><source src="{data_url}" type="video/mp4"></video>'
    display(HTML(html))

def frames_to_mp4(frames, path, fps=30):
    imageio.mimwrite(path, frames, fps=fps, macro_block_size=None)


# %%
def capture_frame_from_env(env):
    try:
        # robosuite common call
        frame = env.render(camera_name='frontview')
        if frame is None:
            frame = env.render()
    except Exception:
        frame = None
    if frame is None:
        try:
            frame = env.unwrapped.sim.render(width=640, height=480, camera_name='frontview')
        except Exception:
            frame = None
    return frame

# %%
# Create a NutAssembly env (best-effort). Change env_name if needed.
env_name = 'NutAssemblySquare'
fixed_kp = 150.0
try:
    import robosuite as suite
    from robosuite.wrappers import GymWrapper
    from robosuite import load_composite_controller_config
    use_spd_manifold = False
    use_lie_group = False
    use_llm_prior = False
    use_fixed = True
    controller_config = load_composite_controller_config(controller="BASIC", robot="panda")
    phantom_parts = ["left", "torso", "head", "base", "legs"]
    for part in phantom_parts:
        controller_config["body_parts"].pop(part, None)
    arm_config = controller_config["body_parts"]["right"]
    arm_config["type"] = "OSC_POSE"
    arm_config["impedance_mode"] = "riemannian_kp" if use_spd_manifold else "fixed" if use_fixed else "variable_kp"
    arm_config["kp_limits"] = [1, 300]
    arm_config["damping_ratio_limits"] = [1.0, 1.0]
    if use_fixed:
        arm_config["kp"] = fixed_kp
    env = suite.make(
        env_name=env_name,
        robots='Panda',
        controller_configs=controller_config, 
        has_renderer=False, 
        use_object_obs=True, 
        has_offscreen_renderer=True, 
        use_camera_obs=True, 
        camera_names='frontview', 
        reward_shaping=True
    )
    env = GymWrapper(env)
    env = GeometricWrapper(
        env, 
        use_spd_manifold=use_spd_manifold, 
        use_lie_group=use_lie_group, 
        use_llm_prior=use_llm_prior,
        use_fixed=use_fixed,
        is_eval=True
    )
    print('Wrapped with GeometricWrapper')
except Exception as e:
    raise RuntimeError('Failed to create Robosuite NutAssembly environment. Ensure robosuite is installed and the env name is correct.') from e

print('Env created:', env_name, 'wrapped type:', type(env))
obs = env.reset()
print('Reset done. If obs is dict, keys: ', list(obs.keys()) if isinstance(obs, dict) else type(obs))


# %%
def compute_handle_pos(raw_obs, offset=0.04, quat_debug=False):
    "Return (handle_pos, nut_pos, chosen_quat) or (None, None, None)"
    if not isinstance(raw_obs, dict):
        return None, None, None
    nut_pos = None
    nut_quat = None
    for k, v in raw_obs.items():
        kl = k.lower()
        if 'nut' in kl and 'pos' in kl and 'to_' not in kl:
            nut_pos = np.asarray(v).flatten()[:3]
        if 'nut' in kl and 'quat' in kl and 'to_' not in kl:
            nut_quat = np.asarray(v).flatten()[:4]
    if nut_pos is None:
        return None, None, None
    if nut_quat is None:
        return nut_pos, nut_pos, None
    q_raw = np.asarray(nut_quat).flatten()
    if q_raw.size < 4:
        return nut_pos, nut_pos, None
    # try two common orderings and pick the one whose local Z aligns best with world Z (upright heuristic)
    orders = [q_raw[:4], np.array([q_raw[1], q_raw[2], q_raw[3], q_raw[0]])]
    best = None
    best_score = -1.0
    best_rot = None
    for q in orders:
        try:
            nq = q / (np.linalg.norm(q) + 1e-12)
            rot = R.from_quat(nq)
            local_z = rot.apply([0.0, 0.0, 1.0])
            score = abs(np.dot(local_z, np.array([0.0, 0.0, 1.0])))
            if score > best_score:
                best_score = score
                best = nq
                best_rot = rot
        except Exception:
            continue
    if best_rot is None:
        return nut_pos, nut_pos, nut_quat
    # use local +X, enforce negative sign and offset (nut_pos - local_x * offset)
    offset_v = best_rot.apply([offset, 0.0, 0.0])
    # axis_unit = local_x / (np.linalg.norm(local_x) + 1e-12)
    handle_pos = nut_pos + offset_v
    if quat_debug:
        print('Chosen quat (x,y,z,w):', best, 'offset_x=', offset_v, 'best_rot=', best_rot, 'nut_quat=', nut_quat)
    return handle_pos, nut_pos, best


# %%
def determine_action_indices(env):
    action_dim = int(env.action_space.shape[-1])
    # print(action_dim, 'action space shape:', env.action_space.shape)
    # Heuristic: if action_dim large, assume SPD manifold layout -> pos_idx 9, else 6
    if hasattr(env, 'use_spd_manifold') and getattr(env, 'use_spd_manifold'):
        pos_idx = 9
    else:
        pos_idx = 6 if action_dim > 7 else 0 # fixed
    gripper_idx = max(0, action_dim - 1)
    return action_dim, pos_idx, gripper_idx

def scripted_primitive_policy(env, raw_obs, prev_delta_ori, step_phase=0.0, quat_debug=False):
    "Scripted primitive policy for NutAssembly: uses `compute_handle_pos` and a simple PD approach."
    action_dim, pos_idx, gripper_idx = determine_action_indices(env)
    action = np.zeros((action_dim,), dtype=np.float32)

    # Extract TCP pose/quaternion from raw_obs when available
    tcp_pos = None
    tcp_quat = None
    try:
        if isinstance(raw_obs, dict):
            if 'robot0_eef_pos' in raw_obs:
                tcp_pos = np.asarray(raw_obs['robot0_eef_pos']).flatten()[:3]
            if 'robot0_eef_quat' in raw_obs:
                tcp_quat = np.asarray(raw_obs['robot0_eef_quat']).flatten()[:4]
    except Exception:
        pass

    handle_pos, nut_pos, chosen_quat = compute_handle_pos(raw_obs, offset=0.04, quat_debug=quat_debug)
    if handle_pos is None:
        # Fallback: sample an action or return zeros
        try:
            return env.action_space.sample()
        except Exception:
            return action

    # Obtain peg position (MuJoCo direct) as a fallback target
    peg_pos = None
    try:
        sim = env.unwrapped.sim
        peg_id = sim.model.body_name2id('peg1')
        peg_pos = np.array(sim.data.body_xpos[peg_id]) + np.array([0.0, 0.0, 0.08])
    except Exception:
        peg_pos = nut_pos

    # print(env)
    setattr(env, 'suppress_forced_gripper', True)
    # print('before getattr:', getattr(env, 'suppress_forced_gripper', False))

    # Targets: grasp (handle), midpoint, hover over peg
    grasp_target = handle_pos + np.array([0.0, 0.0, 0.01])
    mid_target = handle_pos + np.array([0.025, 0.0, 0.05])
    hover_target = peg_pos + np.array([0.0, 0.0, 0.01])

    OPEN = -1.0
    CLOSE = 1.0

    # Phase selection based on normalized step_phase in [0,1]
    phase = float(step_phase) if step_phase is not None else 0.0
    if phase < 0.25:
        target = grasp_target
        gripper_act = OPEN
    elif phase < 0.40:
        target = handle_pos
        gripper_act = OPEN
    elif phase < 0.50:
        target = handle_pos
        gripper_act = CLOSE
    elif phase < 0.75:
        target = mid_target
        gripper_act = CLOSE
    else:
        target = hover_target
        gripper_act = CLOSE

    # Position PD (simple proportional for demo). Tune gain as needed.
    if tcp_pos is None:
        delta_pos = np.zeros(3, dtype=np.float32)
    else:
        primitive_approach_gain = 10
        delta_pos = (target - tcp_pos) * (primitive_approach_gain * 0.015)

    # Orientation correction: small rotation-vector in EEF local frame
    delta_ori = np.zeros(3, dtype=np.float32)
    ori_scale = 0.2       # The Rotational Gain (Smaller fractional steps)
    smooth_alpha = 0.15   # The Acceleration Curve (Gentle ease-in)
    max_ori_step = 0.02
    
    if tcp_quat is not None and chosen_quat is not None:
        try:
            r_current = R.from_quat(tcp_quat)
            r_nut = R.from_quat(chosen_quat)
            
            target_z = np.array([0.0, 0.0, -1.0])
            
            # 1. Get the Nut's X-axis, flattened to the table
            nut_x = r_nut.apply([1.0, 0.0, 0.0])
            target_x_base = np.array([nut_x[0], nut_x[1], 0.0])
            target_x_base = target_x_base / (np.linalg.norm(target_x_base) + 1e-12)
            
            # 2. Get the Gripper's X-axis, flattened to the table
            gripper_x = r_current.apply([1.0, 0.0, 0.0])
            gripper_x_flat = np.array([gripper_x[0], gripper_x[1], 0.0])
            gripper_x_flat = gripper_x_flat / (np.linalg.norm(gripper_x_flat) + 1e-12)
            
            # --- THE FIX: Pick the side using ONLY 2D flat alignment ---
            # If the dot product is negative, the handle is facing away. Flip it ONCE.
            if np.dot(target_x_base, gripper_x_flat) < 0:
                target_x = -target_x_base
            else:
                target_x = target_x_base
                
            # 3. Build the single, perfectly stable target matrix
            target_y = np.cross(target_z, target_x)
            target_matrix = np.column_stack((target_x, target_y, target_z))
            r_target = R.from_matrix(target_matrix)
            
            # Calculate exact World Frame error
            r_error_world = r_target * r_current.inv()
            delta_ori_raw = r_error_world.as_rotvec() 
            
            # Scale, smooth, and CAP THE SPEED
            scaled = delta_ori_raw * ori_scale
            smoothed = prev_delta_ori * (1.0 - smooth_alpha) + scaled * smooth_alpha
            
            norm = np.linalg.norm(smoothed)
            if norm > max_ori_step and norm > 1e-12:
                # print(smoothed, 'norm=', norm, 'exceeds max_ori_step=', max_ori_step, '- capping!')
                smoothed = (smoothed / norm) * max_ori_step
                
            delta_ori = smoothed
            prev_delta_ori = smoothed
            
        except Exception as e:
            print('Orientation compute failed:', e)
            delta_ori = np.zeros(3, dtype=np.float32)

    # delta_pos = np.zeros(3, dtype=np.float32)
    delta_pos = np.clip(delta_pos, -1.0, 1.0)
    delta_ori = np.clip(delta_ori, -1.0, 1.0)
    try:
        # Don't forget to give the robot its stiffness muscles!
        # action[:pos_idx] = np.array([0.2, 0.2, 0.2, 0.0, 0.0, 0.0, 0.7, 0.7, 0.7]) 
        # action[:pos_idx] = np.array([0.5, 0.5, 0.5, 0.5, 0.5, 0.5]) 
        action[pos_idx:pos_idx + 3] = delta_pos
        action[pos_idx + 3:pos_idx + 6] = delta_ori 
        print('delta_pos:', delta_pos, 'delta_ori:', delta_ori, 'gripper_act:', gripper_act)
        action[gripper_idx] = float(gripper_act)
    except Exception as e:
        # best-effort fallback
        print('Failed to insert action components, falling back to sampling or zeros.', e)
        try:
            action = env.action_space.sample()
        except Exception:
            pass

    return action, target, tcp_pos, nut_pos, prev_delta_ori

# %%
# Run the scripted primitive and record a short video
out_dir = 'outputs'
os.makedirs(out_dir, exist_ok=True)
out_path = os.path.join(out_dir, 'nutassembly_primitive.mp4')
num_steps = 350
frames = []
obs = env.reset()
action_dim = int(env.action_space.shape[-1])
settle_action = np.zeros(action_dim, dtype=np.float32)
settle_action[-1] = -1.0 # Robosuite OPEN (+1.0)

for _ in range(20):
    try:
        step_out = env.step(settle_action)
        # Keep tracking the observation so the next step has fresh data
        obs = step_out[0] if isinstance(step_out, (tuple, list)) else step_out
    except Exception:
        pass

prev_delta_ori = np.zeros(3, dtype=np.float32)
for t in range(num_steps):
    # Retrieve raw observations from the unwrapped env when possible
    try:
        raw_obs = env.unwrapped._get_observations()
    except Exception:
        raw_obs = obs if isinstance(obs, dict) else None
    phase = float(t) / max(1, num_steps - 1)
    action, target, tcp_pos, nut_pos, prev_delta_ori = scripted_primitive_policy(env, raw_obs, prev_delta_ori, step_phase=phase, quat_debug=False)
    # Step the env with the scripted action
    try:
        obs, rew, terminated, truncated, info = env.step(action)
    except Exception as e:
        print('env.step failed:', e)
        break
    # Optionally print handle debug info occasionally
    if t % 25 == 0:
        try:
            hpos, npos, q = compute_handle_pos(raw_obs)
            print(f'step {t} handle_pos={hpos} nut_pos={npos} quat={q}')
            # print(f'step {t}, phase {phase:.2f}: target={target} tcp_pos={tcp_pos} nut_pos={nut_pos}')
        except Exception:
            pass
    # Capture a visual frame (with fallbacks)
    frame = capture_frame_from_env(env)
    if frame is None and isinstance(obs, dict):
        for k in ('frontview_image', 'frontview_camera', 'camera_front_image', 'frontview'):
            if k in obs:
                f = np.asarray(obs[k])
                if f is not None:
                    frame = f
                    break
    if frame is not None:
        # Some renderers produce upside-down frames; flip vertically if it looks tall
        if frame.ndim == 4:
            frame = frame[0]
        try:
            frame = np.flipud(frame)
        except Exception:
            pass
        frames.append(frame)
    if terminated or truncated:
        break

setattr(env, 'suppress_forced_gripper', False)
# print('after setattr:', getattr(env, 'suppress_forced_gripper', True))

if len(frames) == 0:
    print('No frames captured; enable offscreen rendering or adjust camera names.')
else:
    frames_to_mp4(frames, out_path, fps=30)
    print('Saved video to', out_path)
    display_video(out_path)


# %%
nut_pos=[-0.1108856,  0.17959426, 0.82998947]
nut_quat=[-8.76542435e-08,  5.48543713e-07,  1.57792594e-01,  9.87472277e-01]
local_x = R.from_quat(nut_quat).apply([0.35, 0.0, 0.0])
print('nut_pos=', nut_pos, 'nut_quat=', nut_quat, 'local_x=', local_x)
# print('axis_unit=', axis_unit*0.5)
handle_pos = nut_pos + local_x
print('handle_pos=', handle_pos)

-2.60821301e-02 -0.11166225


