from stable_baselines3.common.callbacks import BaseCallback
from collections import defaultdict
import numpy as np
import wandb
import torch
from scipy.spatial.transform import Rotation as R

class RobosuiteLoggingCallback(BaseCallback):
    """
    SB3 callback that aggregates per-episode LLM mode distributions,
    force-per-mode correlations, and all physics/smoothness metrics
    from the GeometricWrapper info dict.

    Parameters
    ----------
    modes : list[str] | None
        LLM mode names for the current task. Defaults to the Wipe task modes.
        Pass the list from `LLMImpedancePlanner.mode_names` for other tasks.
    """

    def __init__(self, modes: list[str] | None = None, verbose=0):
        super().__init__(verbose)
        self._modes = modes 
        if modes is not None:
            self._mode_to_int = {m: i for i, m in enumerate(self._modes)}

            # Per-env episode accumulators
            self._mode_counts: dict[int, dict[str, int]] = defaultdict(lambda: defaultdict(int))
            self._mode_force:  dict[int, dict[str, list]] = defaultdict(lambda: defaultdict(list))

    def _on_step(self) -> bool:
        dones = self.locals.get("dones")
        infos = self.locals.get("infos")

        if infos is not None:
            for idx, info in enumerate(infos):
                mode = info.get("llm/impedance_mode")
                if mode is not None:
                    self._mode_counts[idx][mode] += 1
                    force = info.get("step/contact_force")
                    if force is not None:
                        self._mode_force[idx][mode].append(float(force))

        if dones is not None and infos is not None:
            for idx, done in enumerate(dones):
                if not done:
                    continue
                info = infos[idx]

                # ── LLM mode distribution ────────────────────────────────────
                if self._modes is not None:
                    counts = self._mode_counts[idx]
                    total_steps = max(sum(counts.values()), 1)
                    mode_pcts = {m: counts.get(m, 0) / total_steps for m in self._modes}

                    for m, pct in mode_pcts.items():
                        self.logger.record(f"llm/pct_{m}", pct)

                    wandb.log({
                        "llm/episode_mode_distribution": wandb.plot.bar(
                            wandb.Table(
                                columns=["mode", "fraction"],
                                data=[[m, pct] for m, pct in mode_pcts.items()]
                            ),
                            "mode", "fraction",
                            title="LLM Mode Distribution (this episode)"
                        ),
                        "llm/dominant_mode_int": self._mode_to_int.get(
                            max(counts, key=counts.get) if counts else self._modes[0], 0
                        ),
                    }, step=self.num_timesteps, commit=False)

                    for m in self._modes:
                        forces = self._mode_force[idx][m]
                        if forces:
                            self.logger.record(f"llm/avg_force_during_{m}", np.mean(forces))

                    # Reset per-episode accumulators
                    self._mode_counts[idx] = defaultdict(int)
                    self._mode_force[idx] = defaultdict(list)

                # ── Standard episode-end physics / smoothness metrics ────────
                if "success" in info:
                    self.logger.record("rollout/success_rate", float(info["success"]))

                _METRIC_KEYS = [
                    ("smoothness/avg_cond_num",           "smoothness/avg_cond_num"),
                    ("smoothness/max_cond_num",           "smoothness/max_cond_num"),
                    ("smoothness/avg_euclidean_jerk",     "smoothness/avg_euclidean_jerk"),
                    ("smoothness/avg_riemannian_jerk",    "smoothness/avg_riemannian_jerk"),
                    ("smoothness/avg_coupling_magnitude", "smoothness/avg_coupling_magnitude"),
                    ("smoothness/max_ang_accel",          "smoothness/max_ang_accel"),
                    ("smoothness/std_force",              "smoothness/std_force"),
                    ("smoothness/avg_force",              "smoothness/avg_force"),
                    ("physics/avg_stiffness",             "physics/avg_stiffness"),
                    ("physics/avg_force",                 "physics/avg_force"),
                    # LLM-specific (present only if using the LLM planner)
                    ("llm/total_queries",                 "llm/total_queries"),
                    ("llm/total_latency_seconds",         "llm/total_latency_seconds"),
                    ("llm/avg_latency_seconds",           "llm/avg_latency_seconds"),
                    # Task-specific (present only for the relevant task)
                    ("physics/raw_wipe_percentage",       "physics/raw_wipe_percentage"),
                    ("physics/insertion_depth",           "physics/insertion_depth"),
                    ("physics/peg_aligned",               "physics/peg_aligned"),
                    # Safety
                    ("physics/max_force_violation_count", "safety/max_force_violations"),
                    ("physics/joint_violation_count",     "safety/joint_violations"),
                    # Per-episode averages
                    ("physics/contact_step_ratio",        "physics/contact_step_ratio"),
                    ("physics/kp_trans_x_avg",            "physics/kp_trans_x_avg"),
                    ("physics/kp_trans_y_avg",            "physics/kp_trans_y_avg"),
                    ("physics/kp_trans_z_avg",            "physics/kp_trans_z_avg"),
                    ("physics/kp_rot_x_avg",              "physics/kp_rot_x_avg"),
                    ("physics/kp_rot_y_avg",              "physics/kp_rot_y_avg"),
                    ("physics/kp_rot_z_avg",              "physics/kp_rot_z_avg"),
                    ("safety/joint_violation",            "safety/joint_violations"),
                ]
                for key, log_key in _METRIC_KEYS:
                    if key in info:
                        self.logger.record(log_key, info[key])

        return True

class VideoRecorderCallback(BaseCallback):
    """Callback for recording a single rollout to wandb at regular intervals.

    Uses a separate `video_env` (single env) to run a deterministic episode
    and upload frames via `wandb.Video`.
    """
    def __init__(self, video_env, eval_freq: int, fps: int = 20, primitive_init: str = "none",
                 primitive_setup_steps: int = 40, primitive_hover_height: float = 0.10,
                 primitive_approach_gain: float = 10.0, primitive_fast_gain: float = 2.0,
                 quat_debug: bool = False,
                 verbose=0):
        super().__init__(verbose=verbose)
        self.video_env = video_env
        self.eval_freq = eval_freq
        self.fps = fps
        self._last_step = 0
        # Primitive visualization settings
        self.primitive_init = primitive_init
        self.primitive_setup_steps = int(primitive_setup_steps)
        self.primitive_hover_height = float(primitive_hover_height)
        self.primitive_approach_gain = float(primitive_approach_gain)
        self.primitive_fast_gain = float(primitive_fast_gain)
        self.quat_debug = bool(quat_debug)
        # Only record one video by default (can be changed later)
        self._recorded_once = False

    def _on_step(self) -> bool:
        # Skip if we already recorded one video
        # if getattr(self, '_recorded_once', False):
        #     return True
        try:
            current = int(self.num_timesteps)
        except Exception:
            return True

        if current - self._last_step >= self.eval_freq:
            self._last_step = current
            try:
                self._record_video(self.num_timesteps)
                # Mark that we've recorded one video (prevent further recordings)
                self._recorded_once = True
            except Exception as e:
                print(f"Video recording failed: {e}")
        return True

    def _record_video(self, global_step: int):
        # Single deterministic episode
        frames = []
        teleport_wrapper = self.video_env.env
        teleport_wrapper.frames.clear()

        reset_out = self.video_env.reset()
        frames += teleport_wrapper.frames

        obs = reset_out[0] if isinstance(reset_out, (tuple, list)) else reset_out
        def _model_obs_from_raw(raw_obs):
            # Prefer wrapper's own flattening when available
            try:
                if hasattr(self.video_env, '_flatten_obs'):
                    return self.video_env._flatten_obs(raw_obs)
            except Exception:
                pass

            # Fallback: try to remove image/camera keys from a dict-like obs
            candidate = raw_obs[0] if isinstance(raw_obs, (tuple, list)) else raw_obs
            if isinstance(candidate, dict):
                filtered = {
                    k: v
                    for k, v in candidate.items()
                    if not any(x in k.lower() for x in ('image', 'camera', 'frontview', 'agentview', 'rgb'))
                }
                try:
                    parts = [np.asarray(v).flatten() for v in filtered.values()]
                    if parts:
                        return np.concatenate(parts).astype(np.float32)
                except Exception:
                    pass

            return candidate
        
        # Optionally run a visualization primitive before the agent episode
        if self.primitive_init and self.primitive_init.lower() in ("scripted", "teleport", "both"):
            # Check the entire wrapper stack to see if the primitive wrapper exists
            env_ptr = self.video_env
            has_primitive_wrapper = False
            while hasattr(env_ptr, 'env'):
                if any(x in type(env_ptr).__name__ for x in ("RobosuiteScriptedPrimitiveWrapper", "RobosuiteTeleportWrapper")):
                    has_primitive_wrapper = True
                    break
                env_ptr = env_ptr.env

            if has_primitive_wrapper:
                print(f"Video env already contains primitive wrapper, skipping manual execution.")
            else:
                obs = self._run_visual_primitive(frames, obs)

        
        done = False
        while not done:
            # Capture frame
            frame = self._capture_frame()
            if frame is not None:
                frames.append(frame)

            # Predict action
            model_obs = _model_obs_from_raw(obs)
            try:
                action, _ = self.model.predict(model_obs, deterministic=True)
            except Exception:
                # Best-effort fallback
                action, _ = self.model.predict(obs, deterministic=True)
            
            # Step environment
            step_out = self.video_env.step(action)
            if len(step_out) == 5:
                obs, _, terminated, truncated, _ = step_out
                done = bool(terminated or truncated)
            elif len(step_out) == 4:
                obs, _, terminated, info = step_out
                done = bool(terminated)
            else:
                obs = step_out
                done = False
            
        # print(f"Recorded video of {len(frames)} frames at step {global_step}.")
        if frames:
            video_array = np.stack(frames, axis=0)
            if video_array.shape[-1] == 4:
                # RGBA -> RGB
                video_array = video_array[..., :3]
            video_array = np.transpose(video_array, (0, 3, 1, 2))
            try:
                wandb.log({"eval/video": wandb.Video(video_array, fps=self.fps, format="mp4")}, step=global_step)
            except Exception as e:
                print(f"WandB video upload failed: {e}")

    def _capture_frame(self):
        """Capture a single RGB frame from the video env (render or raw obs fallback)."""
        frame = None
        try:
            frame = self.video_env.render()
        except Exception:
            frame = None

        if frame is None:
            raw_obs = None
            try:
                raw_obs = self.video_env.env.unwrapped._get_observations()
            except Exception:
                try:
                    raw_obs = self.video_env.unwrapped._get_observations()
                except Exception:
                    raw_obs = None

            if isinstance(raw_obs, dict):
                img_key = None
                for k in ("frontview_image", "agentview_image"):
                    if k in raw_obs:
                        img_key = k
                        break
                if img_key is None:
                    for k in raw_obs.keys():
                        if any(x in k.lower() for x in ("image", "camera", "rgb")):
                            img_key = k
                            break
                if img_key is not None:
                    try:
                        frame = raw_obs[img_key]
                    except Exception:
                        frame = None

        if frame is not None:
            if isinstance(frame, torch.Tensor):
                frame = frame.cpu().numpy()
            frame = np.asarray(frame, dtype=np.uint8)
            if frame.ndim == 4:
                frame = frame[0]
            try:
                frame = np.flipud(frame)
            except Exception:
                pass
        return frame

    def _determine_action_indices(self):
        action_dim = int(self.video_env.action_space.shape[-1])
        # Position slice index matches GeometricWrapper mapping
        pos_idx = 9 if getattr(self.video_env, 'use_spd_manifold', False) else 6
        gripper_idx = max(0, action_dim - 1)
        return action_dim, pos_idx, gripper_idx

    def _get_nut_pose(self, raw_obs, eef_pos=None):
        """Deterministic handle selection: negative local-X offset of 4cm.

        This enforces: handle_pos = nut_pos - R(nut_quat).apply([1,0,0]) * 0.04
        It returns (handle_pos, nut_pos, chosen_quat).
        """
        if not isinstance(raw_obs, dict) or eef_pos is None:
            return None, None, None

        nut_pos = None
        nut_quat = None

        # 1. Extract absolute position and quaternion if available
        for k, v in raw_obs.items():
            if 'nut' in k.lower() and 'pos' in k.lower() and 'to_' not in k.lower():
                nut_pos = np.asarray(v).flatten()[:3]
            if 'nut' in k.lower() and 'quat' in k.lower() and 'to_' not in k.lower():
                nut_quat = np.asarray(v).flatten()[:4]

        # 2. Fallback to relative if absolute not available
        if nut_pos is None:
            for k, v in raw_obs.items():
                if 'nut' in k.lower() and 'to_robot0_eef_pos' in k.lower():
                    nut_pos = eef_pos - np.asarray(v).flatten()[:3]
                if 'nut' in k.lower() and 'to_robot0_eef_quat' in k.lower():
                    nut_quat = np.asarray(v).flatten()[:4]

        if nut_pos is None:
            return None, None, None

        if nut_quat is None:
            # No orientation available; return centroid as fallback
            return nut_pos, nut_pos, None

        try:
            q_raw = np.asarray(nut_quat).flatten()
            if q_raw.size < 4:
                return nut_pos, nut_pos, None

            # Try common orderings and pick the one whose local Z aligns best with world Z
            orders = [q_raw[:4].astype(np.float64), np.array([q_raw[1], q_raw[2], q_raw[3], q_raw[0]], dtype=np.float64)]
            best_idx = 0
            best_zdot = -1.0
            rots = []
            for idx, qq in enumerate(orders):
                nq = qq / (np.linalg.norm(qq) + 1e-12)
                try:
                    rot_c = R.from_quat(nq)
                except Exception:
                    rots.append(None)
                    continue
                zvec = rot_c.apply([0.0, 0.0, 1.0])
                zdot = abs(float(np.dot(zvec, [0.0, 0.0, 1.0])))
                rots.append(rot_c)
                if zdot > best_zdot:
                    best_zdot = zdot
                    best_idx = idx

            rot = rots[best_idx]
            nq = orders[best_idx] / (np.linalg.norm(orders[best_idx]) + 1e-12)

            # Use local +X as the presumed handle axis, enforce negative sign and 4cm offset
            local_x = rot.apply([1.0, 0.0, 0.0])
            axis_unit = local_x / (np.linalg.norm(local_x) + 1e-12)
            handle_pos = nut_pos - (axis_unit * 0.04)

            if getattr(self, 'quat_debug', False):
                print(f"Chosen quat ordering index={best_idx} (0=x,y,z,w,1=w,x,y,z), local_x={axis_unit}, handle_pos={handle_pos}")

            return handle_pos, nut_pos, nq
        except Exception as e:
            if getattr(self, 'quat_debug', False):
                print(f"Deterministic nut handle computation failed: {e}")
            return nut_pos, nut_pos, nut_quat

    def _get_peg_pos(self):
        """Bypass the observation dict and ask MuJoCo directly where the peg is."""
        try:
            # Drill down through the wrappers to the core Robosuite env
            sim = self.video_env.unwrapped.sim
            
            # Change this to "RoundPeg" if your task requires the round one!
            peg_id = sim.model.body_name2id("peg1") #peg 2 for round peg
            peg_pos = np.array(sim.data.body_xpos[peg_id])

            peg_top_pos = peg_pos + np.array([0.0, 0.0, 0.08])
            return peg_top_pos
        except Exception as e:
            print(f"MuJoCo Peg lookup failed: {e}")
            return None

    def _run_visual_primitive(self, frames, initial_obs):
        prim = (self.primitive_init or '').lower()
        f = self._capture_frame()
        if f is not None: frames.append(f)

        raw_obs = None
        try: raw_obs = self.video_env.unwrapped._get_observations()
        except Exception:
            try: raw_obs = self.video_env.env.unwrapped._get_observations()
            except Exception: raw_obs = None

        eef_pos = np.asarray(raw_obs.get('robot0_eef_pos')).flatten() if isinstance(raw_obs, dict) and 'robot0_eef_pos' in raw_obs else None

        # --- GEOMETRY UPDATES ---
        handle_pos, centroid_pos, nut_quat = self._get_nut_pose(raw_obs, eef_pos)
        if handle_pos is None:
            print("Could not locate Nut!")
            return initial_obs
        # Debug: print both quaternion orderings and their local axes so we can verify conventions
        if getattr(self, 'quat_debug', False):
            raw_nut_q = None
            if isinstance(raw_obs, dict):
                for k, v in raw_obs.items():
                    if 'nut' in k.lower() and 'quat' in k.lower() and 'to_' not in k.lower():
                        try:
                            raw_nut_q = np.asarray(v).flatten()
                        except Exception:
                            raw_nut_q = None
                        break
            if raw_nut_q is not None and raw_nut_q.size >= 4:
                orders = [raw_nut_q[:4], np.array([raw_nut_q[1], raw_nut_q[2], raw_nut_q[3], raw_nut_q[0]])]
                labels = ['x,y,z,w', 'w,x,y,z']
                for label, qq in zip(labels, orders):
                    try:
                        nq = qq / (np.linalg.norm(qq) + 1e-12)
                        rot = R.from_quat(nq)
                        lx = rot.apply([1.0, 0.0, 0.0])
                        ly = rot.apply([0.0, 1.0, 0.0])
                        lz = rot.apply([0.0, 0.0, 1.0])
                        print(f"quat_order={label} local_x={lx} local_y={ly} local_z={lz}")
                    except Exception as e:
                        print(f"quat debug failed for order {label}: {e}")
            else:
                print("quat_debug: raw nut quaternion not found in obs")

        peg_pos = self._get_peg_pos()
        print(f"Primitive targets - Handle: {handle_pos}, Peg: {peg_pos}, Centroid fallback: {centroid_pos}")
        if peg_pos is None:
            # Fallback to hovering over the nut's original starting spot
            peg_pos = centroid_pos

        action_dim, pos_idx, gripper_idx = self._determine_action_indices()
        steps = self.primitive_setup_steps
        current_obs = initial_obs

        # Temporarily tell the env not to force-close the gripper so our scripted
        # primitive can open/close it as intended.
        try:
            setattr(self.video_env, 'suppress_forced_gripper', True)
        except Exception:
            pass
        
        # EXPLICIT GRIPPER VARIABLES (Based on your docs check)
        OPEN = -1.0
        CLOSE = 1.0
        steps = 100

        # Orientation control helpers: smoothing, scaling, and safety limits
        prev_delta_ori = np.zeros(3, dtype=np.float32)
        ori_scale = 0.18  # scale from rotation-vector to action range (reduced)
        smooth_alpha = 0.5
        max_ori_step = 0.08  # radians per step max (tighter)
        apply_ori_dist = 0.12  # only apply orientation corrections when within this distance to handle (increased)
        for step in range(steps):
            try: current_raw = self.video_env.unwrapped._get_observations()
            except Exception: current_raw = raw_obs
            tcp_pos = np.asarray(current_raw.get('robot0_eef_pos', eef_pos)) if current_raw is not None else eef_pos
            tcp_quat = np.asarray(current_raw.get('robot0_eef_quat')) # [x, y, z, w]
            
            # Target 1: The Handle (to grab it)
            grasp_target = handle_pos + np.array([0.0, 0.0, 0.005]) # Tiny Z-offset for fingers

            mid_target = handle_pos + (peg_pos - handle_pos) * 0.5 + np.array([0.0, 0.0, self.primitive_hover_height*2]) # Midpoint between handle and peg, but hovering above
            # Target 2: Hover over the PEG (for the final handover)
            hover_target = peg_pos + np.array([0.0, 0.0, self.primitive_hover_height])

            if prim == 'teleport':
                if step < steps - 20:
                    target = grasp_target
                    gripper_act = OPEN 
                elif step < steps - 5:
                    target = grasp_target
                    gripper_act = CLOSE 
                else:
                    target = hover_target
                    gripper_act = CLOSE 
                # delta_pos = (target - tcp_pos) * self.primitive_fast_gain
            else:
                # Scripted
                phase = step / max(1, steps)
                
                # if phase < 0.33:
                #     target = grasp_target + np.array([0.0, 0.0, self.primitive_hover_height]) # Hover over handle
                #     gripper_act = OPEN 
                if phase < 0.33:
                    target = grasp_target # Descend to handle
                    gripper_act = OPEN 
                elif phase < 0.50:
                    target = grasp_target # Squeeze handle
                    gripper_act = CLOSE 
                elif phase < 0.70:
                    target = mid_target # Lift and move halfway to PEG
                    gripper_act = CLOSE
                else:
                    target = hover_target # Lift and move to PEG
                    gripper_act = CLOSE 

            
            delta_pos = (target - tcp_pos) * (self.primitive_approach_gain * 0.2)
            # 2. ORIENTATION (The New Wrist Twist!)
            delta_ori = np.zeros(3)
            if nut_quat is not None and tcp_quat is not None:
                # Compute a small rotation-vector (axis*angle) in the gripper's local frame
                # that rotates the gripper's local X axis onto the nut's local X axis.
                try:
                    r_current = R.from_quat(tcp_quat)
                    r_nut = R.from_quat(nut_quat)
                    nut_x_vec = r_nut.apply([1.0, 0.0, 0.0])
                    gripper_x_vec = r_current.apply([1.0, 0.0, 0.0])

                    # unit vectors
                    g = gripper_x_vec / (np.linalg.norm(gripper_x_vec) + 1e-12)
                    nvec = nut_x_vec / (np.linalg.norm(nut_x_vec) + 1e-12)
                    dot = np.clip(np.dot(g, nvec), -1.0, 1.0)
                    angle = float(np.arccos(dot))

                    if angle < 1e-3:
                        new_delta_ori_local = np.zeros(3, dtype=np.float32)
                    else:
                        axis_world = np.cross(g, nvec)
                        axis_norm = np.linalg.norm(axis_world)
                        if axis_norm < 1e-6:
                            # Parallel or opposite: pick an arbitrary perpendicular axis
                            axis_world = np.array([0.0, 0.0, 1.0])
                        else:
                            axis_world = axis_world / axis_norm

                        # Express axis in gripper (local) frame
                        axis_local = r_current.inv().apply(axis_world)
                        new_delta_ori_local = (axis_local * angle).astype(np.float32)

                    # Only apply orientation correction when sufficiently close to the handle
                    apply_ori = True
                    try:
                        if tcp_pos is not None and handle_pos is not None:
                            apply_ori = (np.linalg.norm(tcp_pos - handle_pos) <= apply_ori_dist)
                    except Exception:
                        apply_ori = True

                    scaled = new_delta_ori_local * ori_scale
                    if not apply_ori:
                        delta_ori = np.zeros(3, dtype=np.float32)
                    else:
                        # scale, smooth and cap the rotation-vector
                        smoothed = prev_delta_ori * (1.0 - smooth_alpha) + scaled * smooth_alpha
                        norm = np.linalg.norm(smoothed)
                        if norm > max_ori_step and norm > 1e-12:
                            smoothed = (smoothed / norm) * max_ori_step
                        delta_ori = smoothed
                        prev_delta_ori = smoothed

                    if getattr(self, 'quat_debug', False):
                        print(f"angle={angle:.3f}, new_delta_local={new_delta_ori_local}, scaled={scaled}, applied={apply_ori}")
                except Exception as e:
                    if getattr(self, 'quat_debug', False):
                        print(f"Orientation correction failed: {e}")
                    delta_ori = np.zeros(3, dtype=np.float32)
                
            delta_pos = np.clip(delta_pos, -1.0, 1.0)
            delta_ori = np.clip(delta_ori, -1.0, 1.0)

            if getattr(self, 'quat_debug', False):
                print(f"delta_ori={delta_ori * ori_scale}")

            scripted_action = np.zeros((action_dim,), dtype=np.float32)
            scripted_action[pos_idx:pos_idx + 3] = delta_pos
            scripted_action[pos_idx + 3:pos_idx + 6] = delta_ori * ori_scale
            scripted_action[gripper_idx] = float(gripper_act)

            # print('action', scripted_action)

            try:
                step_out = self.video_env.step(scripted_action)
                current_obs = step_out[0] if isinstance(step_out, (tuple, list)) else step_out
            except Exception: pass

            f = self._capture_frame()
            if f is not None: frames.append(f)

        # Clear suppression (best-effort)
        try:
            setattr(self.video_env, 'suppress_forced_gripper', False)
        except Exception:
            pass

        return current_obs

