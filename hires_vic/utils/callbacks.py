from stable_baselines3.common.callbacks import BaseCallback
from collections import defaultdict
import numpy as np
import wandb

# Default mode list for the Wipe task (backwards-compatible)
_WIPE_MODES = ["approach", "contact_edge", "wipe", "lift"]


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
        self._modes = modes or _WIPE_MODES
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
