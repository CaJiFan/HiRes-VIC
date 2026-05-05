from stable_baselines3.common.callbacks import BaseCallback

class RobosuiteLoggingCallback(BaseCallback):
    def __init__(self, verbose=0):
        super().__init__(verbose)

    def _on_step(self) -> bool:
        dones = self.locals.get("dones")
        infos = self.locals.get("infos")

        if infos is not None:
            for idx, info in enumerate(infos):
                # Log LLM mode at every step
                if "llm/impedance_mode" in info:
                    self.logger.record("llm/impedance_mode", info["llm/impedance_mode"])
        
        if dones is not None and infos is not None:
            for idx, done in enumerate(dones):
                if done:  # Only log when the episode finishes
                    info = infos[idx]
                    # Log Standard Metrics (if available)
                    if "success" in info:
                        self.logger.record("rollout/success_rate", float(info["success"]))
                    
                    # if "eval/raw_wipe_percentage" in info:
                    #     self.logger.record("eval/raw_wipe_percentage", info["eval/raw_wipe_percentage"])
                    #     self.logger.record("eval/kp_trans_x_avg", info["eval/kp_trans_x_avg"])
                    #     self.logger.record("eval/kp_trans_y_avg", info["eval/kp_trans_y_avg"])
                    #     self.logger.record("eval/kp_trans_z_avg", info["eval/kp_trans_z_avg"])
                    #     self.logger.record("eval/kp_rot_x_avg", info["eval/kp_rot_x_avg"])
                    #     self.logger.record("eval/kp_rot_y_avg", info["eval/kp_rot_y_avg"])
                    #     self.logger.record("eval/kp_rot_z_avg", info["eval/kp_rot_z_avg"])
                        
                    #     self.logger.record("time/total_timesteps", self.num_timesteps)
                    
                    # Log New Physics Metrics (from our wrapper)
                    if "physics/avg_stiffness" in info:
                        self.logger.record("physics/avg_stiffness", info["physics/avg_stiffness"])
                    if "physics/avg_force" in info:
                        self.logger.record("physics/avg_force", info["physics/avg_force"])
                    if "physics/raw_wipe_percentage" in info:
                        self.logger.record("physics/raw_wipe_percentage", info["physics/raw_wipe_percentage"])
                    if "physics/max_force_violation_count" in info:
                        self.logger.record("safety/max_force_violations", info["physics/max_force_violation_count"])
                    if "physics/joint_violation_count" in info:
                        self.logger.record("safety/joint_violations", info["physics/joint_violation_count"])
                    if "physics/kp_trans_x_avg" in info:
                        self.logger.record("physics/kp_trans_x_avg", info["physics/kp_trans_x_avg"])
                    if "physics/kp_trans_y_avg" in info:
                        self.logger.record("physics/kp_trans_y_avg", info["physics/kp_trans_y_avg"])
                    if "physics/kp_trans_z_avg" in info:
                        self.logger.record("physics/kp_trans_z_avg", info["physics/kp_trans_z_avg"])
                    if "physics/kp_rot_x_avg" in info:
                        self.logger.record("physics/kp_rot_x_avg", info["physics/kp_rot_x_avg"])
                    if "physics/kp_rot_y_avg" in info:
                        self.logger.record("physics/kp_rot_y_avg", info["physics/kp_rot_y_avg"])
                    if "physics/kp_rot_z_avg" in info:
                        self.logger.record("physics/kp_rot_z_avg", info["physics/kp_rot_z_avg"])
                    if "safety/joint_violation" in info:
                        self.logger.record("safety/joint_violations", info["safety/joint_violation"])
                        
        return True