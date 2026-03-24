from stable_baselines3.common.callbacks import BaseCallback

class RobosuiteLoggingCallback(BaseCallback):
    def __init__(self, verbose=0):
        super().__init__(verbose)

    def _on_step(self) -> bool:
        dones = self.locals.get("dones")
        infos = self.locals.get("infos")
        
        if dones is not None and infos is not None:
            for idx, done in enumerate(dones):
                if done:  # Only log when the episode finishes
                    info = infos[idx]
                    
                    # avg_stiff = info.get("physics/avg_stiffness", -1)
                    # avg_force = info.get("physics/avg_force", -1)
                    # print(f"\n[DEBUG CALLBACK] Episode Finished! "
                        #   f"Avg Stiff: {avg_stiff:.2f} | "
                        #   f"Avg Force: {avg_force:.2f} N\n")
                    
                    # Log Standard Metrics (if available)
                    if "success" in info:
                        self.logger.record("rollout/success_rate", float(info["success"]))
                    
                    # Log New Physics Metrics (from our wrapper)
                    if "physics/avg_stiffness" in info:
                        self.logger.record("physics/avg_stiffness", info["physics/avg_stiffness"])
                    if "physics/avg_force" in info:
                        self.logger.record("physics/avg_force", info["physics/avg_force"])
                    if "physics/max_force_violation_count" in info:
                        self.logger.record("safety/max_force_violations", info["physics/max_force_violation_count"])
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