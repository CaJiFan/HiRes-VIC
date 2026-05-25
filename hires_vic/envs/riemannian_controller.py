import numpy as np
from robosuite.controllers.parts.arm.osc import OperationalSpaceController
from robosuite.utils.control_utils import opspace_matrices, nullspace_torques, orientation_error

class RiemannianController(OperationalSpaceController):
    def __init__(self, impedance_mode="fixed", **kwargs):
        self.is_riemannian = (impedance_mode == "riemannian_kp")
        
        # 1. TRICK THE BASE CLASS: Bypass the hardcoded IMPEDANCE_MODES assertion
        base_mode = "fixed" if self.is_riemannian else impedance_mode
        super().__init__(impedance_mode=base_mode, **kwargs)
        
        # 2. RESTORE OUR CUSTOM MODE
        self.impedance_mode = impedance_mode
        
        if self.is_riemannian:
            # Add the 12 extra dimensions (6 for flat pos matrix + 3 for rot kp)
            self.control_dim += 12
            
            # Initialize placeholder matrices
            self.kp_pos_matrix = np.zeros((3, 3))
            self.kd_pos_matrix = np.zeros((3, 3))
            self.kp_ori_array = np.zeros(3)
            self.kd_ori_array = np.zeros(3)
        # print(f'🔧 Riemannian Controller initialized! Mode: {self.impedance_mode}')

    def set_goal(self, action):
        if self.is_riemannian:
            # 1. Intercept and parse our custom Riemannian action space
            kp_pos_flat, kp_ori, goal_update = action[:9], action[9:12], action[12:]
            
            # 2. Positional Matrix Math (Eigen Decomposition)
            self.kp_pos_matrix = kp_pos_flat.reshape((3, 3))
            kp_sym = (self.kp_pos_matrix + self.kp_pos_matrix.T) / 2.0
            eigenvalues, eigenvectors = np.linalg.eigh(kp_sym)
            eigenvalues = np.maximum(eigenvalues, 1.0)
            self.kd_pos_matrix = eigenvectors @ np.diag(2.0 * np.sqrt(eigenvalues)) @ eigenvectors.T
            
            # 3. Rotational Array Math
            self.kp_ori_array = np.clip(kp_ori, self.kp_min[3:6], self.kp_max[3:6])
            self.kd_ori_array = 2.0 * np.sqrt(self.kp_ori_array)
            
            # 4. TRICK THE BASE CLASS AGAIN: Let super() handle the coordinate translation
            self.impedance_mode = "fixed"
            super().set_goal(goal_update)
            self.impedance_mode = "riemannian_kp"
        else:
            super().set_goal(action)

    @property
    def control_limits(self):
        """Overrides the limits specifically for the Riemannian space"""
        if self.is_riemannian:
            kp_pos_low = np.full(9, -1000.0)
            kp_pos_high = np.full(9, 1000.0)
            kp_ori_low = self.kp_min[3:6]
            kp_ori_high = self.kp_max[3:6]
            pose_low = self.input_min
            pose_high = self.input_max
            
            low = np.concatenate([kp_pos_low, kp_ori_low, pose_low])
            high = np.concatenate([kp_pos_high, kp_ori_high, pose_high])
            return low, high
        else:
            return super().control_limits

    def run_controller(self):
        """
        Why copy this whole method? Because the base OSC controller uses element-wise 
        multiplication (np.multiply) for the stiffness. We MUST override this specific 
        physics calculation to use dense matrix multiplication (np.dot) for the SPD Manifold.
        """
        self.update()

        # --- Goal Extraction (Same as Base OSC) ---
        desired_world_pos = None
        if self.interpolator_pos is not None:
            if self.interpolator_pos.order == 1:
                desired_world_pos = self.interpolator_pos.get_interpolated_goal()
        else:
            if self.input_ref_frame == "base":
                desired_world_pos = self.origin_pos + np.dot(self.origin_ori, self.goal_pos)
            elif self.input_ref_frame == "world":
                desired_world_pos = self.goal_pos

        if self.interpolator_ori is not None:
            self.relative_ori = orientation_error(self.ref_ori_mat, self.ori_ref)
            ori_error = self.interpolator_ori.get_interpolated_goal()
        else:
            if self.input_ref_frame == "base":
                desired_world_ori = np.dot(self.origin_ori, self.goal_ori)
            elif self.input_ref_frame == "world":
                desired_world_ori = self.goal_ori
            ori_error = orientation_error(desired_world_ori, self.ref_ori_mat)

        # --- Error Math ---
        position_error = desired_world_pos - self.ref_pos
        base_pos_vel = np.array(self.sim.data.get_site_xvelp(f"{self.naming_prefix}{self.part_name}_center"))
        vel_pos_error = -(self.ref_pos_vel - base_pos_vel)

        base_ori_vel = np.array(self.sim.data.get_site_xvelr(f"{self.naming_prefix}{self.part_name}_center"))
        vel_ori_error = -(self.ref_ori_vel - base_ori_vel)

        # --- CRITICAL DIFFERENCE: DENSE MATRIX DOT PRODUCT ---
        if self.is_riemannian:
            desired_force = np.dot(self.kp_pos_matrix, position_error) + np.dot(self.kd_pos_matrix, vel_pos_error)
            desired_torque = np.multiply(ori_error, self.kp_ori_array) + np.multiply(vel_ori_error, self.kd_ori_array)
        else:
            desired_force = np.multiply(np.array(position_error), np.array(self.kp[0:3])) + np.multiply(vel_pos_error, self.kd[0:3])
            desired_torque = np.multiply(np.array(ori_error), np.array(self.kp[3:6])) + np.multiply(vel_ori_error, self.kd[3:6])

        # --- Nullspace & Torque Execution (Same as Base OSC) ---
        lambda_full, lambda_pos, lambda_ori, nullspace_matrix = opspace_matrices(
            self.mass_matrix, self.J_full, self.J_pos, self.J_ori
        )

        if self.uncoupling:
            decoupled_force = np.dot(lambda_pos, desired_force)
            decoupled_torque = np.dot(lambda_ori, desired_torque)
            decoupled_wrench = np.concatenate([decoupled_force, decoupled_torque])
        else:
            desired_wrench = np.concatenate([desired_force, desired_torque])
            decoupled_wrench = np.dot(lambda_full, desired_wrench)

        self.torques = np.dot(self.J_full.T, decoupled_wrench) + self.torque_compensation
        self.torques += nullspace_torques(
            self.mass_matrix, nullspace_matrix, self.initial_joint, self.joint_pos, self.joint_vel
        )

        # Note: We do NOT call super().run_controller() here, as we fully replaced its physics.
        # Instead, we mimic the top-level Controller cleanup if needed.
        return self.torques