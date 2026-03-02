import numpy as np
import scipy.linalg
from robosuite.utils.control_utils import orientation_error, opspace_matrices, nullspace_torques
from robosuite.controllers.parts.arm.osc import OperationalSpaceController
from scipy.spatial.transform import Rotation


class GRL_OperationalSpaceController(OperationalSpaceController):
    def __init__(self, **kwargs):
        # Initialize the base class
        super().__init__(**kwargs)
        
        # Override the control dimension so Robosuite's CompositeController 
        # routes exactly 18 elements to the arm, rather than 6.
        self.control_dim = 18

    @property
    def control_limits(self):
        """
        Because we changed control_dim to 18, we must also return an 18D boundary 
        array to satisfy Robosuite's action_spec builder, otherwise it will crash.
        """
        # 1. Dummy limits for the 9 matrix elements (SB3 restricts them to [-1, 1] before the Exp map anyway)
        kp_pos_low = np.full(9, -1000.0)
        kp_pos_high = np.full(9, 1000.0)
        
        # 2. Extract the rotational stiffness limits (indices 3, 4, 5 of kp_min/max)
        kp_ori_low = self.kp_min[3:6]
        kp_ori_high = self.kp_max[3:6]
        
        # 3. Pose limits (Pos and Ori deltas)
        pose_low = self.input_min
        pose_high = self.input_max
        
        # Concatenate them all into a single 18D array
        low = np.concatenate([kp_pos_low, kp_ori_low, pose_low])
        high = np.concatenate([kp_pos_high, kp_ori_high, pose_high])
        
        return low, high

    def set_goal(self, action):
        """
        Sets goal based on input @action. If self.impedance_mode is not "fixed", then the input will be parsed into the
        delta values to update the goal position / pose and the kp and/or damping_ratio values to be immediately updated
        internally before executing the proceeding control loop.

        Note that @action expected to be in the following format, based on impedance mode!

            :Mode `'fixed'`: [joint pos command]
            :Mode `'variable'`: [damping_ratio values, kp values, joint pos command]
            :Mode `'variable_kp'`: [kp values, joint pos command]

        Args:
            action (Iterable): Desired relative joint position goal state
        """
        # Update state
        self.update()
        # 1. Parse your custom action array
        # Assuming action layout: [0:9] is flattened 3x3 Kp_pos, [9:12] is Kp_ori diagonals, [12:] is pose goal
        kp_pos_flat = action[0:9]
        kp_ori = action[9:12]
        goal_update = action[12:]
        
        # 2. Reconstruct the 3x3 matrix
        self.kp_pos_matrix = kp_pos_flat.reshape((3, 3))
        self.kp_ori_array = np.clip(kp_ori, self.kp_min[3:6], self.kp_max[3:6])
        
        # 3. Calculate Damping Matrix (Kd = 2 * sqrtm(Kp))
        # scipy.linalg.sqrtm computes the matrix square root
        self.kd_pos_matrix = 2.0 * scipy.linalg.sqrtm(self.kp_pos_matrix).real
        self.kd_ori_array = 2.0 * np.sqrt(self.kp_ori_array)

        # If we're using deltas, interpret actions as such
        if self.input_type == "delta":
            delta = goal_update
            scaled_delta = self.scale_action(delta)
            self.goal_pos = self.compute_goal_pos(scaled_delta[0:3])
            if self.use_ori is True:
                self.goal_ori = self.compute_goal_ori(scaled_delta[3:6])
            else:
                self.goal_ori = self.compute_goal_ori(np.zeros(3))
        # Else, interpret actions as absolute values
        elif self.input_type == "absolute":
            abs_action = goal_update
            self.goal_pos = abs_action[0:3]
            if self.use_ori is True:
                self.goal_ori = Rotation.from_rotvec(abs_action[3:6]).as_matrix()
            else:
                self.goal_ori = self.compute_goal_ori(np.zeros(3))
        else:
            raise ValueError(f"Unsupport input_type {self.input_type}")

        if self.interpolator_pos is not None:
            self.interpolator_pos.set_goal(self.goal_pos)

        if self.interpolator_ori is not None:
            self.ori_ref = np.array(self.ref_ori_mat)  # reference is the current orientation at start
            self.interpolator_ori.set_goal(
                orientation_error(self.goal_ori, self.ori_ref)
            )  # goal is the total orientation error
            self.relative_ori = np.zeros(3)  # relative orientation always starts at 0

    def run_controller(self):
        """
        Calculates the torques required to reach the desired setpoint.

        Executes Operational Space Control (OSC) -- either position only or position and orientation.

        A detailed overview of derivation of OSC equations can be seen at:
        http://khatib.stanford.edu/publications/pdfs/Khatib_1987_RA.pdf

        Returns:
             np.array: Command torques
        """
        # Update state
        self.update()

        desired_world_pos = None
        # Only linear interpolator is currently supported
        if self.interpolator_pos is not None:
            # Linear case
            if self.interpolator_pos.order == 1:
                desired_world_pos = self.interpolator_pos.get_interpolated_goal()
            else:
                # Nonlinear case not currently supported
                pass
        else:
            if self.input_ref_frame == "base":
                # compute goal based on current base position and orientation
                desired_world_pos = self.origin_pos + np.dot(self.origin_ori, self.goal_pos)
            elif self.input_ref_frame == "world":
                desired_world_pos = self.goal_pos
            else:
                raise ValueError

        if self.interpolator_ori is not None:
            # relative orientation based on difference between current ori and ref
            self.relative_ori = orientation_error(self.ref_ori_mat, self.ori_ref)

            ori_error = self.interpolator_ori.get_interpolated_goal()
        else:
            if self.input_ref_frame == "base":
                # compute goal based on current base orientation
                desired_world_ori = np.dot(self.origin_ori, self.goal_ori)
            elif self.input_ref_frame == "world":
                desired_world_ori = self.goal_ori
            else:
                raise ValueError
            ori_error = orientation_error(desired_world_ori, self.ref_ori_mat)

        # Compute desired force and torque based on errors
        position_error = desired_world_pos - self.ref_pos
        base_pos_vel = np.array(self.sim.data.get_site_xvelp(f"{self.naming_prefix}{self.part_name}_center"))
        vel_pos_error = -(self.ref_pos_vel - base_pos_vel)

        # Use np.dot() for matrix multiplication instead of np.multiply()
        desired_force = np.dot(self.kp_pos_matrix, position_error) + \
                        np.dot(self.kd_pos_matrix, vel_pos_error)

        # print(f"Position error: {position_error}, Velocity error: {vel_pos_error}")
        # print(f"Desired force: {desired_force}")
                        
        base_ori_vel = np.array(self.sim.data.get_site_xvelr(f"{self.naming_prefix}{self.part_name}_center"))
        vel_ori_error = -(self.ref_ori_vel - base_ori_vel)

        # Keep orientation standard (element-wise)
        desired_torque = np.multiply(ori_error, self.kp_ori_array) + \
                         np.multiply(vel_ori_error, self.kd_ori_array)


        # Compute nullspace matrix (I - Jbar * J) and lambda matrices ((J * M^-1 * J^T)^-1)
        lambda_full, lambda_pos, lambda_ori, nullspace_matrix = opspace_matrices(
            self.mass_matrix, self.J_full, self.J_pos, self.J_ori
        )

        # Decouples desired positional control from orientation control
        if self.uncoupling:
            decoupled_force = np.dot(lambda_pos, desired_force)
            decoupled_torque = np.dot(lambda_ori, desired_torque)
            decoupled_wrench = np.concatenate([decoupled_force, decoupled_torque])
        else:
            desired_wrench = np.concatenate([desired_force, desired_torque])
            decoupled_wrench = np.dot(lambda_full, desired_wrench)

        # Gamma (without null torques) = J^T * F + gravity compensations
        self.torques = np.dot(self.J_full.T, decoupled_wrench) + self.torque_compensation
        # Calculate and add nullspace torques (nullspace_matrix^T * Gamma_null) to final torques
        # Note: Gamma_null = desired nullspace pose torques, assumed to be positional joint control relative
        #                     to the initial joint positions
        self.torques += nullspace_torques(
            self.mass_matrix, nullspace_matrix, self.initial_joint, self.joint_pos, self.joint_vel
        )

        # Always run superclass call for any cleanups at the end
        # super().run_controller()

        return self.torques
