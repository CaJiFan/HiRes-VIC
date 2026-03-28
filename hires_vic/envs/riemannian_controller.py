import math

import numpy as np
from scipy.spatial.transform import Rotation

import robosuite.utils.transform_utils as T
from robosuite.controllers.parts.controller import Controller
from robosuite.utils.control_utils import *

# Supported impedance modes
IMPEDANCE_MODES = {"fixed", "variable", "variable_kp", "riemannian_kp"}

class RiemannianController(Controller):
    def __init__(
        self,
        sim,
        ref_name,
        joint_indexes,
        actuator_range,
        input_max=1,
        input_min=-1,
        output_max=(0.05, 0.05, 0.05, 0.5, 0.5, 0.5),
        output_min=(-0.05, -0.05, -0.05, -0.5, -0.5, -0.5),
        kp=150,
        damping_ratio=1,
        impedance_mode="fixed",
        kp_limits=(0, 300),
        damping_ratio_limits=(0, 100),
        policy_freq=20,
        position_limits=None,
        orientation_limits=None,
        interpolator_pos=None,
        interpolator_ori=None,
        control_ori=True,
        input_type="delta",
        input_ref_frame="base",
        uncouple_pos_ori=True,
        lite_physics=True,
        **kwargs,  # does nothing; used so no error raised when dict is passed with extra terms used previously
    ):

        super().__init__(
            sim,
            ref_name=ref_name,
            joint_indexes=joint_indexes,
            actuator_range=actuator_range,
            lite_physics=lite_physics,
            part_name=kwargs.get("part_name", None),
            naming_prefix=kwargs.get("naming_prefix", None),
        )
        # Determine whether this is pos ori or just pos
        self.use_ori = control_ori
        # Determine whether we want to use delta or absolute values as inputs
        self.input_type = input_type
        assert self.input_type in ["delta", "absolute"], f"Input type must be delta or absolute, got: {self.input_type}"

        # determine reference frame wrt actions are set
        self.input_ref_frame = input_ref_frame
        assert self.input_ref_frame in [
            "world",
            "base",
        ], f"Input reference frame must be world or base, got: {self.input_ref_frame}"

        # Control dimension
        self.control_dim = 6 if self.use_ori else 3
        self.name_suffix = "POSE" if self.use_ori else "POSITION"

        # input and output max and min (allow for either explicit lists or single numbers)
        self.input_max = self.nums2array(input_max, self.control_dim)
        self.input_min = self.nums2array(input_min, self.control_dim)
        self.output_max = self.nums2array(output_max, self.control_dim)
        self.output_min = self.nums2array(output_min, self.control_dim)

        # kp kd
        self.kp = self.nums2array(kp, 6)
        self.kd = 2 * np.sqrt(self.kp) * damping_ratio

        # kp and kd limits
        self.kp_min = self.nums2array(kp_limits[0], 6)
        self.kp_max = self.nums2array(kp_limits[1], 6)
        self.damping_ratio_min = self.nums2array(damping_ratio_limits[0], 6)
        self.damping_ratio_max = self.nums2array(damping_ratio_limits[1], 6)

        # Verify the proposed impedance mode is supported
        assert impedance_mode in IMPEDANCE_MODES, (
            "Error: Tried to instantiate OSC controller for unsupported "
            "impedance mode! Inputted impedance mode: {}, Supported modes: {}".format(impedance_mode, IMPEDANCE_MODES)
        )

        # Impedance mode
        self.impedance_mode = impedance_mode

        # Add to control dim based on impedance_mode
        if self.impedance_mode == "variable":
            self.control_dim += 12
        elif self.impedance_mode == "variable_kp":
            self.control_dim += 6
        elif self.impedance_mode == "riemannian_kp":
            self.control_dim += 12
        # limits
        self.position_limits = np.array(position_limits) if position_limits is not None else position_limits
        self.orientation_limits = np.array(orientation_limits) if orientation_limits is not None else orientation_limits

        # control frequency
        self.control_freq = policy_freq

        # interpolator
        self.interpolator_pos = interpolator_pos
        self.interpolator_ori = interpolator_ori

        # whether or not pos and ori want to be uncoupled
        self.uncoupling = uncouple_pos_ori

        # initialize goals
        self.goal_pos = None
        self.goal_ori = None

        # initialize orientation references
        self.relative_ori = np.zeros(3)
        self.ori_ref = None

        # initialize origin pos and ori
        self.origin_pos = None
        self.origin_ori = None

        print(f'Riemannian Controller initialized! Using {self.impedance_mode} impedance mode')

    def set_goal(self, action):
        """
        Sets goal based on input @action. If self.impedance_mode is not "fixed", then the input will be parsed into the
        delta values to update the goal position / pose and the kp and/or damping_ratio values to be immediately updated
        internally before executing the proceeding control loop.

        Note that @action expected to be in the following format, based on impedance mode!

            :Mode `'fixed'`: [joint pos command]
            :Mode `'variable'`: [damping_ratio values, kp values, joint pos command]
            :Mode `'variable_kp'`: [kp values, joint pos command]
            :Mode `'riemannian_kp'`: [Mandel notiation trans kp, rot kp, joint pos command]

        Args:
            action (Iterable): Desired relative joint position goal state
        """
        # Update state
        self.update()

        # Parse action based on the impedance mode, and update kp / kd as necessary
        if self.impedance_mode == "variable":
            damping_ratio, kp, goal_update = action[:6], action[6:12], action[12:]
            self.kp = np.clip(kp, self.kp_min, self.kp_max)
            self.kd = 2 * np.sqrt(self.kp) * np.clip(damping_ratio, self.damping_ratio_min, self.damping_ratio_max)
        elif self.impedance_mode == "variable_kp":
            kp, goal_update = action[:6], action[6:]
            self.kp = np.clip(kp, self.kp_min, self.kp_max)
            self.kd = 2 * np.sqrt(self.kp)  # critically damped
        elif self.impedance_mode == "riemannian_kp":
            kp_pos_flat, kp_ori, goal_update = action[:9], action[9:12], action[12:]
            self.kp_pos_matrix = kp_pos_flat.reshape((3, 3))
            # self.kp_pos_matrix = np.clip(self.kp_pos_matrix, self.kp_min, self.kp_max)

            # Ensure absolute symmetry
            kp_sym = (self.kp_pos_matrix + self.kp_pos_matrix.T) / 2.0

            # Fast eigendecomposition (specifically for symmetric/Hermitian matrices)
            eigenvalues, eigenvectors = np.linalg.eigh(kp_sym)
            eigenvalues = np.maximum(eigenvalues, 0.0)

            # Kd = 2 * Q * sqrt(Lambda) * Q^T
            self.kd_pos_matrix = eigenvectors @ np.diag(2.0 * np.sqrt(eigenvalues)) @ eigenvectors.T
            
            self.kp_ori_array = np.clip(kp_ori, self.kp_min[3:6], self.kp_max[3:6])
            self.kd_ori_array = 2.0 * np.sqrt(self.kp_ori_array)
        else:  # This is case "fixed"
            goal_update = action

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

    def world_to_origin_frame(self, vec):
        """
        transform vector from world to reference coordinate frame
        """

        # world rotation matrix is just identity
        world_frame = np.eye(4)
        world_frame[:3, 3] = vec

        origin_frame = T.make_pose(self.origin_pos, self.origin_ori)
        origin_frame_inv = T.pose_inv(origin_frame)
        vec_origin_pose = T.pose_in_A_to_pose_in_B(world_frame, origin_frame_inv)
        vec_origin_pos, _ = T.mat2pose(vec_origin_pose)
        return vec_origin_pos

    def goal_origin_to_eef_pose(self):
        origin_pose = T.make_pose(self.origin_pos, self.origin_ori)
        ee_pose = T.make_pose(self.ref_pos, self.ref_ori_mat)
        origin_pose_inv = T.pose_inv(origin_pose)
        return T.pose_in_A_to_pose_in_B(ee_pose, origin_pose_inv)

    def compute_goal_pos(self, delta, goal_update_mode=None):
        """
        Compute new goal position, given a delta to update. Can either update the new goal based on
        current achieved position or current deisred goal. Updating based on current deisred goal can be useful
        if we want the robot to adhere with a sequence of target poses as closely as possible,
        without lagging or overshooting.

        Args:
            delta (np.array): Desired relative change in position [x, y, z]
            goal_update_mode (str): either "achieved" (achieved position) or "desired" (desired goal)

        Returns:
            np.array: updated goal position in the controller frame
        """
        if goal_update_mode is None:
            goal_update_mode = self._goal_update_mode
        assert goal_update_mode in ["achieved", "desired"]

        if self.goal_pos is None:
            # if goal is not already set, set it to current position (in controller ref frame)
            if self.input_ref_frame == "base":
                self.goal_pos = self.world_to_origin_frame(self.ref_pos)
            elif self.input_ref_frame == "world":
                self.goal_pos = self.ref_pos
            else:
                raise ValueError

        if goal_update_mode == "desired":
            # update new goal wrt current desired goal
            goal_pos = self.goal_pos + delta
        elif goal_update_mode == "achieved":
            # update new goal wrt current achieved position
            if self.input_ref_frame == "base":
                goal_pos = self.world_to_origin_frame(self.ref_pos) + delta
            elif self.input_ref_frame == "world":
                goal_pos = self.ref_pos + delta
            else:
                raise ValueError

        if self.position_limits is not None:
            # to be implemented later
            raise NotImplementedError

        return goal_pos

    def compute_goal_ori(self, delta, goal_update_mode=None):
        """
        Compute new goal orientation, given a delta to update. Can either update the new goal based on
        current achieved position or current deisred goal. Updating based on current deisred goal can be useful
        if we want the robot to adhere with a sequence of target poses as closely as possible,
        without lagging or overshooting.

        Args:
            delta (np.array): Desired relative change in orientation, in axis-angle form [ax, ay, az]
            goal_update_mode (str): either "achieved" (achieved position) or "desired" (desired goal)

        Returns:
            np.array: updated goal orientation in the controller frame
        """
        if goal_update_mode is None:
            goal_update_mode = self._goal_update_mode
        assert goal_update_mode in ["achieved", "desired"]

        if self.goal_ori is None:
            # if goal is not already set, set it to current orientation (in controller ref frame)
            if self.input_ref_frame == "base":
                self.goal_ori = self.goal_origin_to_eef_pose()[:3, :3]
            elif self.input_ref_frame == "world":
                self.goal_ori = self.ref_ori_mat
            else:
                raise ValueError

        # convert axis-angle value to rotation matrix
        quat_error = T.axisangle2quat(delta)
        rotation_mat_error = T.quat2mat(quat_error)

        if self._goal_update_mode == "desired":
            # update new goal wrt current desired goal
            goal_ori = np.dot(rotation_mat_error, self.goal_ori)
        elif self._goal_update_mode == "achieved":
            # update new goal wrt current achieved orientation
            if self.input_ref_frame == "base":
                curr_goal_ori = self.goal_origin_to_eef_pose()[:3, :3]
            elif self.input_ref_frame == "world":
                curr_goal_ori = self.ref_ori_mat
            else:
                raise ValueError
            goal_ori = np.dot(rotation_mat_error, curr_goal_ori)
        else:
            raise ValueError

        # check for orientation limits
        if np.array(self.orientation_limits).any():
            # to be implemented later
            raise NotImplementedError
        return goal_ori

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

        # F_r = kp * pos_err + kd * vel_err
        if self.impedance_mode == "riemannian_kp":
            desired_force = np.dot(self.kp_pos_matrix, position_error) + \
                            np.dot(self.kd_pos_matrix, vel_pos_error)
        else:
            desired_force = np.multiply(np.array(position_error), np.array(self.kp[0:3])) + np.multiply(
                vel_pos_error, self.kd[0:3]
            )

        base_ori_vel = np.array(self.sim.data.get_site_xvelr(f"{self.naming_prefix}{self.part_name}_center"))
        vel_ori_error = -(self.ref_ori_vel - base_ori_vel)

        # Tau_r = kp * ori_err + kd * vel_err
        if self.impedance_mode == "riemannian_kp":
            desired_torque = np.multiply(ori_error, self.kp_ori_array) + \
                             np.multiply(vel_ori_error, self.kd_ori_array)
        else:
            desired_torque = np.multiply(np.array(ori_error), np.array(self.kp[3:6])) + np.multiply(
                vel_ori_error, self.kd[3:6]
            )

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
        super().run_controller()

        return self.torques

    def update_origin(self, origin_pos, origin_ori):
        """
        Optional function to implement in subclass controllers that will take in @origin_pos and @origin_ori and update
        internal configuration to account for changes in the respective states. Useful for controllers in which the origin
        is a frame of reference that is dynamically changing, e.g., adapting the arm to move along with a moving base.

        Args:
            origin_pos (3-tuple): x,y,z position of controller reference in mujoco world coordinates
            origin_ori (np.array): 3x3 rotation matrix orientation of controller reference in mujoco world coordinates
        """
        self.origin_pos = origin_pos
        self.origin_ori = origin_ori

    def update_initial_joints(self, initial_joints):
        # First, update from the superclass method
        super().update_initial_joints(initial_joints)

        # We also need to reset the goal in case the old goals were set to the initial confguration
        self.reset_goal()

    def set_goal_update_mode(self, goal_update_mode):
        self._goal_update_mode = goal_update_mode

    def reset_goal(self, goal_update_mode="achieved"):
        """
        Resets the goal to the current state of the robot.

        Args:
            goal_update_mode (str): set mode for updating controller goals,
                either "achieved" (achieved position) or "desired" (desired goal).
        """
        self.goal_ori = np.array(self.ref_ori_mat)
        self.goal_pos = np.array(self.ref_pos)

        assert goal_update_mode in ["achieved", "desired"]
        self._goal_update_mode = goal_update_mode

        # Also reset interpolators if required

        if self.interpolator_pos is not None:
            self.interpolator_pos.set_goal(self.goal_pos)

        if self.interpolator_ori is not None:
            self.ori_ref = np.array(self.ref_ori_mat)  # reference is the current orientation at start
            self.interpolator_ori.set_goal(
                orientation_error(self.goal_ori, self.ori_ref)
            )  # goal is the total orientation error
            self.relative_ori = np.zeros(3)  # relative orientation always starts at 0

    @property
    def control_limits(self):
        """
        Returns the limits over this controller's action space, overrides the superclass property
        Returns the following (generalized for both high and low limits), based on the impedance mode:

            :Mode `'fixed'`: [joint pos command]
            :Mode `'variable'`: [damping_ratio values, kp values, joint pos command]
            :Mode `'variable_kp'`: [kp values, joint pos command]
            :Mode `'riemannian_kp'`: [Mandel notiation, kp rot, pose command]

        Returns:
            2-tuple:

                - (np.array) minimum action values
                - (np.array) maximum action values
        """
        if self.impedance_mode == "variable":
            low = np.concatenate([self.damping_ratio_min, self.kp_min, self.input_min])
            high = np.concatenate([self.damping_ratio_max, self.kp_max, self.input_max])
        elif self.impedance_mode == "variable_kp":
            low = np.concatenate([self.kp_min, self.input_min])
            high = np.concatenate([self.kp_max, self.input_max])
        elif self.impedance_mode == "riemannian_kp":
            kp_pos_low = np.full(9, -1000.0)
            kp_pos_high = np.full(9, 1000.0)
            
            kp_ori_low = self.kp_min[3:6]
            kp_ori_high = self.kp_max[3:6]
            
            pose_low = self.input_min
            pose_high = self.input_max
            
            low = np.concatenate([kp_pos_low, kp_ori_low, pose_low])
            high = np.concatenate([kp_pos_high, kp_ori_high, pose_high])
        else:  # This is case "fixed"
            low, high = self.input_min, self.input_max
        return low, high

    def delta_to_abs_action(self, delta_ac, goal_update_mode):
        """
        helper function that converts delta action into absolute action
        """
        abs_pos = self.compute_goal_pos(delta_ac[0:3], goal_update_mode=goal_update_mode)
        abs_ori = self.compute_goal_ori(delta_ac[3:6], goal_update_mode=goal_update_mode)
        abs_rot = T.quat2axisangle(T.mat2quat(abs_ori))
        abs_action = np.concatenate([abs_pos, abs_rot])
        return abs_action

    @property
    def name(self):
        return "OSC_" + self.name_suffix
