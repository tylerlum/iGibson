import os
import logging

import ipdb
import numpy as np

import pybullet as p

from igibson.robots.manipulation_robot import ManipulationRobot

from pb_planning.pb_tools.ikfast.franka_panda import ik as franka_panda
import pb_planning

import mtv
from mtv import utils

log = logging.getLogger(__name__)


class Panda(ManipulationRobot):

    def _load(self, simulator):
        """
        Loads this pybullet model into the simulation. Should return a list of unique body IDs corresponding
        to this model.

        :param simulator: Simulator, iGibson simulator reference

        :return Array[int]: List of unique pybullet IDs corresponding to this model. This will usually
            only be a single value
        """
        log.debug("Loading robot model file: {}".format(self.model_file))

        # A persistent reference to simulator is needed for AG in ManipulationRobot
        self.simulator = simulator

        # Set the control frequency if one was not provided.
        expected_control_freq = 1.0 / simulator.render_timestep
        if self.control_freq is None:
            log.debug(
                "Control frequency is None - being set to default of 1 / render_timestep: %.4f", expected_control_freq
            )
            self.control_freq = expected_control_freq
        else:
            assert np.isclose(
                expected_control_freq, self.control_freq
            ), "Stored control frequency does not match environment's render timestep."

        # Set flags for loading model
        flags = p.URDF_USE_MATERIAL_COLORS_FROM_MTL
        if self.self_collision:
            flags = flags | p.URDF_USE_SELF_COLLISION | p.URDF_USE_SELF_COLLISION_EXCLUDE_PARENT

        # Run some sanity checks and load the model
        model_file_type = self.model_file.split(".")[-1]
        if model_file_type == "urdf":
            self.model_type = "URDF"
            body_ids = (p.loadURDF(self.model_file, globalScaling=self.scale, flags=flags, useFixedBase=True),)
        else:
            self.model_type = "MJCF"
            assert self.scale == 1.0, (
                "robot scale must be 1.0 because pybullet does not support scaling " "for MJCF model (p.loadMJCF)"
            )
            body_ids = p.loadMJCF(self.model_file, flags=flags)

        # Load into simulator and initialize states
        for body_id in body_ids:
            simulator.load_object_in_renderer(self, body_id, self.class_id, **self._rendering_params)

        self.body_id = body_ids[0]
        self.tool_link_id = utils.link_from_name(self.body_id, 'tool_link')
        self.hand_link_id = utils.link_from_name(self.body_id, 'panda_hand')

        return body_ids

    @property
    def model_name(self):
        return 'Panda'

    @property
    def controller_order(self):
        return f'arm_{self.default_arm}', f'gripper_{self.default_arm}'

    @property
    def default_joint_pos(self):
        return np.array([-np.pi / 2, -0.3, 0., -np.pi / 2., 0., np.pi / 2, 0.8])

    @property
    def model_file(self):
        return os.path.join(pb_planning.get_base_path(), franka_panda.FRANKA_URDF)

    def _create_discrete_action_space(self):
        raise NotImplementedError

    @property
    def eef_link_names(self):
        return {self.default_arm: 'tool_link'}

    @property
    def finger_link_names(self):
        return {self.default_arm: ['panda_leftfinger', 'panda_rightfinger']}

    @property
    def finger_joint_names(self):
        return {self.default_arm: ['panda_finger_joint1', 'panda_finger_joint2']}

    @property
    def arm_control_idx(self):
        return {self.default_arm: np.array([0, 1, 2, 3, 4, 5, 6])}

    @property
    def gripper_control_idx(self):
        return {self.default_arm: np.array([7, 8])}

    def get_camera_eye_target_up(self):
        hand_pos, hand_quat = utils.get_link_pose(self.body_id, self.hand_link_id)
        hand_R = utils.R_from_quat(hand_quat)
        hand_x, hand_y, hand_z = hand_R[:, 0], hand_R[:, 1], hand_R[:, 2]
        eye = hand_pos + 0.01 * hand_z - 0.13 * hand_x
        target = hand_pos + 1. * hand_z + 0.3 * hand_x
        up = -hand_z
        return eye, target, up

    @property
    def eyes(self):
        return None
