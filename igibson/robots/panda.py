import os
import logging

import numpy as np

import pybullet as p
import collections

from igibson.robots.manipulation_robot import ManipulationRobot

from pb_planning.pb_tools.ikfast.franka_panda import ik as franka_panda
import pb_planning

log = logging.getLogger(__name__)


# Moved from utils function to be standalone
CLIENT = 0
BASE_LINK = -1
BodyInfo = collections.namedtuple('BodyInfo', ['base_name', 'body_name'])

JointInfo = collections.namedtuple(
    'JointInfo',
    [
        'joint_index',
        'joint_name',
        'joint_type',
        'q_index',
        'u_index',
        'flags',
        'joint_amping',
        'joint_friction',
        'joint_lower_limit',
        'joint_upper_limit',
        'joint_max_force',
        'joint_max_velocity',
        'link_name',
        'joint_axis',
        'parent_frame_pos',
        'parent_frame_orn',
        'parent_index'
    ]
)


def get_joint_info(body, joint):
    return JointInfo(*p.getJointInfo(body, joint, physicsClientId=CLIENT))

def get_link_name(body, link):
    if link == BASE_LINK:
        return get_base_name(body)
    return get_joint_info(body, link).link_name.decode('UTF-8')

def get_num_joints(body):
    return p.getNumJoints(body, physicsClientId=CLIENT)

def get_joints(body):
    return list(range(get_num_joints(body)))

def get_body_info(body):
    return BodyInfo(*p.getBodyInfo(body, physicsClientId=CLIENT))

def get_base_name(body):
    return get_body_info(body).base_name.decode(encoding='UTF-8')

def link_from_name(body, name):
    if name == get_base_name(body):
        return BASE_LINK
    for link in get_joints(body):
        if get_link_name(body, link) == name:
            return link
    raise ValueError(body, name)


class Panda(ManipulationRobot):

    def __init__(self, camera_pitch_deg, camera_dist_to_target, **kwargs):
        super().__init__(**kwargs)
        self.camera_pitch_deg, self.camera_dist_to_target = camera_pitch_deg, camera_dist_to_target

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
        self.tool_link_id = link_from_name(self.body_id, 'tool_link')
        self.hand_link_id = link_from_name(self.body_id, 'panda_hand')

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

    @property
    def eyes(self):
        return None
