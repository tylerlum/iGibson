import pybullet as p
from gibson2.utils.mesh_util import perspective, lookat, xyz2mat, quat2rotmat, mat2xyz, \
    safemat2quat, xyzw2wxyz, ortho, transform_vertex
import numpy as np


class InstanceGroup(object):
    """
    InstanceGroup is a set of visual objects, it is grouped together because they are kinematically connected.
    Robots and articulated objects are represented as instance groups.
    """

    def __init__(self,
                 objects,
                 id,
                 link_ids,
                 pybullet_uuid,
                 class_id,
                 poses_trans,
                 poses_rot,
                 dynamic,
                 robot=None,
                 use_pbr=True,
                 use_pbr_mapping=True,
                 shadow_caster=True
                 ):
        """
        :param objects: visual objects
        :param id: id this instance_group
        :param link_ids: link_ids in pybullet
        :param pybullet_uuid: body id in pybullet
        :param class_id: class_id to render semantics
        :param poses_trans: initial translations for each visual object
        :param poses_rot: initial rotation matrix for each visual object
        :param dynamic: is the instance group dynamic or not
        :param robot: The robot associated with this InstanceGroup
        """
        # assert(len(objects) > 0) # no empty instance group
        self.objects = objects
        self.poses_trans = poses_trans
        self.poses_rot = poses_rot
        self.id = id
        self.link_ids = link_ids
        self.class_id = class_id
        self.robot = robot
        if len(objects) > 0:
            self.renderer = objects[0].renderer
        else:
            self.renderer = None

        self.pybullet_uuid = pybullet_uuid
        self.dynamic = dynamic
        self.tf_tree = None
        self.use_pbr = use_pbr
        self.use_pbr_mapping = use_pbr_mapping
        self.shadow_caster = shadow_caster
        self.roughness = 1
        self.metalness = 0
        # Determines whether object will be rendered
        self.hidden = False
        # Indices into optimized buffers such as color information and transformation buffer
        # These values are used to set buffer information during simulation
        self.or_buffer_indices = None

    def render(self, shadow_pass=0):
        """
        Render this instance group
        """
        if self.renderer is None:
            return

        self.renderer.r.initvar(self.renderer.shaderProgram,
                                self.renderer.V,
                                self.renderer.lightV,
                                shadow_pass,
                                self.renderer.P,
                                self.renderer.lightP,
                                self.renderer.camera,
                                self.renderer.lightpos,
                                self.renderer.lightcolor)

        for i, visual_object in enumerate(self.objects):
            for object_idx in visual_object.VAO_ids:
                self.renderer.r.init_pos_instance(self.renderer.shaderProgram,
                                                  self.poses_trans[i],
                                                  self.poses_rot[i])
                current_material = self.renderer.materials_mapping[self.renderer.mesh_materials[object_idx]]
                self.renderer.r.init_material_instance(self.renderer.shaderProgram,
                                                       float(
                                                           self.class_id) / 255.0,
                                                       current_material.kd,
                                                       float(current_material.is_texture()),
                                                       float(self.use_pbr),
                                                       float(self.use_pbr_mapping),
                                                       float(self.metalness),
                                                       float(self.roughness),
                                                       current_material.transform_param
                                                       )

                try:
                    texture_id = current_material.texture_id
                    metallic_texture_id = current_material.metallic_texture_id
                    roughness_texture_id = current_material.roughness_texture_id
                    normal_texture_id = current_material.normal_texture_id

                    if texture_id is None:
                        texture_id = -1
                    if metallic_texture_id is None:
                        metallic_texture_id = -1
                    if roughness_texture_id is None:
                        roughness_texture_id = -1
                    if normal_texture_id is None:
                        normal_texture_id = -1

                    if self.renderer.msaa:
                        buffer = self.renderer.fbo_ms
                    else:
                        buffer = self.renderer.fbo
                    self.renderer.r.draw_elements_instance(
                        self.renderer.materials_mapping[self.renderer.mesh_materials[object_idx]].is_texture(),
                        texture_id,
                        metallic_texture_id,
                        roughness_texture_id,
                        normal_texture_id,
                        self.renderer.depth_tex_shadow,
                        self.renderer.VAOs[object_idx],
                        self.renderer.faces[object_idx].size,
                        self.renderer.faces[object_idx],
                        buffer)
                finally:
                    self.renderer.r.cglBindVertexArray(0)
        self.renderer.r.cglUseProgram(0)

    def get_pose_in_camera(self):
        mat = self.renderer.V.dot(self.pose_trans.T).dot(self.pose_rot).T
        pose = np.concatenate([mat2xyz(mat), safemat2quat(mat[:3, :3].T)])
        return pose

    def set_position(self, pos):
        """
        Set positions for each part of this InstanceGroup

        :param pos: New translations
        """

        self.pose_trans = np.ascontiguousarray(xyz2mat(pos))

    def set_rotation(self, quat):
        """
        Set rotations for each part of this InstanceGroup

        :param quat: New quaternion in w,x,y,z
        """

        self.pose_rot = np.ascontiguousarray(quat2rotmat(quat))

    def dump(self):
        vertices_info = []
        faces_info = []
        for i, visual_obj in enumerate(self.objects):
            for vertex_data_index, face_data_index in zip(visual_obj.vertex_data_indices, visual_obj.face_indices):
                vertices_info.append(transform_vertex(self.renderer.vertex_data[vertex_data_index],
                                                      pose_trans=self.poses_trans[i],
                                                      pose_rot=self.poses_rot[i]))
                faces_info.append(self.renderer.faces[face_data_index])
        return vertices_info, faces_info

    def __str__(self):
        return "InstanceGroup({}) -> Objects({})".format(
            self.id, ",".join([str(object.id) for object in self.objects]))

    def __repr__(self):
        return self.__str__()


class Robot(InstanceGroup):
    def __init__(self, *args, **kwargs):
        super(Robot, self).__init__(*args, **kwargs)

    def __str__(self):
        return "Robot({}) -> Objects({})".format(
            self.id, ",".join([str(object.id) for object in self.objects]))


class Instance(object):
    """
    Instance is one instance of a visual object. One visual object can have multiple instances to save memory.
    """

    def __init__(self, object, id, class_id, pybullet_uuid, pose_trans, pose_rot, dynamic, softbody,
                 use_pbr=True,
                 use_pbr_mapping=True,
                 shadow_caster=True
                 ):
        self.object = object
        self.pose_trans = pose_trans
        self.pose_rot = pose_rot
        self.id = id
        self.class_id = class_id
        self.renderer = object.renderer
        self.pybullet_uuid = pybullet_uuid
        self.dynamic = dynamic
        self.softbody = softbody
        self.use_pbr = use_pbr
        self.use_pbr_mapping = use_pbr_mapping
        self.shadow_caster = shadow_caster
        self.roughness = 1
        self.metalness = 0
        # Determines whether object will be rendered
        self.hidden = False
        # Indices into optimized buffers such as color information and transformation buffer
        # These values are used to set buffer information during simulation
        self.or_buffer_indices = None

    def render(self, shadow_pass=0):
        """
        Render this instance
        shadow_pass = 0: normal rendering mode, disable shadow
        shadow_pass = 1: enable_shadow, rendering depth map from light space
        shadow_pass = 2: use rendered depth map to calculate shadow
        """
        if self.renderer is None:
            return

        # softbody: reload vertex position
        if self.softbody:
            # construct new vertex position into shape format
            object_idx = self.object.VAO_ids[0]
            vertices = p.getMeshData(self.pybullet_uuid)[1]
            vertices_flattened = [
                item for sublist in vertices for item in sublist]
            vertex_position = np.array(vertices_flattened).reshape(
                (len(vertices_flattened) // 3, 3))
            shape = self.renderer.shapes[object_idx]
            n_indices = len(shape.mesh.indices)
            np_indices = shape.mesh.numpy_indices().reshape((n_indices, 3))
            shape_vertex_index = np_indices[:, 0]
            shape_vertex = vertex_position[shape_vertex_index]

            # update new vertex position in buffer data
            new_data = self.renderer.vertex_data[object_idx]
            new_data[:, 0:shape_vertex.shape[1]] = shape_vertex
            new_data = new_data.astype(np.float32)

            # transform and rotation already included in mesh data
            self.pose_trans = np.eye(4)
            self.pose_rot = np.eye(4)

            # update buffer data into VBO
            self.renderer.r.render_softbody_instance(
                self.renderer.VAOs[object_idx], self.renderer.VBOs[object_idx], new_data)

        self.renderer.r.initvar(self.renderer.shaderProgram,
                                self.renderer.V,
                                self.renderer.lightV,
                                shadow_pass,
                                self.renderer.P,
                                self.renderer.lightP,
                                self.renderer.camera,
                                self.renderer.lightpos,
                                self.renderer.lightcolor)

        self.renderer.r.init_pos_instance(self.renderer.shaderProgram,
                                          self.pose_trans,
                                          self.pose_rot)

        for object_idx in self.object.VAO_ids:
            current_material = self.renderer.materials_mapping[
                self.renderer.mesh_materials[object_idx]]
            self.renderer.r.init_material_instance(self.renderer.shaderProgram,
                                                   float(self.class_id) / 255.0,
                                                   current_material.kd,
                                                   float(current_material.is_texture()),
                                                   float(self.use_pbr),
                                                   float(self.use_pbr_mapping),
                                                   float(self.metalness),
                                                   float(self.roughness),
                                                   current_material.transform_param)
            try:

                texture_id = current_material.texture_id
                metallic_texture_id = current_material.metallic_texture_id
                roughness_texture_id = current_material.roughness_texture_id
                normal_texture_id = current_material.normal_texture_id

                if texture_id is None:
                    texture_id = -1
                if metallic_texture_id is None:
                    metallic_texture_id = -1
                if roughness_texture_id is None:
                    roughness_texture_id = -1
                if normal_texture_id is None:
                    normal_texture_id = -1

                if self.renderer.msaa:
                    buffer = self.renderer.fbo_ms
                else:
                    buffer = self.renderer.fbo

                self.renderer.r.draw_elements_instance(
                    self.renderer.materials_mapping[self.renderer.mesh_materials[object_idx]].is_texture(),
                    texture_id,
                    metallic_texture_id,
                    roughness_texture_id,
                    normal_texture_id,
                    self.renderer.depth_tex_shadow,
                    self.renderer.VAOs[object_idx],
                    self.renderer.faces[object_idx].size,
                    self.renderer.faces[object_idx],
                    buffer)
            finally:
                self.renderer.r.cglBindVertexArray(0)

        self.renderer.r.cglUseProgram(0)

    def get_pose_in_camera(self):
        mat = self.renderer.V.dot(self.pose_trans.T).dot(self.pose_rot).T
        pose = np.concatenate([mat2xyz(mat), safemat2quat(mat[:3, :3].T)])
        return pose

    def set_position(self, pos):
        self.pose_trans = np.ascontiguousarray(xyz2mat(pos))

    def set_rotation(self, quat):
        """
        :param quat: New quaternion in w,x,y,z
        """
        self.pose_rot = np.ascontiguousarray(quat2rotmat(quat))

    def dump(self):
        vertices_info = []
        faces_info = []
        for vertex_data_index, face_index in zip(self.object.vertex_data_indices, self.object.face_indices):
            vertices_info.append(transform_vertex(self.renderer.vertex_data[vertex_data_index],
                                                  pose_rot=self.pose_rot,
                                                  pose_trans=self.pose_trans))
            faces_info.append(self.renderer.faces[face_index])
        return vertices_info, faces_info

    def __str__(self):
        return "Instance({}) -> Object({})".format(self.id, self.object.id)

    def __repr__(self):
        return self.__str__()
