import numpy as np
from gibson2.object_states.aabb import AABB
from gibson2.object_states.adjacency import HorizontalAdjacency, flatten_planes
from gibson2.object_states.kinematics import KinematicsMixin
from gibson2.object_states.object_state_base import BooleanState, RelativeObjectState


class NextTo(KinematicsMixin, RelativeObjectState, BooleanState):
    @staticmethod
    def get_dependencies():
        return KinematicsMixin.get_dependencies() + [HorizontalAdjacency]

    def set_value(self, other, new_value):
        raise NotImplementedError()

    def get_value(self, other):
        objA_states = self.obj.states
        objB_states = other.states

        assert AABB in objA_states
        assert AABB in objB_states

        objA_aabb = objA_states[AABB].get_value()
        objB_aabb = objB_states[AABB].get_value()

        objA_lower, objA_upper = objA_aabb
        objB_lower, objB_upper = objB_aabb
        distance_vec = []
        for dim in range(3):
            glb = max(objA_lower[dim], objB_lower[dim])
            lub = min(objA_upper[dim], objB_upper[dim])
            distance_vec.append(max(0, glb - lub))
        distance = np.linalg.norm(np.array(distance_vec))
        objA_dims = objA_upper - objA_lower
        objB_dims = objB_upper - objB_lower
        avg_aabb_length = np.mean(objA_dims + objB_dims)

        # If the distance is longer than acceptable, return False.
        if distance > avg_aabb_length * (1./6.):
            return False

        # Otherwise, check if the other object shows up in the adjacency list.
        adjacency_this = self.obj.states[HorizontalAdjacency].get_value()
        other_body_id = other.get_body_id()
        in_any_horizontal_adjacency_of_this = any(
            (other_body_id in adjacency_list.positive_neighbors or
             other_body_id in adjacency_list.negative_neighbors)
            for adjacency_list in flatten_planes(adjacency_this)
        )
        if in_any_horizontal_adjacency_of_this:
            return True

        # If not, check in the adjacency lists of `other`. Maybe it's shorter than us etc.
        adjacency_other = other.states[HorizontalAdjacency].get_value()
        this_body_id = self.obj.get_body_id()
        in_any_horizontal_adjacency_of_other = any(
            (this_body_id in adjacency_list.positive_neighbors or
             this_body_id in adjacency_list.negative_neighbors)
            for adjacency_list in flatten_planes(adjacency_other)
        )

        return in_any_horizontal_adjacency_of_other