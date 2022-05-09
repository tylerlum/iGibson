from igibson import controllers

class PandaIKFastController(controllers.ManipulationController):
    def __init__(
            self,
            base_body_id,
            task_link_id
    ):


    def _command_to_control(self, command, control_dict):

        def panda_ik(self, tool_pose, max_time, tries=1):
            conf = None
            max_time = max_time
            i = 0
            while conf is None and i < tries:
                iterator = ikfast.closest_inverse_kinematics(self.panda, franka_panda.PANDA_INFO, self.panda_tool_link,
                                                             tool_pose, verbose=False, max_time=max_time,
                                                             max_candidates=np.inf, max_distance=np.inf)
                conf = next(iterator, None)
                max_time = max_time * 3
                i += 1
            return conf
