import torch
from components.active_interconnection import ActiveInterconnection
from components.helpers import rotate_vector_2d

# ========================================================================================================
    
# class Gaze_Fixation_AI(ActiveInterconnection):
#     def __init__(self):
#         required_estimators = ['PolarTargetPos', 'RobotVel']
#         super().__init__(required_estimators)

#     def implicit_interconnection_model(self, meas_dict):
#         return torch.stack([
#             # angular vel should match lateral vel / distance
#             torch.atleast_1d(meas_dict['RobotVel'][2] - (- meas_dict['RobotVel'][1] / meas_dict['PolarTargetPos'][0])),
#             # angle should be 0
#             torch.atleast_1d(meas_dict['PolarTargetPos'][1]),
#         ]).squeeze()
    
# TODO: which properties should be constrained? only angular_vel, or also angle?

class Gaze_Fixation_AI(ActiveInterconnection):
    def __init__(self):
        required_estimators = ['PolarTargetPos', 'RobotVel']
        super().__init__(required_estimators)

    def implicit_interconnection_model(self, meas_dict):
        rtf_vel = rotate_vector_2d(meas_dict['PolarTargetPos'][1], meas_dict['RobotVel'][:2])
        return torch.atleast_1d(rtf_vel[1] / meas_dict['PolarTargetPos'][0] + meas_dict['RobotVel'][2])