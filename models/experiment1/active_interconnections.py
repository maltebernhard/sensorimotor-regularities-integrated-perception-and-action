from typing import Dict
import torch
from components.measurement_model import ActiveInterconnection
from components.helpers import rotate_vector_2d

# ========================================================================================================
    
class Triangulation_AI(ActiveInterconnection):
    def __init__(self, object_name:str="Target") -> None:
        self.object_name = object_name
        required_estimators = [f'Polar{object_name}Pos', 'RobotVel']
        super().__init__(required_estimators)

    def implicit_interconnection_model(self, meas_dict: Dict[str, torch.Tensor]):
        # NOTE: think about suppressing gradient propagation of changes in offset angle in this AI
            # BAD because it suppresses an action component
            # GOOD because we don't want rotation to influence triangulation
            # not sure whether this suppression will have additional effects other than the desired one
        rtf_vel = rotate_vector_2d(meas_dict[f'Polar{self.object_name}Pos'][1], meas_dict['RobotVel'][:2])
        angular_vel = meas_dict[f'Polar{self.object_name}Pos'][3] + meas_dict['RobotVel'][2]
        if torch.abs(rtf_vel[1]) == 0.0 or torch.abs(angular_vel) == 0.0:
            triangulated_distance = meas_dict[f'Polar{self.object_name}Pos'][0]
        else:
            triangulated_distance = torch.abs(rtf_vel[1] / angular_vel)
        return torch.stack([
            torch.atleast_1d(triangulated_distance - meas_dict[f'Polar{self.object_name}Pos'][0]),
            torch.atleast_1d(- rtf_vel[0] - meas_dict[f'Polar{self.object_name}Pos'][2]),
        ]).squeeze()

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