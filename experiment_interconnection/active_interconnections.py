import torch
from typing import List, Dict
from components.active_interconnection import ActiveInterconnection
from components.estimator import RecursiveEstimator

# ========================================================================================================
    
class Gaze_Fixation_AI(ActiveInterconnection):
    def __init__(self, estimators, device):
        required_estimators = ['PolarTargetPos', 'RobotVel']
        super().__init__(estimators, required_estimators, device)

    def implicit_interconnection_model(self, meas_dict):
        return torch.atleast_1d(meas_dict['RobotVel'][2] - (- meas_dict['RobotVel'][1] / meas_dict['PolarTargetPos'][0]))

class Gaze_Fixation_Relative_AI(ActiveInterconnection):
    def __init__(self, estimators, device):
        required_estimators = ['PolarTargetPos', 'RobotVel']
        super().__init__(estimators, required_estimators, device)

    def implicit_interconnection_model(self, meas_dict):
        offset_angle = meas_dict[f'PolarTargetPos'][1]
        robot_target_frame_rotation_matrix = torch.stack([
            torch.stack([torch.cos(-offset_angle), -torch.sin(-offset_angle)]),
            torch.stack([torch.sin(-offset_angle), torch.cos(-offset_angle)]),
        ]).squeeze()
        robot_target_frame_vel = torch.matmul(robot_target_frame_rotation_matrix, meas_dict['RobotVel'][:2])
        return torch.atleast_1d(robot_target_frame_vel[1] / meas_dict['PolarTargetPos'][0] + meas_dict['RobotVel'][2])
    
class Gaze_Fixation_Constrained_AI(ActiveInterconnection):
    def __init__(self, estimators, device):
        required_estimators = ['PolarTargetPos', 'RobotVel']
        super().__init__(estimators, required_estimators, device)

    def implicit_interconnection_model(self, meas_dict):
        # TODO: expand to more constrained values?
        return torch.stack([
            torch.atleast_1d(meas_dict['RobotVel'][2] - (- meas_dict['RobotVel'][1] / meas_dict['PolarTargetPos'][0])),
            torch.atleast_1d(meas_dict['PolarTargetPos'][1]),
        ]).squeeze()