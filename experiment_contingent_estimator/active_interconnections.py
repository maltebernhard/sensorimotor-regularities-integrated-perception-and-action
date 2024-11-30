import torch
from typing import List, Dict
from components.active_interconnection import ActiveInterconnection
from components.estimator import RecursiveEstimator

# ========================================================================================================

class Vel_AI(ActiveInterconnection):
    """
    Measurement Model:
    vel state:          robot vel
    vel obs:            robot vel observation
    """
    def __init__(self, estimators, device) -> None:
        required_estimators = ['RobotState', 'vel_frontal', 'vel_lateral', 'vel_rot']
        super().__init__(estimators, required_estimators, device)

    def implicit_interconnection_model(self, meas_dict):
        return torch.stack([
            meas_dict['vel_frontal'] - meas_dict['RobotState'][0],
            meas_dict['vel_lateral'] - meas_dict['RobotState'][1],
            meas_dict['vel_rot'] - meas_dict['RobotState'][2]
        ]).squeeze()

class Angle_Meas_AI(ActiveInterconnection):
    def __init__(self, estimators: List[RecursiveEstimator], device, object_name:str="Target") -> None:
        self.object_name = object_name
        required_estimators = ['RobotState', 'target_offset_angle', 'del_target_offset_angle']
        super().__init__(estimators, required_estimators, device)

    def implicit_interconnection_model(self, meas_dict: Dict[str, torch.Tensor]):
        return torch.stack([
            meas_dict['del_target_offset_angle'] - meas_dict['RobotState'][2],
        ]).squeeze()

class Triangulation_AI(ActiveInterconnection):
    def __init__(self, estimators: List[RecursiveEstimator], device) -> None:
        required_estimators = ['RobotState']
        super().__init__(estimators, required_estimators, device)

    def implicit_interconnection_model(self, meas_dict: Dict[str, torch.Tensor]):
        offset_angle = meas_dict['RobotState'][1]
        robot_target_frame_rotation_matrix = torch.stack([
            torch.stack([torch.cos(-offset_angle), -torch.sin(-offset_angle)]),
            torch.stack([torch.sin(-offset_angle), torch.cos(-offset_angle)]),
        ]).squeeze()
        robot_target_frame_vel = torch.matmul(robot_target_frame_rotation_matrix, meas_dict['RobotVel'][:2])
        if self.estimate_vel:
            angular_vel = meas_dict[f'Polar{self.object_name}Pos'][3] + meas_dict['RobotVel'][2]
        else:
            angular_vel = meas_dict[f'del_{self.object_name[0].lower() + self.object_name[1:]}_offset_angle'] + meas_dict['RobotVel'][2]
        # TODO: tan or plain?
        #triangulated_distance = torch.abs(robot_target_frame_vel[1] / torch.tan(angular_vel))
        if robot_target_frame_vel[1] == 0 or angular_vel == 0.0:
            triangulated_distance = meas_dict[f'Polar{self.object_name}Pos'][0]
        else:
            triangulated_distance = torch.abs(robot_target_frame_vel[1] / angular_vel)

        if self.estimate_vel:
            return torch.stack([
                torch.atleast_1d(triangulated_distance - meas_dict[f'Polar{self.object_name}Pos'][0]),
                torch.atleast_1d(- robot_target_frame_vel[0] - meas_dict[f'Polar{self.object_name}Pos'][2]),
            ]).squeeze()
        else:
            return torch.atleast_1d(triangulated_distance - meas_dict[f'Polar{self.object_name}Pos'][0])