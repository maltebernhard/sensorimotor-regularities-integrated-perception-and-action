import torch
from typing import List, Dict
from components.active_interconnection import ActiveInterconnection
from components.estimator import RecursiveEstimator

# ========================================================================================================

class Triangulation_AI(ActiveInterconnection):
    def __init__(self, object_name:str="Target") -> None:
        self.object_name = object_name
        required_estimators = [f'Polar{object_name}Pos', 'RobotState']
        super().__init__(required_estimators)

    def implicit_interconnection_model(self, meas_dict: Dict[str, torch.Tensor]):
        offset_angle = meas_dict[f'Polar{self.object_name}Pos'][1]
        robot_target_frame_rotation_matrix = torch.stack([
            torch.stack([torch.cos(-offset_angle), -torch.sin(-offset_angle)]),
            torch.stack([torch.sin(-offset_angle), torch.cos(-offset_angle)]),
        ]).squeeze()
        robot_target_frame_vel = torch.matmul(robot_target_frame_rotation_matrix, meas_dict['RobotState'][:2])
        angular_vel = meas_dict[f'Polar{self.object_name}Pos'][3] + meas_dict['RobotState'][2]
        # TODO: tan or plain?
        #triangulated_distance = torch.abs(robot_target_frame_vel[1] / torch.tan(angular_vel))
        if torch.abs(robot_target_frame_vel[1]) == 0.0 or torch.abs(angular_vel) == 0.0:
            triangulated_distance = meas_dict[f'Polar{self.object_name}Pos'][0]
        else:
            triangulated_distance = torch.abs(robot_target_frame_vel[1] / angular_vel)

        print("Robot Target Frame Vel: ", robot_target_frame_vel[1])
        print(f"Angular Vel: {angular_vel.item()} = {meas_dict[f'Polar{self.object_name}Pos'][3].item()} + {meas_dict['RobotState'][2].item()}")
        print("Triangulated Distance: ", triangulated_distance.item())

        return torch.stack([
            torch.atleast_1d(triangulated_distance - meas_dict[f'Polar{self.object_name}Pos'][0]),
            torch.atleast_1d(- robot_target_frame_vel[0] - meas_dict[f'Polar{self.object_name}Pos'][2]),
        ]).squeeze()
    
class DistanceUpdaterAcc(ActiveInterconnection):
    def __init__(self):
        required_estimators = ['PolarTargetPos', 'RobotState']
        super().__init__(required_estimators)

    def implicit_interconnection_model(self, meas_dict):
        return torch.atleast_1d(meas_dict['PolarTargetPos'][0] - meas_dict['RobotState'][6])
    
class DistanceUpdaterVel(ActiveInterconnection):
    def __init__(self):
        required_estimators = ['PolarTargetPos', 'RobotState']
        super().__init__(required_estimators)

    def implicit_interconnection_model(self, meas_dict):
        return torch.atleast_1d(meas_dict['PolarTargetPos'][0] - meas_dict['RobotState'][3])