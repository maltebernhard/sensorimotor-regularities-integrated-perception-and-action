import torch
from typing import List, Dict
from components.active_interconnection import ActiveInterconnection
from components.estimator import RecursiveEstimator

# ========================================================================================================

class Triangulation_AI(ActiveInterconnection):
    def __init__(self, device, object_name:str="Target") -> None:
        self.object_name = object_name
        required_estimators = [f'Polar{object_name}Pos', 'RobotVel']
        super().__init__(required_estimators, device)

    def implicit_interconnection_model(self, meas_dict: Dict[str, torch.Tensor]):
        # TODO: suppresses gradient propagation of changes in offset angle in this AI
            # BAD because it suppresses an action component
            # GOOD because we don't want rotation to influence triangulation
            # NOTE: not sure whether this suppression will have additional effects other than the desired one
        #offset_angle = torch.tensor([meas_dict[f'Polar{self.object_name}Pos'][1].item()])
        offset_angle = meas_dict[f'Polar{self.object_name}Pos'][1]
        robot_target_frame_rotation_matrix = torch.stack([
            torch.stack([torch.cos(-offset_angle), -torch.sin(-offset_angle)]),
            torch.stack([torch.sin(-offset_angle), torch.cos(-offset_angle)]),
        ]).squeeze()
        robot_target_frame_vel = torch.matmul(robot_target_frame_rotation_matrix, meas_dict['RobotVel'][:2])
        angular_vel = meas_dict[f'Polar{self.object_name}Pos'][3] + meas_dict['RobotVel'][2]
        if torch.abs(robot_target_frame_vel[1]) == 0.0 or torch.abs(angular_vel) == 0.0:
            triangulated_distance = meas_dict[f'Polar{self.object_name}Pos'][0]
        else:
            # TODO: tan or plain?
            #triangulated_distance = torch.abs(robot_target_frame_vel[1] / torch.tan(angular_vel))
            triangulated_distance = torch.abs(robot_target_frame_vel[1] / angular_vel)

        return torch.stack([
            torch.atleast_1d(triangulated_distance - meas_dict[f'Polar{self.object_name}Pos'][0]),
            torch.atleast_1d(- robot_target_frame_vel[0] - meas_dict[f'Polar{self.object_name}Pos'][2]),
        ]).squeeze()