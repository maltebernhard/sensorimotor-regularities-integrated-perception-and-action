import torch
from typing import List, Dict
from components.active_interconnection import ActiveInterconnection
from components.estimator import RecursiveEstimator

# ========================================================================================================

class Radius_Pos_VisAngle_AI(ActiveInterconnection):
    """
    Measurement Model:
    pos state:          target position and vel in robot frame
    radius state:       target radius
    visual angle obs:   perceived angle of obstacle in robot's visual field
    """
    def __init__(self, estimators: List[RecursiveEstimator], device, object_name: str) -> None:
        self.object_name = object_name
        required_estimators = [f'Cartesian{self.object_name}Pos', f'{self.object_name}Rad', f'{self.object_name[0].lower() + self.object_name[1:]}_visual_angle']
        super().__init__(estimators, required_estimators, device)

    def implicit_interconnection_model(self, meas_dict):
        return torch.asin(torch.minimum(torch.ones_like(meas_dict[f'{self.object_name}Rad'][0]), meas_dict[f'{self.object_name}Rad'][0] / meas_dict[f'Cartesian{self.object_name}Pos'][:2].norm())) - meas_dict[f'{self.object_name[0].lower() + self.object_name[1:]}_visual_angle'] / 2
    
class Visibility_Triangulation_AI(ActiveInterconnection):
    def __init__(self, estimators: List[RecursiveEstimator], device, object_name:str="Target") -> None:
        self.object_name = object_name
        required_estimators = [f'Polar{object_name}Pos', 'RobotVel', f'{object_name}Visibility']
        super().__init__(estimators, required_estimators, device)

    def implicit_interconnection_model(self, meas_dict: Dict[str, torch.Tensor]):
        if meas_dict[f'{self.object_name}Visibility'][0] < 0.5:
            print("TARGET NOT VISIBLE")
            return torch.atleast_1d(torch.abs(meas_dict[f'Polar{self.object_name}Pos'][1]) - (torch.pi - 1.0))

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

# TODO: making these two estimators update each other is stupid, it seems
class Cartesian_Polar_AI(ActiveInterconnection):
    def __init__(self, estimators: List[RecursiveEstimator], device, object_name:str="Target") -> None:
        self.object_name = object_name
        required_estimators = [f'Polar{object_name}Pos', f'Cartesian{object_name}Pos']
        super().__init__(estimators, required_estimators, device)

    def implicit_interconnection_model(self, meas_dict: Dict[str, torch.Tensor]):
        return torch.stack([
            meas_dict[f'Polar{self.object_name}Pos'][0] - meas_dict[f'Cartesian{self.object_name}Pos'][:2].norm(),
            meas_dict[f'Polar{self.object_name}Pos'][1] - torch.atan2(meas_dict[f'Cartesian{self.object_name}Pos'][1], meas_dict[f'Cartesian{self.object_name}Pos'][0]),
            #TODO: add derivatives
        ]).squeeze()
