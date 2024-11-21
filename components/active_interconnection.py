from abc import ABC, abstractmethod
from typing import Dict, List

import torch

from components.estimator import RecursiveEstimator
from components.measurement_model import ImplicitMeasurementModel
    
# ==================================================================================================================================================

class ActiveInterconnection(ABC, ImplicitMeasurementModel):
    def __init__(self, estimators: List[RecursiveEstimator], device):
        meas_config = {estimator.id: estimator.state_mean.size().numel() for estimator in estimators}
        super().__init__(meas_config=meas_config, device=device)
        self.connected_estimators: Dict[str, RecursiveEstimator] = {est.id: est for est in estimators}

    def add_estimator(self, estimator: RecursiveEstimator):
        self.connected_estimators[estimator.id] = estimator

    def get_state_dict(self, buffer_dict, estimator_id):
        return {id: buffer_dict[id]['state_mean'] for id in self.connected_estimators.keys() if id != estimator_id}

    def implicit_measurement_model(self, x, meas_dict):
        missing_key = next(key for key in self.connected_estimators.keys() if key not in meas_dict)
        assert sum(1 for key in self.connected_estimators.keys() if key not in meas_dict) == 1, "There should be exactly one missing key"
        meas_dict[missing_key] = x
        return self.implicit_interconnection_model(meas_dict)

    @abstractmethod
    def implicit_interconnection_model(self, meas_dict):
        pass

# ==================================================================================================================================================

class Pos_Angle_Vel_AI(ActiveInterconnection):
    """
    Measurement Model:
    pos state:          target position and vel in robot frame
    vel state:          robot vel
    angle state:        target angular offset
    """
    def __init__(self, estimators, object_name: str, device) -> None:
        self.object_name = object_name
        super().__init__(estimators, device)

    def implicit_interconnection_model(self, meas_dict):
        # avoids NaN from atan2(0,0)
        if meas_dict[f"{self.object_name}Pos"][1] == 0.0 and [f"{self.object_name}Pos"][0] == 0.0:
            angle = torch.tensor([0.0]).to(self.device)
        else:
            angle = torch.atan2(meas_dict[f"{self.object_name}Pos"][1],meas_dict[f"{self.object_name}Pos"][0])
        angle_diff = meas_dict[f"{self.object_name[0].lower() + self.object_name[1:]}_offset_angle"] - angle
        return torch.concat([
            ((angle_diff + torch.pi) % (2*torch.pi)) - torch.pi,
            - meas_dict['RobotVel'] - meas_dict[f"{self.object_name}Pos"][2:],
        ]).squeeze()

class Pos_Angle_AI(ActiveInterconnection):
    """
    Measurement Model:
    pos state:          target position and vel in robot frame
    angle state:        target angular offset
    """
    def __init__(self, estimators, object_name: str, device) -> None:
        self.object_name = object_name
        super().__init__(estimators, device)

    def implicit_interconnection_model(self, meas_dict):
        # avoids NaN from atan2(0,0)
        if meas_dict[f"{self.object_name}Pos"][1] == 0.0 and [f"{self.object_name}Pos"][0] == 0.0:
            angle = torch.tensor([0.0]).to(self.device)
        else:
            angle = torch.atan2(meas_dict[f"{self.object_name}Pos"][1],meas_dict[f"{self.object_name}Pos"][0])
        angle_diff = meas_dict[f"{self.object_name[0].lower() + self.object_name[1:]}_offset_angle"] - angle
        return ((angle_diff + torch.pi) % (2*torch.pi)) - torch.pi

class Vel_AI(ActiveInterconnection):
    """
    Measurement Model:
    vel state:          robot vel
    vel obs:            robot vel observation
    """
    def __init__(self, estimators, device) -> None:
        super().__init__(estimators, device)

    def implicit_interconnection_model(self, meas_dict):
        return torch.stack([
            meas_dict['vel_frontal'] - meas_dict['RobotVel'][0],
            meas_dict['vel_lateral'] - meas_dict['RobotVel'][1],
            meas_dict['vel_rot'] - meas_dict['RobotVel'][2]
        ]).squeeze()

class Radius_Pos_VisAngle_AI(ActiveInterconnection):
    """
    Measurement Model:
    pos state:          target position and vel in robot frame
    radius state:       target radius
    visual angle obs:   perceived angle of obstacle in robot's visual field
    """
    def __init__(self, estimators: List[RecursiveEstimator], obstacle_id, device) -> None:
        self.obstacle_id = obstacle_id
        super().__init__(estimators, device)

    def implicit_interconnection_model(self, meas_dict):
        return torch.asin(torch.minimum(torch.ones_like(meas_dict[f'Obstacle{self.obstacle_id}Rad'][0]), meas_dict[f'Obstacle{self.obstacle_id}Rad'][0] / meas_dict[f'Obstacle{self.obstacle_id}Pos'][:2].norm())) - meas_dict[f'obstacle{self.obstacle_id}_visual_angle'] / 2
    
class Polar_Angle_Vel_AI(ActiveInterconnection):
    """
    Measurement Model:
    polar pos state:    object position and vel in polar robot frame
    angle observation:  target polar angle and vel
    """
    def __init__(self, estimators: List[RecursiveEstimator], object_name: str, device) -> None:
        self.object_name = object_name
        super().__init__(estimators, device)

    def implicit_interconnection_model(self, meas_dict):
        return torch.stack([
            meas_dict[f'{self.object_name[0].lower() + self.object_name[1:]}_offset_angle'] - meas_dict[f'Polar{self.object_name}Pos'][1],
            meas_dict[f'del_{self.object_name[0].lower() + self.object_name[1:]}_offset_angle'] - meas_dict[f'Polar{self.object_name}Pos'][3],
        ]).squeeze()
    
class Polar_Angle_Vel_NonCart_AI(ActiveInterconnection):
    """
    Measurement Model:
    polar pos state:    object position and vel in polar robot frame
    angle observation:  target polar angle and vel
    """
    def __init__(self, estimators: List[RecursiveEstimator], object_name: str, device) -> None:
        self.object_name = object_name
        super().__init__(estimators, device)

    def implicit_interconnection_model(self, meas_dict: Dict[str, torch.Tensor]):
        offset_angle = meas_dict[f'{self.object_name[0].lower() + self.object_name[1:]}_offset_angle']
        robot_target_frame_rotation_matrix = torch.stack([
            torch.stack([torch.cos(-offset_angle), -torch.sin(-offset_angle)]),
            torch.stack([torch.sin(-offset_angle), torch.cos(-offset_angle)]),
        ]).squeeze()
        robot_target_frame_vel = torch.matmul(robot_target_frame_rotation_matrix, meas_dict['RobotVel'][:2])
        angular_vel = meas_dict[f'del_{self.object_name[0].lower() + self.object_name[1:]}_offset_angle'] + meas_dict['RobotVel'][2]
        # TODO: tan or plain?
        triangulated_distance = torch.abs(robot_target_frame_vel[1] / torch.tan(angular_vel))
        #triangulated_distance = torch.abs(robot_target_frame_vel[1] / angular_vel)

        return torch.stack([
            torch.atleast_1d(triangulated_distance - meas_dict[f'Polar{self.object_name}Pos'][0]),
            (offset_angle - meas_dict[f'Polar{self.object_name}Pos'][1] + torch.pi) % (2*torch.pi) - torch.pi,
            torch.atleast_1d(- robot_target_frame_vel[0] - meas_dict[f'Polar{self.object_name}Pos'][2]),
            meas_dict[f'del_{self.object_name[0].lower() + self.object_name[1:]}_offset_angle'] - meas_dict[f'Polar{self.object_name}Pos'][3],
        ]).squeeze()
    
class Polar_Angle_AI(ActiveInterconnection):
    """
    Measurement Model:
    polar pos state:    object position in polar robot frame
    angle observation:  target polar angle
    """
    def __init__(self, estimators: List[RecursiveEstimator], object_name: str, device) -> None:
        self.object_name = object_name
        super().__init__(estimators, device)

    def implicit_interconnection_model(self, meas_dict):
        return torch.atleast_1d(meas_dict[f'{self.object_name[0].lower() + self.object_name[1:]}_offset_angle'] - meas_dict[f'Polar{self.object_name}Pos'][1])
    
class Polar_Distance_Vel_AI(ActiveInterconnection):
    """
    Measurement Model:
    polar pos state:    object position and vel in polar robot frame
    pos state:          object position and vel in cartesian robot frame
    """
    def __init__(self, estimators: List[RecursiveEstimator], object_name: str, device) -> None:
        self.object_name = object_name
        super().__init__(estimators, device)

    def implicit_interconnection_model(self, meas_dict):
        return torch.stack([
            meas_dict[f'{self.object_name}Pos'][:2].norm() - meas_dict[f'Polar{self.object_name}Pos'][0],
            meas_dict[f'{self.object_name}Pos'][2:4].norm() - meas_dict[f'Polar{self.object_name}Pos'][2]
        ]).squeeze()
    
class Polar_Distance_AI(ActiveInterconnection):
    """
    Measurement Model:
    polar pos state:    object position in polar robot frame
    pos state:          object position in cartesian robot frame
    """
    def __init__(self, estimators: List[RecursiveEstimator], object_name: str, device) -> None:
        self.object_name = object_name
        super().__init__(estimators, device)

    def implicit_interconnection_model(self, meas_dict: Dict[str, torch.Tensor]):
        return torch.atleast_1d(meas_dict[f'{self.object_name}Pos'][:2].norm() - meas_dict[f'Polar{self.object_name}Pos'][0])