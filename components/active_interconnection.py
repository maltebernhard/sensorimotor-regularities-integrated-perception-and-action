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
        return torch.stack([meas_dict['vel_frontal'] - meas_dict['RobotVel'][0], meas_dict['vel_lateral'] - meas_dict['RobotVel'][1], meas_dict['vel_rot'] - meas_dict['RobotVel'][2]]).squeeze()

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