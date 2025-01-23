import torch
from typing import Dict
from components.measurement_model import MeasurementModel

# ========================================================================================================

class Robot_Vel_MM(MeasurementModel):
    """
    Measurement Model:
    vel state:          robot vel
    vel obs:            robot vel observation
    """
    def __init__(self) -> None:
        required_observations = ['vel_frontal', 'vel_lateral', 'vel_rot']
        super().__init__('RobotVel', required_observations)

    def implicit_measurement_model(self, x, meas_dict):
        return torch.stack([
            meas_dict['vel_frontal'] - x[0],
            meas_dict['vel_lateral'] - x[1],
            meas_dict['vel_rot'] - x[2]
        ]).squeeze()

class Angle_MM(MeasurementModel):
    def __init__(self, object_name:str="Target") -> None:
        self.object_name = object_name
        required_observations = [f'{self.object_name.lower()}_offset_angle', f'{self.object_name.lower()}_offset_angle_dot']
        super().__init__(f'Polar{object_name}Pos', required_observations)

    def implicit_measurement_model(self, x, meas_dict: Dict[str, torch.Tensor]):
        return torch.stack([
            (meas_dict[f'{self.object_name.lower()}_offset_angle'] - x[1] + torch.pi) % (2*torch.pi) - torch.pi,
            meas_dict[f'{self.object_name.lower()}_offset_angle_dot'] - x[3],
        ]).squeeze()