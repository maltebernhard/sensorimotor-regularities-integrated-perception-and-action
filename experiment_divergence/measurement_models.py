import torch
from typing import List, Dict

from components.measurement_model import MeasurementModel

# ========================================================================================================

class Vis_Angle_MM(MeasurementModel):
    def __init__(self, device, object_name:str="Target") -> None:
        self.object_name = object_name
        required_observations = [f'{self.object_name.lower()}_visual_angle', f'{self.object_name.lower()}_visual_angle_dot']
        super().__init__(f'Polar{object_name}PosRadius', required_observations, device)

    def implicit_measurement_model(self, x, meas_dict: Dict[str, torch.Tensor]):
        distance = x[0]
        distance_dot = x[2]
        radius = x[4]
        return torch.stack([
            2 * torch.asin(radius / distance) - meas_dict[f'{self.object_name.lower()}_visual_angle'],
            -2 * radius * distance_dot / (distance**2 * torch.sqrt(1 - (radius / distance)**2)) - meas_dict[f'{self.object_name.lower()}_visual_angle_dot']
        ]).squeeze()

class Angle_Meas_MM(MeasurementModel):
    def __init__(self, device, object_name:str="Target") -> None:
        self.object_name = object_name
        required_observations = [f'{self.object_name.lower()}_offset_angle', f'{self.object_name.lower()}_offset_angle_dot']
        super().__init__(f'Polar{object_name}PosRadius', required_observations, device)

    def implicit_measurement_model(self, x, meas_dict: Dict[str, torch.Tensor]):
        return torch.stack([
            (meas_dict[f'{self.object_name.lower()}_offset_angle'] - x[1] + torch.pi) % (2*torch.pi) - torch.pi,
            meas_dict[f'{self.object_name.lower()}_offset_angle_dot'] - x[3],
        ]).squeeze()