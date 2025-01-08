import torch
from typing import Dict
from components.measurement_model import MeasurementModel

# ========================================================================================================

class Angle_MM(MeasurementModel):
    def __init__(self, device, object_name:str="Target") -> None:
        self.object_name = object_name
        required_observations = [f'{self.object_name.lower()}_offset_angle', f'{self.object_name.lower()}_offset_angle_dot']
        super().__init__(f'Polar{object_name}Angle', required_observations, device)

    def implicit_measurement_model(self, x, meas_dict: Dict[str, torch.Tensor]):
        return torch.stack([
            (meas_dict[f'{self.object_name.lower()}_offset_angle'] - x[0] + torch.pi) % (2*torch.pi) - torch.pi,
            meas_dict[f'{self.object_name.lower()}_offset_angle_dot'] - x[1],
        ]).squeeze()

class Visibility_MM(MeasurementModel):
    def __init__(self, device, object_name:str="Target") -> None:
        self.object_name = object_name
        required_observations = [f'{self.object_name[0].lower() + self.object_name[1:]}_offset_angle']
        super().__init__(f'{object_name}Visibility', required_observations, device)

    def implicit_measurement_model(self, x, meas_dict: Dict[str, torch.Tensor]):
        return torch.ones(1)# - x[0]