import torch
from typing import Dict
from components.measurement_model import MeasurementModel

# ========================================================================================================
    
class Visibility_MM(MeasurementModel):
    def __init__(self, object_name:str="Target") -> None:
        self.object_name = object_name
        required_observations = [f'{self.object_name[0].lower() + self.object_name[1:]}_offset_angle']
        super().__init__(f'{object_name}Visibility', required_observations)

    def implicit_measurement_model(self, x, meas_dict: Dict[str, torch.Tensor]):
        return torch.ones(1)# - x[0]