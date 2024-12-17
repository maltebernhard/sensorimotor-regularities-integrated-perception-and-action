import torch
from typing import Dict
from components.measurement_model import MeasurementModel

# ========================================================================================================
    
# class Visibility_MM(MeasurementModel):
#     def __init__(self, device, object_name:str="Target") -> None:
#         self.object_name = object_name
#         required_observations = [f'{self.object_name[0].lower() + self.object_name[1:]}_offset_angle']
#         super().__init__(f'{object_name}Visibility', required_observations, device)

#     def implicit_measurement_model(self, x, meas_dict: Dict[str, torch.Tensor]):
#         return 1.0 - x[0]
    
class Foveal_Angle_MM(MeasurementModel):
    def __init__(self, device, object_name:str="Target") -> None:
        self.object_name = object_name
        required_observations = [f'{self.object_name[0].lower() + self.object_name[1:]}_offset_angle', f'del_{self.object_name[0].lower() + self.object_name[1:]}_offset_angle']
        super().__init__(f'Foveal{object_name}Angle', required_observations, device)

    def implicit_measurement_model(self, x, meas_dict: Dict[str, torch.Tensor]):
        return torch.stack([
            (x[0] - meas_dict[f'{self.object_name[0].lower() + self.object_name[1:]}_offset_angle'] + torch.pi) % (2*torch.pi) - torch.pi,
            x[1] - meas_dict[f'del_{self.object_name[0].lower() + self.object_name[1:]}_offset_angle'],
        ]).squeeze()