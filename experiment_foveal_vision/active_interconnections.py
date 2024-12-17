import torch
from typing import List, Dict
from components.active_interconnection import ActiveInterconnection
from components.estimator import RecursiveEstimator

# ========================================================================================================

# class Visibility_Angle_AI(ActiveInterconnection):
#     def __init__(self, estimators: List[RecursiveEstimator], device, object_name:str="Target", sensor_angle_rad = torch.pi/2):
#         self.object_name = object_name
#         self.sensor_angle_rad = sensor_angle_rad
#         required_estimators = [f'Polar{object_name}Pos', f'{object_name}Visibility']
#         super().__init__(estimators, required_estimators, device)

#     def visibility_plateau(self, angle: torch.Tensor):
#         half_angle = self.sensor_angle_rad / 2
#         if angle == 0.0: return 1.0
#         elif torch.abs(angle) <= half_angle:
#             return angle/angle
#         elif angle > half_angle:
#             if angle > torch.pi: return 0.0 * angle
#             else: return torch.cos((angle-half_angle) * torch.pi/(torch.pi-half_angle)) / 2 + 0.5
#         else:
#             if angle < -torch.pi: return 0.0 * angle
#             else: return torch.cos((angle+half_angle) * torch.pi/(torch.pi-half_angle)) / 2 + 0.5

#     def visibility(self, angle: torch.Tensor):
#         if torch.abs(angle) > torch.pi:
#             return 0.0 * angle
#         else: return torch.cos(angle) / 2 + 0.5

#     def implicit_interconnection_model(self, meas_dict: Dict[str, torch.Tensor]):
#         #visibility = self.visibility_plateau(meas_dict[f'Polar{self.object_name}Pos'][1])
#         visibility = self.visibility(meas_dict[f'Polar{self.object_name}Pos'][1])
#         return torch.atleast_1d(visibility - meas_dict[f'{self.object_name}Visibility'])
    
class Foveal_Angle_AI(ActiveInterconnection):
    def __init__(self, estimators: List[RecursiveEstimator], device, object_name:str="Target"):
        self.object_name = object_name
        required_estimators = [f'Polar{object_name}Pos', f'Foveal{object_name}Angle']
        super().__init__(estimators, required_estimators, device)

    def implicit_interconnection_model(self, meas_dict: Dict[str, torch.Tensor]):
        return torch.stack([
            (meas_dict[f'Foveal{self.object_name}Angle'][0] - meas_dict[f'Polar{self.object_name}Pos'][1] + torch.pi) % (2*torch.pi) - torch.pi,
            meas_dict[f'Foveal{self.object_name}Angle'][1] - meas_dict[f'Polar{self.object_name}Pos'][3],
        ]).squeeze()