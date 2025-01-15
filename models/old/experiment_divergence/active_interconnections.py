import torch
from typing import List, Dict
from components.active_interconnection import ActiveInterconnection
from components.estimator import RecursiveEstimator

# ========================================================================================================
    
class VisAngle_Rad_AI(ActiveInterconnection):
    def __init__(self, object_name:str="Target") -> None:
        self.object_name = object_name
        required_estimators = [f'Polar{object_name}Pos', f'{self.object_name}VisAngle', f'{self.object_name}Rad']
        super().__init__(required_estimators)

    def implicit_interconnection_model(self, meas_dict: Dict[str, torch.Tensor]):
        distance = meas_dict[f'Polar{self.object_name}Pos'][0]
        distance_dot = meas_dict[f'Polar{self.object_name}Pos'][2]
        vis_angle = meas_dict[f'{self.object_name}VisAngle'][0]
        vis_angle_dot = meas_dict[f'{self.object_name}VisAngle'][1]
        radius = meas_dict[f'{self.object_name}Rad']
        return torch.stack([
            2 * torch.asin(radius / distance) - vis_angle,
            -2 * radius * distance_dot / (distance**2 * torch.sqrt(1 - (radius / distance)**2)) - vis_angle_dot,
        ]).squeeze()