import torch
from typing import Dict
from components.measurement_model import ActiveInterconnection

# ========================================================================================================
    
class Angle_MM(ActiveInterconnection):
    def __init__(self, object_name:str="Target") -> None:
        self.object_name = object_name
        required_observations = [
            f'{self.object_name.lower()}_offset_angle',
            f'{self.object_name.lower()}_offset_angle_dot',
            f'{self.object_name.lower()}_visual_angle',
            f'{self.object_name.lower()}_visual_angle_dot'
        ]
        super().__init__([f'Polar{object_name}Pos'], required_observations)

    def implicit_interconnection_model(self, meas_dict: Dict[str, torch.Tensor]):
        return torch.stack([
            (meas_dict[f'{self.object_name.lower()}_offset_angle']    - meas_dict[f'Polar{self.object_name}Pos'][1] + torch.pi) % (2*torch.pi) - torch.pi,
            meas_dict[f'{self.object_name.lower()}_offset_angle_dot'] - meas_dict[f'Polar{self.object_name}Pos'][3],
            meas_dict[f'{self.object_name.lower()}_visual_angle']     - meas_dict[f'Polar{self.object_name}Pos'][4],
            meas_dict[f'{self.object_name.lower()}_visual_angle_dot'] - meas_dict[f'Polar{self.object_name}Pos'][5],
        ]).squeeze()