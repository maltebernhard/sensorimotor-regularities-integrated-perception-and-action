import torch
from typing import Dict
from components.measurement_model import ActiveInterconnection
from components.helpers import rotate_vector_2d

# ========================================================================================================

class Triangulation_AI(ActiveInterconnection):
    def __init__(self, object_name:str="Target") -> None:
        self.object_name = object_name
        required_estimators = [f'Polar{object_name}Pos', 'RobotVel']
        super().__init__(required_estimators)

    def implicit_interconnection_model(self, meas_dict: Dict[str, torch.Tensor]):
        rtf_vel = rotate_vector_2d(-meas_dict[f'Polar{self.object_name}Pos'][1], meas_dict['RobotVel'][:2])
        angular_vel = meas_dict[f'Polar{self.object_name}Pos'][3]
        if torch.abs(rtf_vel[1]) == 0.0 or torch.abs(angular_vel) == 0.0:
            triangulated_distance = meas_dict[f'Polar{self.object_name}Pos'][0]
        else:
            triangulated_distance = torch.abs(rtf_vel[1] / angular_vel)
        return torch.atleast_1d(triangulated_distance - meas_dict[f'Polar{self.object_name}Pos'][0])