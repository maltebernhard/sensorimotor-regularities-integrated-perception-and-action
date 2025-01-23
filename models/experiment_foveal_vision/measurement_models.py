import torch
from typing import Dict
from components.measurement_model import ActiveInterconnection

# ========================================================================================================
    
class Robot_Vel_MM(ActiveInterconnection):
    """
    Measurement Model:
    vel state:          robot vel
    vel obs:            robot vel observation
    """
    def __init__(self) -> None:
        required_observations = ['vel_frontal', 'vel_lateral', 'vel_rot']
        super().__init__(['RobotVel'], required_observations)

    def implicit_interconnection_model(self, meas_dict):
        return torch.stack([
            meas_dict['vel_frontal'] - meas_dict['RobotVel'][0],
            meas_dict['vel_lateral'] - meas_dict['RobotVel'][1],
            meas_dict['vel_rot'] - meas_dict['RobotVel'][2]
        ]).squeeze()
    
class Angle_MM(ActiveInterconnection):
    def __init__(self, object_name:str="Target") -> None:
        self.object_name = object_name
        required_observations = [f'{self.object_name.lower()}_offset_angle', f'{self.object_name.lower()}_offset_angle_dot']
        super().__init__([f'{object_name}FovealVision'], required_observations)

    def implicit_interconnection_model(self, meas_dict: Dict[str, torch.Tensor]):
        return torch.stack([
            (meas_dict[f'{self.object_name.lower()}_offset_angle'] - meas_dict[f'{self.object_name}FovealVision'][0] + torch.pi) % (2*torch.pi) - torch.pi,
            meas_dict[f'{self.object_name.lower()}_offset_angle_dot'] - meas_dict[f'{self.object_name}FovealVision'][1],
        ]).squeeze()
    
class Distance_MM(ActiveInterconnection):
    def __init__(self, object_name:str="Target") -> None:
        self.object_name = object_name
        required_observations = [f'{self.object_name.lower()}_distance']
        super().__init__([f'Polar{object_name}Pos'], required_observations)

    def implicit_interconnection_model(self, meas_dict: Dict[str, torch.Tensor]):
        return torch.atleast_1d(torch.stack([
            meas_dict[f'{self.object_name.lower()}_distance'] - meas_dict[f'Polar{self.object_name}Pos'][0]
        ]).squeeze())