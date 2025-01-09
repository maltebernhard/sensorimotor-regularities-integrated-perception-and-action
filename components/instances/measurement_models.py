import torch
from typing import Dict
from components.measurement_model import MeasurementModel

# ========================================================================================================

class Pos_Angle_MM(MeasurementModel):
    """
    Measurement Model:
    pos state:          target position and vel in robot frame
    angle state:        target angular offset
    """
    def __init__(self, object_name:str="Target") -> None:
        self.object_name = object_name
        required_observations = [f"{self.object_name[0].lower() + self.object_name[1:]}_offset_angle", "vel_frontal", "vel_lateral", "vel_rot"]
        super().__init__(f"Cartesian{self.object_name}Pos", required_observations)

    def implicit_measurement_model(self, x, meas_dict):
        # avoids NaN from atan2(0,0)
        if x[1] == 0.0 and x[0] == 0.0:
            angle = torch.tensor([0.0])
        else:
            angle = torch.atan2(x[1], x[0])
        angle_diff = meas_dict[f"{self.object_name.lower()}_offset_angle"] - angle
        return torch.concat([
            ((angle_diff + torch.pi) % (2*torch.pi)) - torch.pi,
            - torch.concat([meas_dict['vel_frontal'], meas_dict['vel_lateral'], meas_dict['vel_rot']]) - x[2:],
        ]).squeeze()
    
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