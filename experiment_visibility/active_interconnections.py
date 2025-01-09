import torch
from typing import List, Dict
from components.active_interconnection import ActiveInterconnection
from components.estimator import RecursiveEstimator
from components.helpers import rotate_vector_2d

# ========================================================================================================

class Radius_Pos_VisAngle_AI(ActiveInterconnection):
    """
    Measurement Model:
    pos state:          target position and vel in robot frame
    radius state:       target radius
    visual angle obs:   perceived angle of obstacle in robot's visual field
    """
    def __init__(self, object_name:str="Target") -> None:
        self.object_name = object_name
        required_estimators = [f'Cartesian{self.object_name}Pos', f'{self.object_name}Rad', f'{self.object_name[0].lower() + self.object_name[1:]}_visual_angle']
        super().__init__(required_estimators)

    def implicit_interconnection_model(self, meas_dict):
        return torch.asin(torch.minimum(torch.ones_like(meas_dict[f'{self.object_name}Rad'][0]), meas_dict[f'{self.object_name}Rad'][0] / meas_dict[f'Cartesian{self.object_name}Pos'][:2].norm())) - meas_dict[f'{self.object_name[0].lower() + self.object_name[1:]}_visual_angle'] / 2

class Visibility_Angle_AI(ActiveInterconnection):
    def __init__(self, sensor_angle_rad, object_name:str="Target"):
        self.object_name = object_name
        self.sensor_angle_rad = sensor_angle_rad
        required_estimators = [f'Polar{object_name}Pos', f'{object_name}Visibility']
        super().__init__(required_estimators)

    def visibility_plateau(self, angle: torch.Tensor):
        half_angle = self.sensor_angle_rad / 2
        if angle == 0.0: return 1.0
        elif torch.abs(angle) <= half_angle:
            return angle/angle
        elif angle > half_angle:
            if angle > torch.pi: return 0.0 * angle
            else: return torch.cos((angle-half_angle) * torch.pi/(torch.pi-half_angle)) / 2 + 0.5
        else:
            if angle < -torch.pi: return 0.0 * angle
            else: return torch.cos((angle+half_angle) * torch.pi/(torch.pi-half_angle)) / 2 + 0.5

    def visibility(self, angle: torch.Tensor):
        if torch.abs(angle) > torch.pi:
            return 0.0 * angle
        else: return torch.cos(angle) / 2 + 0.5

    def implicit_interconnection_model(self, meas_dict: Dict[str, torch.Tensor]):
        visibility = self.visibility_plateau(meas_dict[f'Polar{self.object_name}Pos'][1])
        #visibility = self.visibility(meas_dict[f'Polar{self.object_name}Pos'][1])
        return torch.atleast_1d(visibility - meas_dict[f'{self.object_name}Visibility'])
    
class Visibility_Detached_AI(ActiveInterconnection):
    def __init__(self, sensor_angle_rad, object_name:str="Target"):
        self.object_name = object_name
        self.sensor_angle_rad = sensor_angle_rad
        required_estimators = [f'Polar{object_name}Angle', f'{object_name}Visibility']
        super().__init__(required_estimators)

    def visibility_plateau(self, angle: torch.Tensor):
        half_angle = self.sensor_angle_rad / 2
        if angle == 0.0: return 1.0
        elif torch.abs(angle) <= half_angle:
            return angle/angle
        elif angle > half_angle:
            if angle > torch.pi: return 0.0 * angle
            else: return torch.cos((angle-half_angle) * torch.pi/(torch.pi-half_angle)) / 2 + 0.5
        else:
            if angle < -torch.pi: return 0.0 * angle
            else: return torch.cos((angle+half_angle) * torch.pi/(torch.pi-half_angle)) / 2 + 0.5

    def visibility(self, angle: torch.Tensor):
        if torch.abs(angle) > torch.pi:
            return 0.0 * angle
        else: return torch.cos(angle) / 2 + 0.5

    def implicit_interconnection_model(self, meas_dict: Dict[str, torch.Tensor]):
        visibility = self.visibility_plateau(meas_dict[f'Polar{self.object_name}Angle'][0])
        #visibility = self.visibility(meas_dict[f'Polar{self.object_name}Angle'][0])
        return torch.atleast_1d(visibility - meas_dict[f'{self.object_name}Visibility'])

class Triangulation_Visibility_AI(ActiveInterconnection):
    def __init__(self, object_name:str="Target") -> None:
        self.object_name = object_name
        required_estimators = [f'Polar{object_name}Pos', f'{object_name}Visibility', 'RobotVel']
        super().__init__(required_estimators)

    def implicit_interconnection_model(self, meas_dict: Dict[str, torch.Tensor]):
        rtf_vel = rotate_vector_2d(meas_dict[f'Polar{self.object_name}Pos'][1], meas_dict['RobotVel'][:2])
        angular_vel = meas_dict[f'Polar{self.object_name}Pos'][3] + meas_dict['RobotVel'][2]
        if torch.abs(rtf_vel[1]) == 0.0 or torch.abs(angular_vel) == 0.0:
            triangulated_distance = meas_dict[f'Polar{self.object_name}Pos'][0]
        else:
            triangulated_distance = torch.abs(rtf_vel[1] / angular_vel)

        return torch.stack([
            torch.atleast_1d(triangulated_distance - meas_dict[f'Polar{self.object_name}Pos'][0]) * meas_dict[f'{self.object_name}Visibility'],
            torch.atleast_1d(- rtf_vel[0] - meas_dict[f'Polar{self.object_name}Pos'][2]) * meas_dict[f'{self.object_name}Visibility'],
        ]).squeeze()
    
class Triangulation_Detached_AI(ActiveInterconnection):
    def __init__(self, object_name:str="Target") -> None:
        self.object_name = object_name
        required_estimators = [f'Polar{object_name}Distance', f'Polar{object_name}Angle', f'{object_name}Visibility', 'RobotVel']
        super().__init__(required_estimators)

    def implicit_interconnection_model(self, meas_dict: Dict[str, torch.Tensor]):
        rtf_vel = rotate_vector_2d(meas_dict[f'Polar{self.object_name}Angle'][0], meas_dict['RobotVel'][:2])
        angular_vel = meas_dict[f'Polar{self.object_name}Angle'][1] + meas_dict['RobotVel'][2]
        if torch.abs(rtf_vel[1]) == 0.0 or torch.abs(angular_vel) == 0.0:
            triangulated_distance = meas_dict[f'Polar{self.object_name}Distance'][0]
        else:
            triangulated_distance = torch.abs(rtf_vel[1] / angular_vel)
        return torch.stack([
            torch.atleast_1d(triangulated_distance - meas_dict[f'Polar{self.object_name}Distance'][0]) * meas_dict[f'{self.object_name}Visibility'],
            torch.atleast_1d(- rtf_vel[0] - meas_dict[f'Polar{self.object_name}Distance'][1]) * meas_dict[f'{self.object_name}Visibility'],
        ]).squeeze()
