import torch
from typing import List, Dict
from components.active_interconnection import ActiveInterconnection, MeasurementModel
from components.estimator import RecursiveEstimator

# ========================================================================================================

class Radius_Pos_VisAngle_AI(ActiveInterconnection):
    """
    Measurement Model:
    pos state:          target position and vel in robot frame
    radius state:       target radius
    visual angle obs:   perceived angle of obstacle in robot's visual field
    """
    def __init__(self, estimators: List[RecursiveEstimator], device, object_name: str) -> None:
        self.object_name = object_name
        required_estimators = [f'Cartesian{self.object_name}Pos', f'{self.object_name}Rad', f'{self.object_name[0].lower() + self.object_name[1:]}_visual_angle']
        super().__init__(estimators, required_estimators, device)

    def implicit_interconnection_model(self, meas_dict):
        return torch.asin(torch.minimum(torch.ones_like(meas_dict[f'{self.object_name}Rad'][0]), meas_dict[f'{self.object_name}Rad'][0] / meas_dict[f'Cartesian{self.object_name}Pos'][:2].norm())) - meas_dict[f'{self.object_name[0].lower() + self.object_name[1:]}_visual_angle'] / 2

class Triangulation_AI(ActiveInterconnection):
    def __init__(self, estimators: List[RecursiveEstimator], device, object_name:str="Target") -> None:
        self.object_name = object_name
        required_estimators = [f'Polar{object_name}Pos', 'RobotVel']
        super().__init__(estimators, required_estimators, device)

    def implicit_interconnection_model(self, meas_dict: Dict[str, torch.Tensor]):
        # TODO: suppresses gradient propagation of changes in offset angle in this AI
            # BAD because it suppresses an action component
            # GOOD because we don't want rotation to influence triangulation
            # NOTE: not sure whether this suppression will have additional effects other than the desired one
        #offset_angle = torch.tensor([meas_dict[f'Polar{self.object_name}Pos'][1].item()])
        offset_angle = meas_dict[f'Polar{self.object_name}Pos'][1]
        robot_target_frame_rotation_matrix = torch.stack([
            torch.stack([torch.cos(-offset_angle), -torch.sin(-offset_angle)]),
            torch.stack([torch.sin(-offset_angle), torch.cos(-offset_angle)]),
        ]).squeeze()
        robot_target_frame_vel = torch.matmul(robot_target_frame_rotation_matrix, meas_dict['RobotVel'][:2])
        angular_vel = meas_dict[f'Polar{self.object_name}Pos'][3] + meas_dict['RobotVel'][2]
        if torch.abs(robot_target_frame_vel[1]) == 0.0 or torch.abs(angular_vel) == 0.0:
            triangulated_distance = meas_dict[f'Polar{self.object_name}Pos'][0]
        else:
            # TODO: tan or plain?
            #triangulated_distance = torch.abs(robot_target_frame_vel[1] / torch.tan(angular_vel))
            triangulated_distance = torch.abs(robot_target_frame_vel[1] / angular_vel)

        return torch.stack([
            torch.atleast_1d(triangulated_distance - meas_dict[f'Polar{self.object_name}Pos'][0]),
            torch.atleast_1d(- robot_target_frame_vel[0] - meas_dict[f'Polar{self.object_name}Pos'][2]),
        ]).squeeze()
    
class Visibility_Triangulation_AI(ActiveInterconnection):
    def __init__(self, estimators: List[RecursiveEstimator], device, object_name:str="Target") -> None:
        self.object_name = object_name
        required_estimators = [f'Polar{object_name}Pos', 'RobotVel', f'{object_name}Visibility']
        super().__init__(estimators, required_estimators, device)

    def implicit_interconnection_model(self, meas_dict: Dict[str, torch.Tensor]):
        if meas_dict[f'{self.object_name}Visibility'][0] < 0.5:
            print("TARGET NOT VISIBLE")
            return torch.atleast_1d(torch.abs(meas_dict[f'Polar{self.object_name}Pos'][1]) - (torch.pi - 1.0))

        # TODO: suppresses gradient propagation of changes in offset angle in this AI
            # BAD because it suppresses an action component
            # GOOD because we don't want rotation to influence triangulation
            # NOTE: not sure whether this suppression will have additional effects other than the desired one
        #offset_angle = torch.tensor([meas_dict[f'Polar{self.object_name}Pos'][1].item()])
        offset_angle = meas_dict[f'Polar{self.object_name}Pos'][1]
        robot_target_frame_rotation_matrix = torch.stack([
            torch.stack([torch.cos(-offset_angle), -torch.sin(-offset_angle)]),
            torch.stack([torch.sin(-offset_angle), torch.cos(-offset_angle)]),
        ]).squeeze()
        robot_target_frame_vel = torch.matmul(robot_target_frame_rotation_matrix, meas_dict['RobotVel'][:2])
        angular_vel = meas_dict[f'Polar{self.object_name}Pos'][3] + meas_dict['RobotVel'][2]
        if torch.abs(robot_target_frame_vel[1]) == 0.0 or torch.abs(angular_vel) == 0.0:
            triangulated_distance = meas_dict[f'Polar{self.object_name}Pos'][0]
        else:
            # TODO: tan or plain?
            #triangulated_distance = torch.abs(robot_target_frame_vel[1] / torch.tan(angular_vel))
            triangulated_distance = torch.abs(robot_target_frame_vel[1] / angular_vel)

        return torch.stack([
            torch.atleast_1d(triangulated_distance - meas_dict[f'Polar{self.object_name}Pos'][0]),
            torch.atleast_1d(- robot_target_frame_vel[0] - meas_dict[f'Polar{self.object_name}Pos'][2]),
        ]).squeeze()

# TODO: making these two estimators update each other is stupid, it seems
class Cartesian_Polar_AI(ActiveInterconnection):
    def __init__(self, estimators: List[RecursiveEstimator], device, object_name:str="Target") -> None:
        self.object_name = object_name
        required_estimators = [f'Polar{object_name}Pos', f'Cartesian{object_name}Pos']
        super().__init__(estimators, required_estimators, device)

    def implicit_interconnection_model(self, meas_dict: Dict[str, torch.Tensor]):
        return torch.stack([
            meas_dict[f'Polar{self.object_name}Pos'][0] - meas_dict[f'Cartesian{self.object_name}Pos'][:2].norm(),
            meas_dict[f'Polar{self.object_name}Pos'][1] - torch.atan2(meas_dict[f'Cartesian{self.object_name}Pos'][1], meas_dict[f'Cartesian{self.object_name}Pos'][0]),
            #TODO: add derivatives
        ]).squeeze()

class Gaze_Fixation_AI(ActiveInterconnection):
    def __init__(self, estimators, device):
        required_estimators = ['PolarTargetPos', 'RobotVel']
        super().__init__(estimators, required_estimators, device)

    def implicit_interconnection_model(self, meas_dict):
        offset_angle = meas_dict[f'PolarTargetPos'][1]
        robot_target_frame_rotation_matrix = torch.stack([
            torch.stack([torch.cos(-offset_angle), -torch.sin(-offset_angle)]),
            torch.stack([torch.sin(-offset_angle), torch.cos(-offset_angle)]),
        ]).squeeze()
        robot_target_frame_vel = torch.matmul(robot_target_frame_rotation_matrix, meas_dict['RobotVel'][:2])
        return torch.atleast_1d(robot_target_frame_vel[1] / meas_dict['PolarTargetPos'][0] + meas_dict['RobotVel'][2])
    
# ========================================================================================================

class Vel_MM(MeasurementModel):
    """
    Measurement Model:
    vel state:          robot vel
    vel obs:            robot vel observation
    """
    def __init__(self, device) -> None:
        required_observations = ['vel_frontal', 'vel_lateral', 'vel_rot']
        super().__init__('RobotVel', required_observations, device)

    def implicit_measurement_model(self, x, meas_dict):
        return torch.stack([
            meas_dict['vel_frontal'] - x[0],
            meas_dict['vel_lateral'] - x[1],
            meas_dict['vel_rot'] - x[2]
        ]).squeeze()
    
class Angle_Meas_MM(MeasurementModel):
    def __init__(self, device, object_name:str="Target") -> None:
        self.object_name = object_name
        required_observations = [f'{self.object_name[0].lower() + self.object_name[1:]}_offset_angle', f'del_{self.object_name[0].lower() + self.object_name[1:]}_offset_angle']
        super().__init__(f'Polar{object_name}Pos', required_observations, device)

    def implicit_measurement_model(self, x, meas_dict: Dict[str, torch.Tensor]):
        print(meas_dict)
        return torch.stack([
            (meas_dict[f'{self.object_name[0].lower() + self.object_name[1:]}_offset_angle'] - x[1] + torch.pi) % (2*torch.pi) - torch.pi,
            meas_dict[f'del_{self.object_name[0].lower() + self.object_name[1:]}_offset_angle'] - x[3],
        ]).squeeze()
    
class Visibility_MM(MeasurementModel):
    def __init__(self, device, object_name:str="Target") -> None:
        self.object_name = object_name
        required_observations = [f'{self.object_name[0].lower() + self.object_name[1:]}_offset_angle']
        super().__init__(f'{object_name}Visibility', required_observations, device)

    def implicit_measurement_model(self, x, meas_dict: Dict[str, torch.Tensor]):
        return 1.0 - x[0]

class Pos_Angle_MM(MeasurementModel):
    """
    Measurement Model:
    pos state:          target position and vel in robot frame
    angle state:        target angular offset
    """
    def __init__(self, device, object_name:str="Target") -> None:
        self.object_name = object_name
        required_observations = [f"{self.object_name[0].lower() + self.object_name[1:]}_offset_angle", "vel_frontal", "vel_lateral", "vel_rot"]
        super().__init__(f"Cartesian{self.object_name}Pos", required_observations, device)

    def implicit_measurement_model(self, x, meas_dict):
        # avoids NaN from atan2(0,0)
        if x[1] == 0.0 and x[0] == 0.0:
            angle = torch.tensor([0.0]).to(self.device)
        else:
            angle = torch.atan2(x[1], x[0])
        angle_diff = meas_dict[f"{self.object_name[0].lower() + self.object_name[1:]}_offset_angle"] - angle
        return torch.concat([
            ((angle_diff + torch.pi) % (2*torch.pi)) - torch.pi,
            - torch.concat([meas_dict['vel_frontal'], meas_dict['vel_lateral'], meas_dict['vel_rot']]) - x[2:],
        ]).squeeze()