import torch
from typing import List, Dict
from components.active_interconnection import ActiveInterconnection
from components.estimator import RecursiveEstimator

# ========================================================================================================

class Pos_Angle_AI(ActiveInterconnection):
    """
    Measurement Model:
    pos state:          target position and vel in robot frame
    angle state:        target angular offset
    """
    def __init__(self, estimators, device, object_name:str="Target", estimate_vel:bool=True) -> None:
        self.object_name = object_name
        self.estimate_vel = estimate_vel
        required_estimators = [f"{self.object_name}Pos", f"{self.object_name[0].lower() + self.object_name[1:]}_offset_angle"]
        if estimate_vel:
            required_estimators.append("RobotVel")
        super().__init__(estimators, required_estimators, device)

    def implicit_interconnection_model(self, meas_dict):
        # avoids NaN from atan2(0,0)
        if meas_dict[f"{self.object_name}Pos"][1] == 0.0 and [f"{self.object_name}Pos"][0] == 0.0:
            angle = torch.tensor([0.0]).to(self.device)
        else:
            angle = torch.atan2(meas_dict[f"{self.object_name}Pos"][1],meas_dict[f"{self.object_name}Pos"][0])
        angle_diff = meas_dict[f"{self.object_name[0].lower() + self.object_name[1:]}_offset_angle"] - angle
        if self.estimate_vel:
            return torch.concat([
                ((angle_diff + torch.pi) % (2*torch.pi)) - torch.pi,
                - meas_dict['RobotVel'] - meas_dict[f"{self.object_name}Pos"][2:],
            ]).squeeze()
        else:
            return ((angle_diff + torch.pi) % (2*torch.pi)) - torch.pi

class Vel_AI(ActiveInterconnection):
    """
    Measurement Model:
    vel state:          robot vel
    vel obs:            robot vel observation
    """
    def __init__(self, estimators, device) -> None:
        required_estimators = ['RobotVel', 'vel_frontal', 'vel_lateral', 'vel_rot']
        super().__init__(estimators, required_estimators, device)

    def implicit_interconnection_model(self, meas_dict):
        return torch.stack([
            meas_dict['vel_frontal'] - meas_dict['RobotVel'][0],
            meas_dict['vel_lateral'] - meas_dict['RobotVel'][1],
            meas_dict['vel_rot'] - meas_dict['RobotVel'][2]
        ]).squeeze()

class Radius_Pos_VisAngle_AI(ActiveInterconnection):
    """
    Measurement Model:
    pos state:          target position and vel in robot frame
    radius state:       target radius
    visual angle obs:   perceived angle of obstacle in robot's visual field
    """
    def __init__(self, estimators: List[RecursiveEstimator], device, obstacle_id) -> None:
        self.obstacle_id = obstacle_id
        required_estimators = [f'Obstacle{self.obstacle_id}Pos', f'Obstacle{self.obstacle_id}Rad', f'obstacle{self.obstacle_id}_visual_angle']
        super().__init__(estimators, required_estimators, device)

    def implicit_interconnection_model(self, meas_dict):
        return torch.asin(torch.minimum(torch.ones_like(meas_dict[f'Obstacle{self.obstacle_id}Rad'][0]), meas_dict[f'Obstacle{self.obstacle_id}Rad'][0] / meas_dict[f'Obstacle{self.obstacle_id}Pos'][:2].norm())) - meas_dict[f'obstacle{self.obstacle_id}_visual_angle'] / 2
    
class Angle_Meas_AI(ActiveInterconnection):
    def __init__(self, estimators: List[RecursiveEstimator], device, object_name:str="Target", estimate_vel:bool=True) -> None:
        self.object_name = object_name
        self.estimate_vel = estimate_vel
        required_estimators = [f'Polar{object_name}Pos', f'{self.object_name[0].lower() + self.object_name[1:]}_offset_angle']
        if estimate_vel:
            required_estimators.append(f'del_{self.object_name[0].lower() + self.object_name[1:]}_offset_angle')
        super().__init__(estimators, required_estimators, device)

    def implicit_interconnection_model(self, meas_dict: Dict[str, torch.Tensor]):
        if self.estimate_vel:
            return torch.stack([
                (meas_dict[f'{self.object_name[0].lower() + self.object_name[1:]}_offset_angle'] - meas_dict[f'Polar{self.object_name}Pos'][1] + torch.pi) % (2*torch.pi) - torch.pi,
                meas_dict[f'del_{self.object_name[0].lower() + self.object_name[1:]}_offset_angle'] - meas_dict[f'Polar{self.object_name}Pos'][3],
            ]).squeeze()
        else:
            return (meas_dict[f'{self.object_name[0].lower() + self.object_name[1:]}_offset_angle'] - meas_dict[f'Polar{self.object_name}Pos'][1] + torch.pi) % (2*torch.pi) - torch.pi

class Triangulation_AI(ActiveInterconnection):
    def __init__(self, estimators: List[RecursiveEstimator], device, object_name:str="Target", estimate_vel:bool=False) -> None:
        self.object_name = object_name
        self.estimate_vel = estimate_vel
        required_estimators = [f'Polar{object_name}Pos', 'RobotVel']
        if not estimate_vel:
            required_estimators.append(f'del_{self.object_name[0].lower() + self.object_name[1:]}_offset_angle')
        super().__init__(estimators, required_estimators, device)

    def implicit_interconnection_model(self, meas_dict: Dict[str, torch.Tensor]):
        offset_angle = meas_dict[f'Polar{self.object_name}Pos'][1]
        robot_target_frame_rotation_matrix = torch.stack([
            torch.stack([torch.cos(-offset_angle), -torch.sin(-offset_angle)]),
            torch.stack([torch.sin(-offset_angle), torch.cos(-offset_angle)]),
        ]).squeeze()
        robot_target_frame_vel = torch.matmul(robot_target_frame_rotation_matrix, meas_dict['RobotVel'][:2])
        if self.estimate_vel:
            angular_vel = meas_dict[f'Polar{self.object_name}Pos'][3] + meas_dict['RobotVel'][2]
        else:
            angular_vel = meas_dict[f'del_{self.object_name[0].lower() + self.object_name[1:]}_offset_angle'] + meas_dict['RobotVel'][2]
        # TODO: tan or plain?
        #triangulated_distance = torch.abs(robot_target_frame_vel[1] / torch.tan(angular_vel))
        if robot_target_frame_vel[1] == 0 or angular_vel == 0.0:
            triangulated_distance = meas_dict[f'Polar{self.object_name}Pos'][0]
        else:
            triangulated_distance = torch.abs(robot_target_frame_vel[1] / angular_vel)

        if self.estimate_vel:
            return torch.stack([
                torch.atleast_1d(triangulated_distance - meas_dict[f'Polar{self.object_name}Pos'][0]),
                torch.atleast_1d(- robot_target_frame_vel[0] - meas_dict[f'Polar{self.object_name}Pos'][2]),
            ]).squeeze()
        else:
            return torch.atleast_1d(triangulated_distance - meas_dict[f'Polar{self.object_name}Pos'][0])

# TODO: making these two update each other is stupid, it seems
class Cartesian_Polar_AI(ActiveInterconnection):
    def __init__(self, estimators: List[RecursiveEstimator], device, object_name:str="Target", estimate_vel:bool=True) -> None:
        self.object_name = object_name
        required_estimators = [f'Polar{object_name}Pos', f'{object_name}Pos']
        super().__init__(estimators, required_estimators, device)

    def implicit_interconnection_model(self, meas_dict: Dict[str, torch.Tensor]):
        return torch.stack([
            meas_dict[f'Polar{self.object_name}Pos'][0] - meas_dict[f'{self.object_name}Pos'][:2].norm(),
            meas_dict[f'Polar{self.object_name}Pos'][1] - torch.atan2(meas_dict[f'{self.object_name}Pos'][1], meas_dict[f'{self.object_name}Pos'][0]),
            #TODO: add derivatives
        ]).squeeze()
