import torch
from components.aicon import SensorimotorRegularity
from components.helpers import rotate_vector_2d

# ==================================================================================================

class Robot_Vel_MM(SensorimotorRegularity):
    def __init__(self) -> None:
        sensory_components = ['vel_x', 'vel_y', 'vel_z']
        super().__init__(
            id                 = "RobotVel",
            state_component    = "RobotVel",
            action_component   = "RobotVel",
            sensory_components = sensory_components
        )

    def get_predicted_meas(self, state: torch.Tensor, action: torch.Tensor):
        return {
            'vel_x': state[0],
            'vel_y': state[1],
            'vel_z': state[2]
        }
    
# --------------------------------------------------------------------------------------------------------
    
class Distance_MM(SensorimotorRegularity):
    def __init__(self, object_name:str="Target", moving_object:bool=False, fv_noise:dict={}, sensor_angle:float=2*torch.pi) -> None:
        self.object_name = object_name
        self.moving_object = moving_object
        sensory_components = [f"{object_name.lower()}_distance"]
        
        # TODO: consider whether or not to add distance dot sensor
        # if self.moving_object: sensory_components.append(f"{object_name.lower()}_distance_dot")
        super().__init__(
            id                 = f"{object_name} Distance",
            state_component    = f"Polar{object_name}Pos",
            action_component   = "RobotVel",
            sensory_components = sensory_components,
            sensor_angle       = sensor_angle,
            fv_noise           = fv_noise
        )

    def get_predicted_meas(self, state: torch.Tensor, action: torch.Tensor):
        return {
            f"{self.object_name.lower()}_distance": state[0]
        }
    
# --------------------------------------------------------------------------------------------------------
    
class DistanceDot_MM(SensorimotorRegularity):
    def __init__(self, object_name:str="Target", moving_object:bool=False, fv_noise:dict={}, sensor_angle:float=2*torch.pi) -> None:
        self.object_name = object_name
        self.moving_object = moving_object
        sensory_components = [f"{object_name.lower()}_distance", f"{object_name.lower()}_distance_dot"]
        super().__init__(
            id                 = f"{object_name} Distance",
            state_component    = f"Polar{object_name}Pos",
            action_component   = "RobotVel",
            sensory_components = sensory_components,
            sensor_angle       = sensor_angle,
            fv_noise           = fv_noise
        )

    def get_predicted_meas(self, state: torch.Tensor, action: torch.Tensor):
        distance_dot = - rotate_vector_2d(-state[1], action[:2])[0]
        if self.moving_object: distance_dot += state[2]
        return {
            f"{self.object_name.lower()}_distance": state[0],
            f"{self.object_name.lower()}_distance_dot": distance_dot
        }

# --------------------------------------------------------------------------------------------------------

class Angle_MM(SensorimotorRegularity):
    # TODO: adjust to 3d
    def __init__(self, object_name:str="Target", fv_noise:dict={}, sensor_angle:float=2*torch.pi) -> None:
        self.object_name = object_name
        sensory_components = [f"{self.object_name.lower()}_offset_angle"]
        super().__init__(
            id                 = f"{object_name} Angle",
            state_component    = f"Polar{object_name}Pos",
            action_component   = "RobotVel",
            sensory_components = sensory_components,
            sensor_angle       = sensor_angle,
            fv_noise           = fv_noise
        )

    def implicit_interconnection_model(self, meas_dict):
        # overwritten to handle the case when angles like -0.9*pi and 0.9*pi are compared
        key = f"{self.object_name.lower()}_offset_angle"
        return torch.atleast_1d(self.get_predicted_meas(meas_dict[self.state_component], meas_dict[self.action_component])[key] - meas_dict[key] + torch.pi) % (2 * torch.pi) - torch.pi

    def get_predicted_meas(self, state: torch.Tensor, action: torch.Tensor):
        return {
            f"{self.object_name.lower()}_offset_angle": state[1]
        }
    
# --------------------------------------------------------------------------------------------------------

class Triangulation_SMR(SensorimotorRegularity):
    # TODO: adjust to 3d
    def __init__(self, object_name:str="Target", moving_object:bool=False, fv_noise:dict={}, sensor_angle:float=2*torch.pi) -> None:
        self.object_name = object_name
        self.moving_object = moving_object
        sensory_components = [f"{self.object_name.lower()}_offset_angle_dot"]
        super().__init__(
            id                 = f"{object_name} Triangulation",
            state_component    = f"Polar{self.object_name}Pos",
            action_component   = "RobotVel",
            sensory_components = sensory_components,
            sensor_angle       = sensor_angle,
            fv_noise           = fv_noise
        )

    def get_predicted_meas(self, state: torch.Tensor, action: torch.Tensor):
        lateral_vel = rotate_vector_2d(-state[1], action[:2])[1]
        if self.moving_object: lateral_vel -= state[3]*state[0]     # combine robot and target vel components
        return {
            f"{self.object_name.lower()}_offset_angle_dot": - lateral_vel/state[0] - action[2]
        }

# --------------------------------------------------------------------------------------------------------

# class Divergence_SMR(SensorimotorRegularity):
#     # TODO: adjust to 3d
#     def __init__(self, object_name:str="Target",  moving_object:bool=False, fv_noise:dict={}, sensor_angle:float=2*torch.pi) -> None:
#         self.object_name = object_name
#         self.moving_object = moving_object
#         sensory_components = [f'{self.object_name.lower()}_visual_angle', f'{self.object_name.lower()}_visual_angle_dot']
#         super().__init__(
#             id                 = f"{object_name} Divergence",
#             state_component    = f"Polar{self.object_name}Pos",
#             action_component   = "RobotVel",
#             sensory_components = sensory_components,
#             sensor_angle       = sensor_angle,
#             fv_noise           = fv_noise
#         )

#     def get_predicted_meas(self, state: torch.Tensor, action: torch.Tensor):
#         distance_dot = rotate_vector_2d(-state[1], action[:2])[0]
#         if self.moving_object: distance_dot -= state[2]
#         radius = state[4] if self.moving_object else state[2]
#         vis_angle = 2 * torch.asin(radius / state[0]) if -1.0 < radius / state[0] < 1.0 else torch.tensor(torch.pi-1e-3)
#         return {
#             f'{self.object_name.lower()}_visual_angle': vis_angle,
#             f'{self.object_name.lower()}_visual_angle_dot': 2 / state[0] * torch.tan(vis_angle/2) * distance_dot if vis_angle < torch.pi-1e-3 else torch.tensor(0.0),
#         }