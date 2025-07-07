import torch
from components.aicon import SensorimotorRegularity
from components.helpers import world_to_rtf

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
    def __init__(self, object_name:str="Target") -> None:
        self.object_name = object_name
        sensory_components = [f"{object_name.lower()}_distance"]
        super().__init__(
            id                 = f"{object_name} Distance",
            state_component    = f"Polar{object_name}Pos",
            action_component   = "RobotVel",
            sensory_components = sensory_components,
        )

    def get_predicted_meas(self, state: torch.Tensor, action: torch.Tensor):
        return {
            f"{self.object_name.lower()}_distance": state[0]
        }
    
# --------------------------------------------------------------------------------------------------------
    
class DistanceDot_MM(SensorimotorRegularity):
    def __init__(self, object_name:str="Target", moving_object:bool=False) -> None:
        self.object_name = object_name
        self.moving_object = moving_object
        sensory_components = [f"{object_name.lower()}_distance", f"{object_name.lower()}_distance_dot"]
        super().__init__(
            id                 = f"{object_name} Distance",
            state_component    = f"Polar{object_name}Pos",
            action_component   = "RobotVel",
            sensory_components = sensory_components,
        )

    def get_predicted_meas(self, state: torch.Tensor, action: torch.Tensor):
        distance_dot = - world_to_rtf(action, state[1], state[2])[0]
        if self.moving_object: distance_dot += state[2]
        return {
            f"{self.object_name.lower()}_distance": state[0],
            f"{self.object_name.lower()}_distance_dot": distance_dot
        }

# --------------------------------------------------------------------------------------------------------

class Angle_MM(SensorimotorRegularity):
    # TODO: adjust to 3d
    def __init__(self, object_name:str="Target") -> None:
        self.object_name = object_name
        sensory_components = [f"{self.object_name.lower()}_phi", f"{self.object_name.lower()}_theta"]
        super().__init__(
            id                 = f"{object_name} Angle",
            state_component    = f"Polar{object_name}Pos",
            action_component   = "RobotVel",
            sensory_components = sensory_components,
        )

    def implicit_interconnection_model(self, meas_dict):
        key_phi = f"{self.object_name.lower()}_phi"
        key_theta = f"{self.object_name.lower()}_theta"
        return torch.atleast_1d(torch.stack([
            (self.get_predicted_meas(meas_dict[self.state_component], meas_dict[self.action_component])[key_phi] - meas_dict[key_phi] + torch.pi) % (2 * torch.pi) - torch.pi,
            (self.get_predicted_meas(meas_dict[self.state_component], meas_dict[self.action_component])[key_theta] - meas_dict[key_theta] + torch.pi) % (2 * torch.pi) - torch.pi
        ]).squeeze())

    def get_predicted_meas(self, state: torch.Tensor, action: torch.Tensor):
        return {
            f"{self.object_name.lower()}_phi": state[1],
            f"{self.object_name.lower()}_theta": state[2]
        }
    
# --------------------------------------------------------------------------------------------------------

class Triangulation_SMR(SensorimotorRegularity):
    def __init__(self, object_name:str="Target", moving_object:bool=False) -> None:
        self.object_name = object_name
        self.moving_object = moving_object
        sensory_components = [f"{self.object_name.lower()}_phi_dot", f"{self.object_name.lower()}_theta_dot"]
        super().__init__(
            id                 = f"{object_name} Triangulation",
            state_component    = f"Polar{self.object_name}Pos",
            action_component   = "RobotVel",
            sensory_components = sensory_components,
        )

    def implicit_interconnection_model(self, meas_dict):
        key_phi_dot = f"{self.object_name.lower()}_phi_dot"
        key_theta_dot = f"{self.object_name.lower()}_theta_dot"
        predicted_phi_dot = self.get_predicted_meas(meas_dict[self.state_component], meas_dict[self.action_component])[key_phi_dot]
        predicted_theta_dot = self.get_predicted_meas(meas_dict[self.state_component], meas_dict[self.action_component])[key_theta_dot]
        
        print(f"Predicted phi_dot: {predicted_phi_dot:.3f}, Predicted theta_dot: {predicted_theta_dot:.3f}")
        print(f"Measured phi_dot: {meas_dict[key_phi_dot].item():.3f}, Measured theta_dot: {meas_dict[key_theta_dot].item():.3f}")

        return torch.atleast_1d(torch.stack([
            (self.get_predicted_meas(meas_dict[self.state_component], meas_dict[self.action_component])[key_phi_dot] - meas_dict[key_phi_dot]),
            (self.get_predicted_meas(meas_dict[self.state_component], meas_dict[self.action_component])[key_theta_dot] - meas_dict[key_theta_dot])
        ]).squeeze())

    def get_predicted_meas(self, state: torch.Tensor, action: torch.Tensor):
        rtf_vel = world_to_rtf(action, state[1], state[2])
        if self.moving_object: lateral_vel -= torch.atleast_1d([0.0, state[3]*state[0], state[4]*state[0]]) # combine robot and target vel components
        return {
            f"{self.object_name.lower()}_phi_dot": - rtf_vel[1]/state[0],
            f"{self.object_name.lower()}_theta_dot": - rtf_vel[2]/state[0]
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