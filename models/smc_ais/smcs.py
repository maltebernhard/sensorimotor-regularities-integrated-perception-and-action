from typing import Dict
from components.measurement_model import SensorimotorContingency
from components.helpers import rotate_vector_2d
import torch

# ========================================================================================================

class FV_SMC(SensorimotorContingency):
    def __init__(self, state_component:str, action_component:str, sensory_components:list, fv_noise:dict, sensor_angle:float) -> None:
        self.fv_noise:dict = fv_noise
        self.sensor_angle = sensor_angle
        super().__init__(state_component, action_component, sensory_components)

    def get_foveal_noise(self, angle, key):
        if key in self.fv_noise:
            return (abs(angle) * self.fv_noise[key]/(self.sensor_angle/2))
        else:
            return 0.0

    def get_contingent_noise(self, state: torch.Tensor):
        return {key: self.get_foveal_noise(state[1], key) for key in self.required_observations}

# ----------------------------------------------------------------------

class Robot_Vel_MM(SensorimotorContingency):
    def __init__(self) -> None:
        sensory_components = ['vel_frontal', 'vel_lateral', 'vel_rot']
        super().__init__(
            state_component    = "RobotVel",
            action_component   = "RobotVel",
            sensory_components = sensory_components
        )

    def get_predicted_meas(self, state: torch.Tensor, action: torch.Tensor):
        return {
            'vel_frontal': state[0],
            'vel_lateral': state[1],
            'vel_rot':     state[2]
        }

class Distance_MM(FV_SMC):
    def __init__(self, object_name:str="Target", fv_noise:dict={}, sensor_angle:float=2*torch.pi) -> None:
        self.object_name = object_name
        sensory_components = [f"{object_name.lower()}_distance"]
        super().__init__(
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

class Angle_MM(FV_SMC):
    def __init__(self, object_name:str="Target", fv_noise:dict={}, sensor_angle:float=2*torch.pi) -> None:
        self.object_name = object_name
        sensory_components = [f"{self.object_name.lower()}_offset_angle"]
        super().__init__(
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
    
class Triangulation_SMC(FV_SMC):
    def __init__(self, object_name:str="Target", fv_noise:dict={}, sensor_angle:float=2*torch.pi) -> None:
        self.object_name = object_name
        sensory_components = [f"{self.object_name.lower()}_offset_angle_dot"]
        super().__init__(
            state_component    = f"Polar{self.object_name}Pos",
            action_component   = "RobotVel",
            sensory_components = sensory_components,
            sensor_angle       = sensor_angle,
            fv_noise           = fv_noise
        )

    def get_predicted_meas(self, state: torch.Tensor, action: torch.Tensor):
        rtf_vel = rotate_vector_2d(-state[1], action[:2])
        return{
            f"{self.object_name.lower()}_offset_angle_dot": - rtf_vel[1]/state[0] - action[2]
        }
    
class TriangulationVel_SMC(FV_SMC):
    def __init__(self, object_name:str="Target", fv_noise:dict={}, sensor_angle:float=2*torch.pi) -> None:
        self.object_name = object_name
        sensory_components = [f"{self.object_name.lower()}_offset_angle_dot"]
        super().__init__(
            state_component    = f"Polar{self.object_name}Pos",
            action_component   = "RobotVel",
            sensory_components = sensory_components,
            sensor_angle       = sensor_angle,
            fv_noise           = fv_noise
        )

    def get_predicted_meas(self, state: torch.Tensor, action: torch.Tensor):
        lateral_vel = rotate_vector_2d(-state[1], action[:2])[1] - state[3]*state[0]
        return {
            f"{self.object_name.lower()}_offset_angle_dot": - lateral_vel/state[0] - action[2]
        }
    
class Divergence_SMC(FV_SMC):
    def __init__(self, object_name:str="Target", fv_noise:dict={}, sensor_angle:float=2*torch.pi) -> None:
        self.object_name = object_name
        sensory_components = [f'{self.object_name.lower()}_visual_angle', f'{self.object_name.lower()}_visual_angle_dot']
        super().__init__(
            state_component    = f"Polar{self.object_name}Pos",
            action_component   = "RobotVel",
            sensory_components = sensory_components,
            sensor_angle       = sensor_angle,
            fv_noise           = fv_noise
        )

    def get_predicted_meas(self, state: torch.Tensor, action: torch.Tensor):
        rtf_vel = rotate_vector_2d(-state[1], action[:2])
        vis_angle = torch.asin(state[2] / state[0]) * 2 if not state[2] / state[0] >= 1.0 else torch.tensor(torch.pi-1e-3)
        return {
            f"{self.object_name.lower()}_visual_angle": vis_angle,
            f"{self.object_name.lower()}_visual_angle_dot": 2 / state[0] * torch.tan(vis_angle/2) * rtf_vel[0] if vis_angle < torch.pi-1e-3 else torch.tensor(0.0)
        }

class DivergenceVel_SMC(FV_SMC):
    def __init__(self, object_name:str="Target", fv_noise:dict={}, sensor_angle:float=2*torch.pi) -> None:
        self.object_name = object_name
        sensory_components = [f'{self.object_name.lower()}_visual_angle', f'{self.object_name.lower()}_visual_angle_dot']
        super().__init__(
            state_component    = f"Polar{self.object_name}Pos",
            action_component   = "RobotVel",
            sensory_components = sensory_components,
            sensor_angle       = sensor_angle,
            fv_noise           = fv_noise
        )

    def get_predicted_meas(self, state: torch.Tensor, action: torch.Tensor):
        rtf_vel = rotate_vector_2d(-state[1], action[:2]) - torch.stack([state[2], state[3]*state[0]])
        vis_angle = 2 * torch.asin(state[4] / state[0]) if not state[4] / state[0] >= 1.0 else torch.tensor(torch.pi-1e-3)
        return {
            f'{self.object_name.lower()}_visual_angle': vis_angle,
            f'{self.object_name.lower()}_visual_angle_dot': 2 / state[0] * torch.tan(vis_angle/2) * rtf_vel[0] if vis_angle < torch.pi-1e-3 else torch.tensor(0.0),
        }