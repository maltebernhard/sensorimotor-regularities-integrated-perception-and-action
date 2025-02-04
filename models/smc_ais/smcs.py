from components.measurement_model import SensorimotorContingency
from components.helpers import rotate_vector_2d
from models.smc_ais.helpers import get_foveal_noise
import torch

# ========================================================================================================

class Distance_MM(SensorimotorContingency):
    def __init__(self, object_name:str="Target", sensor_noise:dict={}, foveal_vision_noise:dict={}) -> None:
        self.object_name = object_name
        self.sensor_noise = sensor_noise
        self.foveal_vision_noise = foveal_vision_noise
        sensory_components = [f"{object_name.lower()}_distance"]
        super().__init__(
            state_component    = f"Polar{object_name}Pos",
            action_component   = "RobotVel",
            sensory_components = sensory_components
        )

    def transform_state_to_innovation_space(self, state: torch.Tensor, action: torch.Tensor):
        if f"{self.object_name.lower()}_distance" in self.foveal_vision_noise.keys():
            covs = [(get_foveal_noise(state[1], f"{self.object_name.lower()}_distance", self.foveal_vision_noise, 2*torch.pi) + torch.tensor(self.sensor_noise[f"{self.object_name.lower()}_distance"]))**2]
        else:
            covs = None
        return torch.atleast_1d(state[0]), covs


class Robot_Vel_MM(SensorimotorContingency):
    def __init__(self) -> None:
        sensory_components = ['vel_frontal', 'vel_lateral', 'vel_rot']
        super().__init__(
            state_component    = "RobotVel",
            action_component   = "RobotVel",
            sensory_components = sensory_components
        )

    def transform_state_to_innovation_space(self, state: torch.Tensor, action: torch.Tensor):
        return state, None

class Angle_MM(SensorimotorContingency):
    def __init__(self, object_name:str="Target", sensor_noise:dict={}, foveal_vision_noise:dict={}) -> None:
        self.object_name = object_name
        self.sensor_noise = sensor_noise
        self.foveal_vision_noise = foveal_vision_noise
        sensory_components = [f"{self.object_name.lower()}_offset_angle"]
        super().__init__(
            state_component    = f"Polar{object_name}Pos",
            action_component   = "RobotVel",
            sensory_components = sensory_components
        )

    def implicit_interconnection_model(self, meas_dict):
        return (self.transform_state_to_innovation_space(meas_dict[self.state_component], meas_dict[self.action_component])[0] - self.transform_measurements_to_innovation_space(meas_dict) + torch.pi) % (2 * torch.pi) - torch.pi

    def transform_state_to_innovation_space(self, state: torch.Tensor, action: torch.Tensor):
        if f"{self.object_name.lower()}_offset_angle" in self.foveal_vision_noise.keys():
            covs = [(get_foveal_noise(state[1], f"{self.object_name.lower()}_offset_angle", self.foveal_vision_noise, 2*torch.pi) + torch.tensor(self.sensor_noise[f"{self.object_name.lower()}_offset_angle"]))**2]
        else:
            covs = None
        return torch.atleast_1d(state[1]), covs
    
class Triangulation_SMC(SensorimotorContingency):
    def __init__(self, object_name:str="Target", sensor_noise:dict={}, foveal_vision_noise:dict={}) -> None:
        self.object_name = object_name
        self.sensor_noise = sensor_noise
        self.foveal_vision_noise = foveal_vision_noise
        sensory_components = [f"{self.object_name.lower()}_offset_angle_dot"]
        super().__init__(
            state_component    = f"Polar{self.object_name}Pos",
            action_component   = "RobotVel",
            sensory_components = sensory_components
        )

    def transform_state_to_innovation_space(self, state: torch.Tensor, action: torch.Tensor):
        rtf_vel = rotate_vector_2d(-state[1], action[:2])
        meas = - rtf_vel[1]/state[0] - action[2]
        if f"{self.object_name.lower()}_offset_angle_dot" in self.foveal_vision_noise.keys():
            covs = [(get_foveal_noise(state[1], f"{self.object_name.lower()}_offset_angle_dot", self.foveal_vision_noise, 2*torch.pi) + torch.tensor(self.sensor_noise[f"{self.object_name.lower()}_offset_angle_dot"]))**2]
        else:
            covs = None
        return torch.atleast_1d(meas), covs
    
class Divergence_SMC(SensorimotorContingency):
    def __init__(self, object_name:str="Target", sensor_noise:dict={}, foveal_vision_noise:dict={}) -> None:
        self.object_name = object_name
        self.sensor_noise = sensor_noise
        self.foveal_vision_noise = foveal_vision_noise
        sensory_components = [f'{self.object_name.lower()}_visual_angle', f'{self.object_name.lower()}_visual_angle_dot']
        super().__init__(
            state_component    = f"Polar{self.object_name}Pos",
            action_component   = "RobotVel",
            sensory_components = sensory_components
        )

    def transform_state_to_innovation_space(self, state: torch.Tensor, action: torch.Tensor):
        rtf_vel = rotate_vector_2d(-state[1], action[:2])
        meas = [
            state[2],
            2 / state[0] * torch.tan(state[2]/2) * rtf_vel[0] if state[2] < 2*torch.pi else torch.tensor(0.0)
        ]
        
        fv_keys = [f'{self.object_name.lower()}_visual_angle', f'{self.object_name.lower()}_visual_angle_dot']
        if all(key in self.foveal_vision_noise.keys() for key in fv_keys):
            covs = []
            for key in fv_keys:
                covs.append((get_foveal_noise(state[1], key, self.foveal_vision_noise, 2*torch.pi) + torch.tensor(self.sensor_noise[key]))**2)
        else:
            covs = None

        return torch.stack(meas).squeeze(), covs
