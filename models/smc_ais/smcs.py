from components.measurement_model import SensorimotorContingency
from components.helpers import rotate_vector_2d
from models.smc_ais.helpers import get_foveal_noise
import torch

# ========================================================================================================

class SMC_MM(SensorimotorContingency):
    def __init__(self, object_name:str="Target", smcs=[], sensor_noise:dict={}, foveal_vision_noise:dict={}) -> None:
        self.object_name = object_name
        self.smcs = smcs
        self.sensor_noise = sensor_noise
        self.foveal_vision_noise = foveal_vision_noise
        sensory_components = [f"{self.object_name.lower()}_offset_angle"]
        if 'Triangulation' in smcs: sensory_components += [f"{self.object_name.lower()}_offset_angle_dot"]
        if 'Divergence' in smcs: sensory_components    += [f'{self.object_name.lower()}_visual_angle', f'{self.object_name.lower()}_visual_angle_dot']
        super().__init__(
            state_component    = f"Polar{self.object_name}Pos",
            action_component   = "RobotVel",
            sensory_components = sensory_components
        )

    # def transform_measurements_to_innovation_space(self, meas_dict: dict):
    #     return torch.stack([
    #         meas_dict[f'{self.object_name.lower()}_offset_angle'],
    #         meas_dict[f'{self.object_name.lower()}_offset_angle_dot'],
    #         meas_dict[f'{self.object_name.lower()}_visual_angle_dot']/torch.tan(meas_dict[f'{self.object_name.lower()}_visual_angle']/2)
    #     ]).squeeze()

    def transform_state_to_innovation_space(self, state: torch.Tensor, action: torch.Tensor):
        rtf_vel = rotate_vector_2d(-state[1], action[:2])
        meas = [state[1]]
        if 'Triangulation' in self.smcs:
            meas += [
                - rtf_vel[1]/state[0] - action[2],
            ]
        if 'Divergence' in self.smcs:
            meas += [
                state[2],
                2 / state[0] * torch.tan(state[2]/2) * rtf_vel[0]
            ]
        
        if "FovealVision" in self.smcs:
            covs = torch.stack([(get_foveal_noise(state[1], obs, self.foveal_vision_noise, 2*torch.pi) + (torch.tensor(self.sensor_noise[obs]) if obs in self.sensor_noise.keys() else torch.zeros(1)))**2 if obs in self.foveal_vision_noise.keys() else (torch.tensor(self.sensor_noise[obs]) if obs in self.sensor_noise.keys() else torch.zeros(1)).pow(2) for obs in self.required_observations]).squeeze()
        else:
            covs = None

        return torch.stack(meas).squeeze(), covs