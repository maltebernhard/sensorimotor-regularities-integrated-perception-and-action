from typing import Dict
from components.measurement_model import SensorimotorContingency
from components.helpers import rotate_vector_2d
import torch

# ========================================================================================================

class DroneEnv_SMC(SensorimotorContingency):
    def __init__(self, id: str, state_component:str, action_component:str, sensory_components:list, fv_noise:dict={}, sensor_angle:float={}) -> None:
        self.fv_noise:dict = fv_noise
        self.sensor_angle = sensor_angle
        super().__init__(id, state_component, action_component, sensory_components)

    def get_expected_meas_noise(self, buffer_dict: Dict[str,torch.Tensor]) -> Dict[str,torch.Tensor]:
        """
        returns the expected total sensor noise (tuple of mean and stddev) for each sensory component.
        OVERWRITE to include additional effects, such as noise scaling with value (e.g. for robot vel)
        """
        # derive expected noise from state estimate, not from sensor readings
        predicted_meas = self.get_predicted_meas(buffer_dict[self.state_component]['mean'], buffer_dict[self.action_component]['mean'])
        # add "contingent noise" to expectation according to some SMC (e.g. foveal vision)
        contingent_noise = self.get_contingent_noise(buffer_dict[self.state_component]['mean'])
        noise = {}
        for key, obs in self.connected_observations.items():
            if "vel" in key or ("distance" in key and not "dot" in key):
                noise[key] = (
                    obs.static_sensor_noise[0] + contingent_noise[key][0] * predicted_meas[key],
                    obs.static_sensor_noise[1] + contingent_noise[key][1] * torch.abs(predicted_meas[key])
                )
            elif "distance_dot" in key:
                noise[key] = (
                    obs.static_sensor_noise[0] + contingent_noise[key][0] * predicted_meas[key.replace("_dot", "")],
                    obs.static_sensor_noise[1] + contingent_noise[key][1] * torch.abs(predicted_meas[key.replace("_dot", "")])
                )
            # TODO: something about this scaling function doesn't work
            # I want to indicate to the robot that a higher value for rotational vel means more noise
            # however, it seems that this leads to weird behavior
            # elif "angle_dot" in key:
            #     noise[key] = (
            #         obs.static_sensor_noise[0] + contingent_noise[key][0] * buffer_dict[self.state_component]['mean'][1],
            #         obs.static_sensor_noise[1] + contingent_noise[key][1] * torch.abs(buffer_dict[self.state_component]['mean'][1])
            #     )
            else:
                noise[key] = (
                    obs.static_sensor_noise[0] + contingent_noise[key][0],
                    obs.static_sensor_noise[1] + contingent_noise[key][1]
                )
        return noise

    def get_foveal_noise(self, angle, key):
        if key in self.fv_noise:
            return (angle * self.fv_noise[key][0]/(self.sensor_angle/2), abs(angle) * self.fv_noise[key][1]/(self.sensor_angle/2))
        else:
            return (0.0, 0.0)

    def get_contingent_noise(self, state: torch.Tensor):
        return {key: self.get_foveal_noise(state[1], key) for key in self.required_observations}

# --------------------------------------------------------------------------------------------------------

class Robot_Vel_MM(DroneEnv_SMC):
    def __init__(self) -> None:
        sensory_components = ['vel_frontal', 'vel_lateral', 'vel_rot']
        super().__init__(
            id                 = "RobotVel",
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
    
# --------------------------------------------------------------------------------------------------------
    
class Distance_MM(DroneEnv_SMC):
    def __init__(self, object_name:str="Target", moving_object:bool=False, fv_noise:dict={}, sensor_angle:float=2*torch.pi) -> None:
        self.object_name = object_name
        self.moving_object = moving_object
        sensory_components = [f"{object_name.lower()}_distance"]
        if self.moving_object: sensory_components.append(f"{object_name.lower()}_distance_dot")
        super().__init__(
            id                 = f"{object_name} Distance",
            state_component    = f"Polar{object_name}Pos",
            action_component   = "RobotVel",
            sensory_components = sensory_components,
            sensor_angle       = sensor_angle,
            fv_noise           = fv_noise
        )

    def get_predicted_meas(self, state: torch.Tensor, action: torch.Tensor):
        ret = {
            f"{self.object_name.lower()}_distance": state[0]
        }
        if self.moving_object:
            ret[f"{self.object_name.lower()}_distance_dot"] = state[2] - rotate_vector_2d(-state[1], action[:2])[0]
        return ret

# --------------------------------------------------------------------------------------------------------

class Angle_MM(DroneEnv_SMC):
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

class Triangulation_SMC(DroneEnv_SMC):
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

class Divergence_SMC(DroneEnv_SMC):
    def __init__(self, object_name:str="Target",  moving_object:bool=False, fv_noise:dict={}, sensor_angle:float=2*torch.pi) -> None:
        self.object_name = object_name
        self.moving_object = moving_object
        sensory_components = [f'{self.object_name.lower()}_visual_angle', f'{self.object_name.lower()}_visual_angle_dot']
        super().__init__(
            id                 = f"{object_name} Divergence",
            state_component    = f"Polar{self.object_name}Pos",
            action_component   = "RobotVel",
            sensory_components = sensory_components,
            sensor_angle       = sensor_angle,
            fv_noise           = fv_noise
        )

    def get_predicted_meas(self, state: torch.Tensor, action: torch.Tensor):
        rtf_vel = rotate_vector_2d(-state[1], action[:2])
        if self.moving_object: rtf_vel -= torch.stack([state[2], state[3]*state[0]])
        radius = state[4] if self.moving_object else state[2]
        vis_angle = 2 * torch.asin(radius / state[0]) if -1.0 < radius / state[0] < 1.0 else torch.tensor(torch.pi-1e-3)
        return {
            f'{self.object_name.lower()}_visual_angle': vis_angle,
            f'{self.object_name.lower()}_visual_angle_dot': 2 / state[0] * torch.tan(vis_angle/2) * rtf_vel[0] if vis_angle < torch.pi-1e-3 else torch.tensor(0.0),
        }