from typing import Dict
import torch

from components.aicon import DroneEnvAICON as AICON
from models.single_estimator.estimators import Polar_Pos_Estimator
from models.single_estimator.helpers import get_foveal_noise
from models.single_estimator.measurement_models import Angle_MM
from models.single_estimator.goals import PolarGoToTargetGoal

# ========================================================================================================

class SingleEstimatorAICON(AICON):
    def __init__(self, env_config):
        self.type = "SingleEstimator"
        super().__init__(env_config)

    def define_estimators(self):
        estimators = {
            "PolarTargetPos":   Polar_Pos_Estimator(),
        }
        return estimators

    def define_measurement_models(self):
        return {
            "AngleMeasMM":  (Angle_MM(),     ["PolarTargetPos"]),
        }

    def define_active_interconnections(self):
        active_interconnections = {}
        return active_interconnections

    def define_goals(self):
        goals = {
            "PolarGoToTarget": PolarGoToTargetGoal(),
        }
        return goals

    def eval_interconnections(self, buffer_dict: Dict[str, Dict[str, torch.Tensor]]):
        return buffer_dict

    def compute_action_from_gradient(self, gradients):
        # TODO: improve timestep scaling of action generation
        decay = 0.9 ** (self.env_config["timestep"] / 0.05)
        gradient_action = decay * self.last_action - 1e-1 * self.env_config["timestep"] * gradients["PolarGoToTarget"]
        return gradient_action
    
    def print_estimators(self, buffer_dict=None):
        env_state = self.env.get_state()
        print("--------------------------------------------------------------------")
        self.print_estimator("PolarTargetPos", buffer_dict=buffer_dict, print_cov=1)
        print(f"True PolarTargetPos: [{env_state['target_distance']:.3f}, {env_state['target_offset_angle']:.3f}, {env_state['target_distance_dot']:.3f}, {env_state['target_offset_angle_dot']+env_state['vel_rot']:.3f}]")#, {env_state['target_radius']:.3f}]")
        print("--------------------------------------------------------------------")

    def custom_reset(self):
        self.goals["PolarGoToTarget"].desired_distance = self.env.target.distance

    # def get_observation_update_uncertainty(self, key):
    #     if "angle" in key or "_rot" in key:
    #         update_uncertainty: torch.Tensor = 1e-1 * torch.eye(1)
    #     else:
    #         update_uncertainty: torch.Tensor = 5e-1 * torch.eye(1)
    #     return update_uncertainty
    
    def get_custom_sensor_noise(self, obs: dict):
        observation_noise = {}
        for key in obs.keys():
            if   key == "target_offset_angle"     and key in self.env_config["foveal_vision_noise"].keys(): observation_noise[key] = get_foveal_noise(obs["target_offset_angle"], 0, self.env_config["foveal_vision_noise"], self.env_config["robot_sensor_angle"])
            elif key == "target_offset_angle_dot" and key in self.env_config["foveal_vision_noise"].keys(): observation_noise[key] = get_foveal_noise(obs["target_offset_angle"], 1, self.env_config["foveal_vision_noise"], self.env_config["robot_sensor_angle"])
            else: observation_noise[key] = 0.0
        return {key: val*torch.eye(1) for key, val in observation_noise.items()}
    
    def adapt_contingent_measurements(self, buffer_dict: dict):
        predicted_angle = buffer_dict['PolarTargetPos']['state_mean'][1]
        buffer_dict['target_offset_angle']['state_mean']     = predicted_angle
        buffer_dict['target_offset_angle_dot']['state_mean'] = buffer_dict['PolarTargetPos']['state_mean'][3]
        if "target_offset_angle" in self.env_config["foveal_vision_noise"].keys():
            buffer_dict['target_offset_angle']['state_cov']     = buffer_dict['PolarTargetPos']['state_cov'][1,1] + get_foveal_noise(predicted_angle, 0, self.env_config["foveal_vision_noise"], self.env_config["robot_sensor_angle"])# ** 2
        else:
            buffer_dict['target_offset_angle']['state_cov']     = buffer_dict['PolarTargetPos']['state_cov'][1,1]
        if "target_offset_angle_dot" in self.env_config["foveal_vision_noise"].keys():
            buffer_dict['target_offset_angle_dot']['state_cov'] = buffer_dict['PolarTargetPos']['state_cov'][3,3] + get_foveal_noise(predicted_angle, 1, self.env_config["foveal_vision_noise"], self.env_config["robot_sensor_angle"])# ** 2
        else:
            buffer_dict['target_offset_angle_dot']['state_cov'] = buffer_dict['PolarTargetPos']['state_cov'][3,3]
            # print(f"Cont. angle     sensor_noise: {self.obs["target_offset_angle"].sensor_noise.item():.3f} | fv_noise: {get_foveal_noise(predicted_angle, 0, self.env_config["foveal_vision_noise"], self.env_config["robot_sensor_angle"]):.3f}")
            # print(f"Cont. angle_dot sensor_noise: {self.obs["target_offset_angle_dot"].sensor_noise.item():.3f} | fv_noise: {get_foveal_noise(predicted_angle, 1, self.env_config["foveal_vision_noise"], self.env_config["robot_sensor_angle"]):.3f}")
