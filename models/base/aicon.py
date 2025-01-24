from typing import Dict
import torch

from components.aicon import DroneEnvAICON as AICON
from models.base.estimators import Robot_Vel_Estimator, Polar_Pos_Estimator
from models.base.active_interconnections import Triangulation_AI
from models.base.helpers import get_foveal_noise
from models.base.measurement_models import Robot_Vel_MM, Angle_MM
from models.base.goals import PolarGoToTargetGoal

# ========================================================================================================

class BaseAICON(AICON):
    def __init__(self, env_config):
        self.type = "Base"
        super().__init__(env_config)

    def define_estimators(self):
        estimators = {
            "RobotVel":         Robot_Vel_Estimator(),
            "PolarTargetPos":   Polar_Pos_Estimator(),
        }
        return estimators

    def define_measurement_models(self):
        return {
            "VelMM":        (Robot_Vel_MM(), ["RobotVel"]),
            "AngleMeasMM":  (Angle_MM(),     ["PolarTargetPos"]),
        }

    def define_active_interconnections(self):
        active_interconnections = {
            "TriangulationAI": Triangulation_AI(),
        }
        return active_interconnections

    def define_goals(self):
        goals = {
            "PolarGoToTarget": PolarGoToTargetGoal(),
        }
        return goals

    def eval_interconnections(self, buffer_dict: Dict[str, Dict[str, torch.Tensor]]):
        self.REs["PolarTargetPos"].call_update_with_active_interconnection(self.AIs["TriangulationAI"], buffer_dict)
        return buffer_dict

    def compute_action_from_gradient(self, gradients):
        # TODO: improve timestep scaling of action generation
        decay = 0.9 ** (self.env_config["timestep"] / 0.05)
        gradient_action = decay * self.last_action - 1e-1 * self.env_config["timestep"] * gradients["PolarGoToTarget"]
        return gradient_action
    
    def print_estimators(self, buffer_dict=None):
        env_state = self.env.get_state()
        print("--------------------------------------------------------------------")
        self.print_estimator("PolarTargetPos", buffer_dict=buffer_dict, print_cov=2)
        print(f"True PolarTargetPos: [{env_state['target_distance']:.3f}, {env_state['target_offset_angle']:.3f}, {env_state['target_distance_dot']:.3f}, {env_state['target_offset_angle_dot']:.3f}, {env_state['target_radius']:.3f}]")
        print("--------------------------------------------------------------------")
        self.print_estimator("RobotVel", buffer_dict=buffer_dict) 
        print(f"True RobotVel: [{self.env.robot.vel[0]:.3f}, {self.env.robot.vel[1]:.3f}, {self.env.robot.vel_rot:.3f}]")
        print("--------------------------------------------------------------------")

    def custom_reset(self):
        self.goals["PolarGoToTarget"].desired_distance = self.env.target.distance

    def get_observation_update_uncertainty(self, key):
        if "angle" in key or "_rot" in key:
            update_uncertainty: torch.Tensor = 1e-1 * torch.eye(1)
        else:
            update_uncertainty: torch.Tensor = 5e-1 * torch.eye(1)
        return update_uncertainty
    
    def get_custom_sensor_noise(self, obs: dict):
        observation_noise = {}
        for key in obs.keys():
            if   key == "target_offset_angle":        observation_noise[key] = get_foveal_noise(obs["target_offset_angle"], 0, self.env_config["foveal_vision_noise"], self.env_config["robot_sensor_angle"])
            elif key == "target_offset_angle_dot":    observation_noise[key] = get_foveal_noise(obs["target_offset_angle"], 1, self.env_config["foveal_vision_noise"], self.env_config["robot_sensor_angle"])
            else: observation_noise[key] = 0.0
        return {key: val*torch.eye(1) for key, val in observation_noise.items()}
    
    def adapt_contingent_measurements(self, buffer_dict: dict):
        predicted_angle = buffer_dict['PolarTargetPos']['state_mean'][1]
        buffer_dict['target_offset_angle']['state_mean']     = predicted_angle
        buffer_dict['target_offset_angle']['state_cov']      = get_foveal_noise(predicted_angle, 0, self.env_config["foveal_vision_noise"], self.env_config["robot_sensor_angle"]) ** 2
        buffer_dict['target_offset_angle_dot']['state_mean'] = buffer_dict['PolarTargetPos']['state_mean'][3]
        buffer_dict['target_offset_angle_dot']['state_cov']  = get_foveal_noise(predicted_angle, 1, self.env_config["foveal_vision_noise"], self.env_config["robot_sensor_angle"]) ** 2
        buffer_dict['vel_frontal']['state_mean']             = buffer_dict['RobotVel']['state_mean'][0]
        buffer_dict['vel_lateral']['state_mean']             = buffer_dict['RobotVel']['state_mean'][1]
        buffer_dict['vel_rot']['state_mean']                 = buffer_dict['RobotVel']['state_mean'][2]
