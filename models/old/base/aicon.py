from typing import Dict
import torch

from components.aicon import DroneEnvAICON as AICON
from models.old.base.estimators import Robot_Vel_Estimator, Polar_Pos_Estimator
from models.old.base.active_interconnections import Triangulation_AI
from models.old.base.helpers import get_foveal_noise
from models.old.base.measurement_models import Robot_Vel_MM, Angle_MM
from models.old.base.goals import PolarGoToTargetGoal

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
        self.print_estimator("PolarTargetPos", buffer_dict=buffer_dict, print_cov=1)
        print(f"True PolarTargetPos: [{env_state['target_distance']:.3f}, {env_state['target_offset_angle']:.3f}, {env_state['target_distance_dot']:.3f}, {env_state['target_offset_angle_dot']:.3f}]")#, {env_state['target_radius']:.3f}]")
        print("--------------------------------------------------------------------")
        self.print_estimator("RobotVel", buffer_dict=buffer_dict) 
        print(f"True RobotVel: [{self.env.robot.vel[0]:.3f}, {self.env.robot.vel[1]:.3f}, {self.env.robot.vel_rot:.3f}]")
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
            if   key == "target_offset_angle"     and key in self.env_config["fv_noise"].keys(): observation_noise[key] = get_foveal_noise(obs["target_offset_angle"], 0, self.env_config["fv_noise"], self.env_config["robot_sensor_angle"])
            elif key == "target_offset_angle_dot" and key in self.env_config["fv_noise"].keys(): observation_noise[key] = get_foveal_noise(obs["target_offset_angle"], 1, self.env_config["fv_noise"], self.env_config["robot_sensor_angle"])
            else: observation_noise[key] = 0.0
        return {key: val*torch.eye(1) for key, val in observation_noise.items()}
    
    def adapt_contingent_measurements(self, buffer_dict: dict):
        predicted_angle = buffer_dict['PolarTargetPos']['mean'][1]
        buffer_dict['target_offset_angle']['mean']     = predicted_angle
        buffer_dict['target_offset_angle_dot']['mean'] = buffer_dict['PolarTargetPos']['mean'][3]
        buffer_dict['vel_frontal']['mean']             = buffer_dict['RobotVel']['mean'][0]
        buffer_dict['vel_lateral']['mean']             = buffer_dict['RobotVel']['mean'][1]
        buffer_dict['vel_rot']['mean']                 = buffer_dict['RobotVel']['mean'][2]
        if "target_offset_angle" in self.env_config["fv_noise"].keys():
            buffer_dict['target_offset_angle']['cov']     = 10 * (self.obs["target_offset_angle"].sensor_noise + get_foveal_noise(predicted_angle, 0, self.env_config["fv_noise"], self.env_config["robot_sensor_angle"]))# ** 2
        if "target_offset_angle_dot" in self.env_config["fv_noise"].keys():
            buffer_dict['target_offset_angle_dot']['cov'] = 10 * (self.obs["target_offset_angle_dot"].sensor_noise + get_foveal_noise(predicted_angle, 1, self.env_config["fv_noise"], self.env_config["robot_sensor_angle"]))# ** 2
            print(f"Cont. meas angle: {buffer_dict['target_offset_angle']['cov'].item():.3f} | {self.obs["target_offset_angle"].sensor_noise.item():.3f} , {get_foveal_noise(predicted_angle, 0, self.env_config["fv_noise"], self.env_config["robot_sensor_angle"]):.3f}")
            print(f"Cont. meas angle_dot: {buffer_dict['target_offset_angle_dot']['cov'].item():.3f} | {self.obs["target_offset_angle_dot"].sensor_noise.item():.3f} , {get_foveal_noise(predicted_angle, 1, self.env_config["fv_noise"], self.env_config["robot_sensor_angle"]):.3f}")
