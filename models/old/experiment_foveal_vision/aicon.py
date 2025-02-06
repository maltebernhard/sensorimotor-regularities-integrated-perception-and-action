from typing import Dict
import numpy as np
import torch

from components.aicon import DroneEnvAICON as AICON
from models.old.experiment_foveal_vision.estimators import Robot_Vel_Estimator, Polar_Pos_Estimator
from models.old.experiment_foveal_vision.active_interconnections import Triangulation_AI
from models.old.experiment_foveal_vision.helpers import get_foveal_noise
from models.old.experiment_foveal_vision.measurement_models import Distance_MM, Robot_Vel_MM, Angle_MM
from models.old.experiment_foveal_vision.goals import PolarGoToTargetFovealVisionGoal

# ========================================================================================================

class ExperimentFovealVisionAICON(AICON):
    def __init__(self, env_config):
        self.type = "Foveal Vision Experimental"
        super().__init__(env_config)

    def define_estimators(self):
        estimators = {
            "RobotVel":              Robot_Vel_Estimator(),
            "PolarTargetPos":        Polar_Pos_Estimator(),
            "PolarObstacle1Pos":     Polar_Pos_Estimator(object_name="Obstacle1"),
        }
        return estimators

    def define_measurement_models(self):
        return {
            "VelMM":               (Robot_Vel_MM(),                    ["RobotVel"]),
            "TargetAngleMM":       (Angle_MM(),                        ["PolarTargetPos"]),
            "Obstacle1AngleMM":    (Angle_MM(object_name="Obstacle1"), ["PolarObstacle1Pos"]),
            "TargetDistanceMM":    (Distance_MM(),                     ["PolarTargetPos"]),
            "Obstacle1DistanceMM": (Distance_MM(object_name="Obstacle1"), ["PolarObstacle1Pos"]),
        }

    def define_active_interconnections(self):
        active_interconnections = {
            "TriangulationAI":        Triangulation_AI(),
        }
        return active_interconnections

    def define_goals(self):
        goals = {
            "PolarGoToTarget": PolarGoToTargetFovealVisionGoal(),
        }
        return goals

    def eval_interconnections(self, buffer_dict: Dict[str, Dict[str, torch.Tensor]]):
        #self.REs["PolarTargetPos"].call_update_with_active_interconnection(self.AIs["TriangulationAI"], buffer_dict)
        return buffer_dict

    def compute_action_from_gradient(self, gradients):
        # TODO: improve timestep scaling of action generation
        decay = 0.9 ** (self.env_config["timestep"] / 0.05)
        gradient_action = decay * self.last_action - 6e-1 * self.env_config["timestep"] * gradients["PolarGoToTarget"]
        return gradient_action
    
    def print_estimators(self, buffer_dict=None):
        env_state = self.env.get_state()
        print("--------------------------------------------------------")
        self.print_estimator("PolarTargetPos", buffer_dict=buffer_dict, print_cov=2)
        print(f"True PolarTargetPos: [{env_state['target_distance']:.3f}, {env_state['target_offset_angle']:.3f}, {env_state['target_distance_dot']:.3f}, {env_state['target_offset_angle_dot']:.3f}]")
        print("--------------------------------------------------------")
        self.print_estimator("RobotVel", buffer_dict=buffer_dict) 
        print(f"True RobotVel: [{self.env.robot.vel[0]:.3f}, {self.env.robot.vel[1]:.3f}, {self.env.robot.vel_rot:.3f}]")
        print("--------------------------------------------------------")

    def render(self):
        target_mean, target_cov = self.convert_polar_to_cartesian_state(self.REs["PolarTargetPos"].mean, self.REs["PolarTargetPos"].cov)
        obs_mean, obs_cov = self.convert_polar_to_cartesian_state(self.REs["PolarObstacle1Pos"].mean, self.REs["PolarObstacle1Pos"].cov)
        estimator_means: Dict[str, torch.Tensor] = {"PolarTargetPos": target_mean, "PolarObstacle1Pos": obs_mean}
        estimator_covs: Dict[str, torch.Tensor] = {"PolarTargetPos": target_cov, "PolarObstacle1Pos": obs_cov}
        return self.env.render(1.0, {key: np.array(mean.cpu()) for key, mean in estimator_means.items()}, {key: np.array(cov.cpu()) for key, cov in estimator_covs.items()})

    def custom_reset(self):
        self.goals["PolarGoToTarget"].desired_distance = self.env.target.distance

    def get_observation_update_noise(self, key):
        if "angle" in key or "_rot" in key:
            update_uncertainty = 1e-1
        else:
            update_uncertainty = 5e-1
        return update_uncertainty * torch.eye(1)

    def get_custom_sensor_noise(self, obs: dict):
        observation_noise = {}
        for key in obs.keys():
            if key == "target_offset_angle":
                observation_noise[key] = get_foveal_noise(obs["target_offset_angle"], 0, self.env_config["fv_noise"], self.env_config["robot_sensor_angle"])
            elif key == "target_offset_angle_dot":
                observation_noise[key] = get_foveal_noise(obs["target_offset_angle"], 1, self.env_config["fv_noise"], self.env_config["robot_sensor_angle"])
            elif key == "obstacle1_offset_angle":
                observation_noise[key] = get_foveal_noise(obs["obstacle1_offset_angle"], 0, self.env_config["fv_noise"], self.env_config["robot_sensor_angle"])
            elif key == "obstacle1_offset_angle_dot":
                observation_noise[key] = get_foveal_noise(obs["obstacle1_offset_angle"], 1, self.env_config["fv_noise"], self.env_config["robot_sensor_angle"])
            else:
                observation_noise[key] = 0.0
        return {key: val*torch.eye(1) for key, val in observation_noise.items()}
        
    def adapt_contingent_measurements(self, buffer_dict: dict):
        predicted_angle_target = buffer_dict['PolarTargetPos']['mean'][1]
        buffer_dict['target_offset_angle']['mean']     = predicted_angle_target
        buffer_dict['target_offset_angle']['cov']      = get_foveal_noise(predicted_angle_target, 0, self.env_config["fv_noise"], self.env_config["robot_sensor_angle"]) ** 2
        buffer_dict['target_offset_angle_dot']['mean'] = buffer_dict['PolarTargetPos']['mean'][3]
        buffer_dict['target_offset_angle_dot']['cov']  = get_foveal_noise(predicted_angle_target, 1, self.env_config["fv_noise"], self.env_config["robot_sensor_angle"]) ** 2
        
        predicted_angle_obstacle = buffer_dict['PolarObstacle1Pos']['mean'][1]
        buffer_dict['obstacle1_offset_angle']['mean']     = predicted_angle_obstacle
        buffer_dict['obstacle1_offset_angle']['cov']      = get_foveal_noise(predicted_angle_obstacle, 0, self.env_config["fv_noise"], self.env_config["robot_sensor_angle"]) ** 2
        buffer_dict['obstacle1_offset_angle_dot']['mean'] = buffer_dict['PolarObstacle1Pos']['mean'][3]
        buffer_dict['obstacle1_offset_angle_dot']['cov']  = get_foveal_noise(predicted_angle_obstacle, 1, self.env_config["fv_noise"], self.env_config["robot_sensor_angle"]) ** 2

        buffer_dict['vel_frontal']['mean']             = buffer_dict['RobotVel']['mean'][0]
        buffer_dict['vel_lateral']['mean']             = buffer_dict['RobotVel']['mean'][1]
        buffer_dict['vel_rot']['mean']                 = buffer_dict['RobotVel']['mean'][2]