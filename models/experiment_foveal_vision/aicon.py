from typing import Dict
import numpy as np
import torch

from components.aicon import DroneEnvAICON as AICON
from components.helpers import rotate_vector_2d
from models.experiment_foveal_vision.estimators import Robot_Vel_Estimator, Polar_Pos_Estimator, Foveal_Vision_Estimator
from models.experiment_foveal_vision.active_interconnections import Foveal_Angle_AI, Triangulation_AI
from models.experiment_foveal_vision.helpers import get_foveal_noise
from models.experiment_foveal_vision.measurement_models import Distance_MM, Robot_Vel_MM, Angle_MM
from models.experiment_foveal_vision.goals import PolarGoToTargetFovealVisionGoal

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
            "TargetFovealVision":    Foveal_Vision_Estimator(foveal_vision_noise=self.env_config["foveal_vision_noise"], sensor_angle=self.env_config["robot_sensor_angle"]),
            "Obstacle1FovealVision": Foveal_Vision_Estimator(object_name="Obstacle1", foveal_vision_noise=self.env_config["foveal_vision_noise"], sensor_angle=self.env_config["robot_sensor_angle"]),
        }
        return estimators

    def define_measurement_models(self):
        return {
            "VelMM":               (Robot_Vel_MM(),                    ["RobotVel"]),
            "TargetAngleMM":       (Angle_MM(),                        ["TargetFovealVision"]),
            "Obstacle1AngleMM":    (Angle_MM(object_name="Obstacle1"), ["Obstacle1FovealVision"]),
            "TargetDistanceMM":    (Distance_MM(),                     ["PolarTargetPos"]),
            "Obstacle1DistanceMM": (Distance_MM(object_name="Obstacle1"), ["PolarObstacle1Pos"]),
        }

    def define_active_interconnections(self):
        active_interconnections = {
            "TriangulationAI":        Triangulation_AI(),
            "TargetFovealVisionAI":   Foveal_Angle_AI(),
            "Obstacle1FovealVisionAI": Foveal_Angle_AI(object_name="Obstacle1"),
        }
        return active_interconnections

    def define_goals(self):
        def select_goal():
            return PolarGoToTargetFovealVisionGoal()
        goals = {
            "PolarGoToTarget": select_goal(),
        }
        
        return goals

    def eval_interconnections(self, buffer_dict: Dict[str, Dict[str, torch.Tensor]]):
        # PROBLEM: After a new step, a measurement update updates the foveal vision estimators with a falsely "certain" measurement
        # HACK: To avoid this, update either revert this update or make sure that it happens with an uncertain measurement
        # NOTE: doesn't work like this, because angular vel is being adapted by fake action

        # new step:
        # 1. forward model - foveal vision uncertainty
        # 2. meas updates  - state cov becomes falsely small - solution: overwrite uncertainty in meas updates

        self.REs["PolarTargetPos"].call_update_with_active_interconnection(self.AIs["TargetFovealVisionAI"], buffer_dict)
        self.REs["PolarObstacle1Pos"].call_update_with_active_interconnection(self.AIs["Obstacle1FovealVisionAI"], buffer_dict)
        #self.REs["PolarTargetPos"].call_update_with_active_interconnection(self.AIs["TriangulationAI"], buffer_dict)
        
        # self.print_vector(buffer_dict["TargetFovealVision"]["state_mean"], "Post AI Mean")
        # self.print_vector(buffer_dict["TargetFovealVision"]["state_cov"].diag().sqrt(), "Post AI Uct")
        
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
        self.print_estimator("TargetFovealVision", buffer_dict=buffer_dict, print_cov=2)
        print(f"True TargetFovealVision: [{env_state['target_offset_angle']:.3f}, {env_state['target_offset_angle_dot']:.3f}]")
        print("--------------------------------------------------------")
        self.print_estimator("Obstacle1FovealVision", buffer_dict=buffer_dict, print_cov=2)
        print(f"True Obstacle1FovealVision: [{env_state['obstacle1_offset_angle']:.3f}, {env_state['obstacle1_offset_angle_dot']:.3f}]")
        print("--------------------------------------------------------")
        self.print_estimator("RobotVel", buffer_dict=buffer_dict) 
        print(f"True RobotVel: [{self.env.robot.vel[0]:.3f}, {self.env.robot.vel[1]:.3f}, {self.env.robot.vel_rot:.3f}]")
        print("--------------------------------------------------------")

    def render(self):
        target_mean, target_cov = self.convert_polar_to_cartesian_state(self.REs["PolarTargetPos"].state_mean, self.REs["PolarTargetPos"].state_cov)
        obs_mean, obs_cov = self.convert_polar_to_cartesian_state(self.REs["PolarObstacle1Pos"].state_mean, self.REs["PolarObstacle1Pos"].state_cov)
        estimator_means: Dict[str, torch.Tensor] = {"PolarTargetPos": target_mean, "PolarObstacle1Pos": obs_mean}
        estimator_covs: Dict[str, torch.Tensor] = {"PolarTargetPos": target_cov, "PolarObstacle1Pos": obs_cov}
        return self.env.render(1.0, {key: np.array(mean.cpu()) for key, mean in estimator_means.items()}, {key: np.array(cov.cpu()) for key, cov in estimator_covs.items()})

    def custom_reset(self):
        self.goals["PolarGoToTarget"].desired_distance = self.env.target.distance

    def get_observation_update_uncertainty(self, key):
        if "angle" in key or "_rot" in key:
            update_uncertainty = 1e-1
        else:
            update_uncertainty = 5e-1
        return update_uncertainty * torch.eye(1)

    def get_custom_sensor_noise(self, obs: dict):
        observation_noise = {}
        for key in obs.keys():
            if key == "target_offset_angle":
                observation_noise[key] = get_foveal_noise(obs["target_offset_angle"], 0, self.env_config["foveal_vision_noise"], self.env_config["robot_sensor_angle"])
            elif key == "target_offset_angle_dot":
                observation_noise[key] = get_foveal_noise(obs["target_offset_angle"], 1, self.env_config["foveal_vision_noise"], self.env_config["robot_sensor_angle"])
            elif key == "obstacle1_offset_angle":
                observation_noise[key] = get_foveal_noise(obs["obstacle1_offset_angle"], 0, self.env_config["foveal_vision_noise"], self.env_config["robot_sensor_angle"])
            elif key == "obstacle1_offset_angle_dot":
                observation_noise[key] = get_foveal_noise(obs["obstacle1_offset_angle"], 1, self.env_config["foveal_vision_noise"], self.env_config["robot_sensor_angle"])
            else:
                observation_noise[key] = 0.0
        return {key: val*torch.eye(1) for key, val in observation_noise.items()}
        