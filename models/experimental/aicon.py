from typing import Dict
import numpy as np
import torch

from components.aicon import DroneEnvAICON as AICON
from components.helpers import rotate_vector_2d

from models.experimental.active_interconnections import Triangulation_AI
from models.experimental.estimators import Robot_Vel_Estimator, Polar_Pos_Estimator
from models.experimental.goals import PolarGoToTargetGoal
from models.experimental.measurement_models import Angle_MM, Robot_Vel_MM

# ========================================================================================================

class ExperimantalAICON(AICON):
    def __init__(self, env_config):
        self.type = "Experimental"
        super().__init__(env_config)

    def define_estimators(self):
        estimators = {
            "RobotVel":         Robot_Vel_Estimator(),
            "PolarTargetPos":   Polar_Pos_Estimator(),
        }
        return estimators

    def define_measurement_models(self):
        return {
            "VelMM":            (Robot_Vel_MM(), ["RobotVel"]),
            "AngleMeasMM":      (Angle_MM(),     ["PolarTargetPos"]),
        }

    def define_active_interconnections(self):
        active_interconnections = {
            "TriangulationAI":  Triangulation_AI(),
        }
        return active_interconnections

    def define_goals(self):
        goals = {
            "PolarGoToTarget":  PolarGoToTargetGoal(),
        }
        return goals

    def eval_interconnections(self, buffer_dict: Dict[str, Dict[str, torch.Tensor]]):
        self.REs["PolarTargetPos"].call_update_with_active_interconnection(self.AIs["TriangulationAI"], buffer_dict)
        return buffer_dict 

    def compute_action_from_gradient(self, gradients):
        decay = 0.9
        return decay * self.last_action - 0.6 * self.env_config["timestep"] * gradients["PolarGoToTarget"]
    
    def print_estimators(self, buffer_dict=None):
        env_state = self.env.get_state()
        print("--------------------------------------------------------------------")
        self.print_estimator("PolarTargetPos", buffer_dict=buffer_dict, print_cov=2)
        print(f"True PolarTargetPos: [{env_state['target_distance']:.3f}, {env_state['target_offset_angle']:.3f}, {env_state['target_distance_dot']:.3f}, {env_state['target_offset_angle_dot']:.3f}]")
        rtf_vel = rotate_vector_2d(env_state["target_offset_angle"], torch.tensor([env_state["vel_frontal"], env_state["vel_lateral"]]))
        print(f"True Global TargetVel: [{env_state['target_distance_dot']+rtf_vel[0]:.3f}, {env_state['target_offset_angle_dot']+env_state['vel_rot']:.3f}]")
        print("--------------------------------------------------------------------")
        self.print_estimator("RobotVel", buffer_dict=buffer_dict) 
        print(f"True RobotVel: [{self.env.robot.vel[0]:.3f}, {self.env.robot.vel[1]:.3f}, {self.env.robot.vel_rot:.3f}]")
        print("--------------------------------------------------------------------")

    def custom_reset(self):
        self.goals["PolarGoToTarget"].desired_distance = self.env.target.distance

    def get_observation_update_uncertainty(self, key):
        if "angle" in key or "_rot" in key:
            update_uncertainty: torch.Tensor = 2e-1 * torch.eye(1)
        else:
            update_uncertainty: torch.Tensor = 5e-1 * torch.eye(1)
        return update_uncertainty

    def render(self):
        target_mean, target_cov = self.convert_polar_to_cartesian_state(self.REs["PolarTargetPos"].state_mean, self.REs["PolarTargetPos"].state_cov)
        estimator_means: Dict[str, torch.Tensor] = {"PolarTargetPos": target_mean}
        estimator_covs: Dict[str, torch.Tensor] = {"PolarTargetPos": target_cov}
        return self.env.render(1.0, {key: np.array(mean.cpu()) for key, mean in estimator_means.items()}, {key: np.array(cov.cpu()) for key, cov in estimator_covs.items()})

    
