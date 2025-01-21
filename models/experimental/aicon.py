from typing import Dict
import numpy as np
import torch
from torch.func import jacrev

from components.aicon import DroneEnvAICON as AICON
from components.helpers import rotate_vector_2d
from components.instances.measurement_models import Robot_Vel_MM

from models.experimental.active_interconnections import Triangulation_AI
from models.experimental.estimators import Robot_Vel_Estimator_Vel, Polar_Pos_Estimator_Vel
from models.experimental.goals import PolarGoToTargetGoal
from models.experimental.measurement_models import Angle_MM

# ========================================================================================================

class ExperimantalAICON(AICON):
    def __init__(self, env_config):
        self.type = "Base"
        super().__init__(env_config)

    def define_estimators(self):
        estimators = {
            "RobotVel":         Robot_Vel_Estimator_Vel(),
            "PolarTargetGlobalPos":   Polar_Pos_Estimator_Vel(),
        }
        return estimators

    def define_measurement_models(self):
        return {
            "VelMM":            Robot_Vel_MM(),
            "AngleMeasMM":      Angle_MM(),
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
        self.REs["PolarTargetGlobalPos"].call_update_with_active_interconnection(self.AIs["TriangulationAI"], buffer_dict)
        return buffer_dict

#   def compute_goal_action_gradient(self, goal):
#         action = torch.zeros(3)
#         jacobian, step_eval = jacrev(
#             self._eval_goal_with_aux,
#             argnums=0,
#             has_aux=True)(action, goal)
#         return jacobian  

    def compute_action_from_gradient(self, gradients):
            decay = 0.98
            return decay * self.last_action - 0.2 * self.env_config["timestep"] * gradients["PolarGoToTarget"]
            #return - 2.0 * self.env_config["timestep"] * gradients["PolarGoToTarget"]
    
    def print_estimators(self, buffer_dict=None):
        env_state = self.env.get_state()
        print("--------------------------------------------------------------------")
        self.print_estimator("PolarTargetGlobalPos", buffer_dict=buffer_dict, print_cov=2)
        print(f"True PolarTargetGlobalPos: [{env_state['target_distance']:.3f}, {env_state['target_offset_angle']:.3f}, {env_state['target_distance_dot']:.3f}, {env_state['target_offset_angle_dot']:.3f}]")
        rtf_vel = rotate_vector_2d(env_state["target_offset_angle"], torch.tensor([env_state["vel_frontal"], env_state["vel_lateral"]]))
        print(f"True Global TargetVel: [{env_state['target_distance_dot']+rtf_vel[0]:.3f}, {env_state['target_offset_angle_dot']+env_state['vel_rot']:.3f}]")
        print("--------------------------------------------------------------------")
        self.print_estimator("RobotVel", buffer_dict=buffer_dict) 
        print(f"True RobotVel: [{self.env.robot.vel[0]:.3f}, {self.env.robot.vel[1]:.3f}, {self.env.robot.vel_rot:.3f}]")
        print("--------------------------------------------------------------------")

    def custom_reset(self):
        self.goals["PolarGoToTarget"].desired_distance = self.env.target.distance

    def render(self):
        target_mean, target_cov = self.convert_polar_to_cartesian_state(self.REs["PolarTargetGlobalPos"].state_mean, self.REs["PolarTargetGlobalPos"].state_cov)
        estimator_means: Dict[str, torch.Tensor] = {"PolarTargetGlobalPos": target_mean}
        estimator_covs: Dict[str, torch.Tensor] = {"PolarTargetGlobalPos": target_cov}
        return self.env.render(1.0, {key: np.array(mean.cpu()) for key, mean in estimator_means.items()}, {key: np.array(cov.cpu()) for key, cov in estimator_covs.items()})

    
