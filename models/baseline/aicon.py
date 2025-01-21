from typing import Dict
import torch

from components.aicon import DroneEnvAICON as AICON
from components.helpers import rotate_vector_2d
from components.instances.estimators import Robot_Vel_Estimator_Vel, Polar_Pos_Estimator_Vel
from components.instances.active_interconnections import Triangulation_AI
from components.instances.measurement_models import Robot_Vel_MM, Angle_MM
from components.instances.goals import PolarGoToTargetGoal

# ========================================================================================================

class BaselineAICON(AICON):
    def __init__(self, env_config):
        self.type = "Baseline"
        super().__init__(env_config)

    def define_estimators(self):
        estimators = {
            "RobotVel":         Robot_Vel_Estimator_Vel(),
            "PolarTargetPos":   Polar_Pos_Estimator_Vel(),
        }
        return estimators

    def define_measurement_models(self):
        return {
            "VelMM":        Robot_Vel_MM(),
            "AngleMeasMM":  Angle_MM(),
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

    def compute_action(self):
        action = torch.zeros(3)
        # go to target control
        vel_radial = 1.0 / 3.0 * (self.REs["PolarTargetPos"].state_mean[0] - self.goals["PolarGoToTarget"].desired_distance)
        # tangential motion control
        vel_tangential = 1.0 / 10.0 * self.REs["PolarTargetPos"].state_cov[0][0]
        action[:2] = rotate_vector_2d(-self.REs["PolarTargetPos"].state_mean[1], torch.tensor([vel_radial, vel_tangential])).squeeze()
        # rotation control
        action[2] = 2.0 * self.REs["PolarTargetPos"].state_mean[1] #+ 0.01 * self.REs["PolarTargetPos"].state_mean[3]
        return action
    
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
