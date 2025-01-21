from typing import Dict
import numpy as np
import torch

from components.aicon import DroneEnvAICON as AICON
from components.instances.estimators import Robot_Vel_Estimator_Vel, Polar_Pos_Estimator_Vel, Rad_Estimator
from components.instances.measurement_models import Robot_Vel_MM
from components.instances.active_interconnections import Triangulation_AI
from components.instances.goals import PolarGoToTargetGoal

from models.old.experiment_divergence.active_interconnections import VisAngle_Rad_AI
from models.old.experiment_divergence.estimators import Vis_Angle_Estimator
from models.old.experiment_divergence.measurement_models import Vis_Angle_MM, Angle_MM

# ========================================================================================================

class DivergenceAICON(AICON):
    def __init__(self, env_config):
        self.type = "Divergence"
        super().__init__(env_config)

    def define_estimators(self):
        estimators = {
            "RobotVel":         Robot_Vel_Estimator_Vel(),
            "PolarTargetPos":   Polar_Pos_Estimator_Vel(),
            "TargetRad":        Rad_Estimator(),
            "TargetVisAngle":   Vis_Angle_Estimator(),
        }
        return estimators

    def define_measurement_models(self):
        return {
            "VelMM":            Robot_Vel_MM(),
            "AngleMeasMM":      Angle_MM(),
            "VisAngleMM":       Vis_Angle_MM(),
        }

    def define_active_interconnections(self):
        active_interconnections = {
            "TriangulationAI":  Triangulation_AI(),
            "VisAngleRadAI":    VisAngle_Rad_AI(),
        }
        return active_interconnections

    def define_goals(self):
        goals = {
            "PolarGoToTarget":  PolarGoToTargetGoal(),
        }
        return goals

    def eval_interconnections(self, buffer_dict: Dict[str, Dict[str, torch.Tensor]]):
        self.REs["PolarTargetPos"].call_update_with_active_interconnection(self.AIs["TriangulationAI"], buffer_dict)
        self.REs["TargetRad"].call_update_with_active_interconnection(self.AIs["VisAngleRadAI"], buffer_dict)
        self.REs["PolarTargetPos"].call_update_with_active_interconnection(self.AIs["VisAngleRadAI"], buffer_dict)
        return buffer_dict

    def compute_action_from_gradient(self, gradients):
        decay = 0.9
        return decay * self.last_action - 1e-2 * gradients["PolarGoToTarget"]
    
    def print_estimators(self, buffer_dict=None):
        env_state = self.env.get_state()
        print("--------------------------------------------------------------------")
        self.print_estimator("PolarTargetPos", buffer_dict=buffer_dict, print_cov=2)
        print(f"True PolarTargetPos: [{env_state['target_distance']:.3f}, {env_state['target_offset_angle']:.3f}, {env_state['target_distance_dot']:.3f}, {env_state['target_offset_angle_dot']:.3f}]")
        print("--------------------------------------------------------------------")
        self.print_estimator("RobotVel", buffer_dict=buffer_dict) 
        print(f"True RobotVel: [{self.env.robot.vel[0]}, {self.env.robot.vel[1]}, {self.env.robot.vel_rot}]")
        print("--------------------------------------------------------------------")
        self.print_estimator("TargetRad", buffer_dict=buffer_dict, print_cov=2)
        print(f"True TargetRad: 1.0")
        print("--------------------------------------------------------------------")
        self.print_estimator("TargetVisAngle", buffer_dict=buffer_dict)

    def custom_reset(self):
        self.goals["PolarGoToTarget"].desired_distance = self.env.target.distance
