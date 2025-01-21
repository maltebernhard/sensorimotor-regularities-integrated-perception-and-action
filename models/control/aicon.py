from typing import Dict
import torch

from components.aicon import DroneEnvAICON as AICON
from components.instances.estimators import Robot_Vel_Estimator_Vel, Polar_Pos_Estimator_Vel
from components.instances.active_interconnections import Triangulation_AI
from components.instances.measurement_models import Robot_Vel_MM, Angle_MM
from components.instances.goals import PolarGoToTargetGoal

# ========================================================================================================

class ControlAICON(AICON):
    def __init__(self, env_config):
        self.type = "Control"
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

    def compute_action_from_gradient(self, gradients):
        decay = 0.9
        gradient_action = decay * self.last_action - 3e-2 * gradients["PolarGoToTarget"]
        # control
        gradient_action[2] = 2.0 * self.REs["PolarTargetPos"].state_mean[1] + 0.01 * self.REs["PolarTargetPos"].state_mean[3]
        return gradient_action
    
    def print_estimators(self, buffer_dict=None):
        env_state = self.env.get_state()
        print("--------------------------------------------------------------------")
        self.print_estimator("PolarTargetPos", buffer_dict=buffer_dict, print_cov=2)
        print(f"True PolarTargetPos: [{env_state['target_distance']:.3f}, {env_state['target_offset_angle']:.3f}, {env_state['target_distance_dot']:.3f}, {env_state['target_offset_angle_dot']:.3f}, {env_state['target_radius']:.3f}]")
        print("--------------------------------------------------------------------")
        self.print_estimator("RobotVel", buffer_dict=buffer_dict) 
        print(f"True RobotVel: [{self.env.robot.vel[0]}, {self.env.robot.vel[1]}, {self.env.robot.vel_rot}]")
        print("--------------------------------------------------------------------")

    def custom_reset(self):
        self.goals["PolarGoToTarget"].desired_distance = self.env.target.distance
