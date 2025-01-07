from typing import Dict
import numpy as np
import torch

from components.aicon import DroneEnvAICON as AICON
from components.instances.estimators import Robot_Vel_Estimator_Vel, Polar_Pos_Estimator_Vel
from components.instances.active_interconnections import Triangulation_AI
from components.instances.measurement_models import Vel_MM, Angle_Meas_MM
from components.instances.goals import PolarGoToTargetGoal

# ========================================================================================================

class ControlAICON(AICON):
    def __init__(self, env_config):
        self.type = "Control"
        super().__init__(**env_config)

    def define_estimators(self):
        estimators = {
            "RobotVel": Robot_Vel_Estimator_Vel(self.device),
            "PolarTargetPos": Polar_Pos_Estimator_Vel(self.device, "PolarTargetPos"),
        }
        return estimators

    def define_measurement_models(self):
        return {
            "VelMM": Vel_MM(self.device),
            "AngleMeasMM": Angle_Meas_MM(self.device),
        }

    def define_active_interconnections(self):
        active_interconnections = {
            "TriangulationAI": Triangulation_AI([self.REs["PolarTargetPos"], self.REs["RobotVel"]], self.device),
        }
        return active_interconnections

    def define_goals(self):
        goals = {
            "PolarGoToTarget": PolarGoToTargetGoal(self.device),
        }
        return goals

    def eval_update(self, action: torch.Tensor, new_step: bool, buffer_dict: Dict[str, Dict[str, torch.Tensor]]):
        u = self.get_control_input(action)

        self.REs["RobotVel"].call_predict(u, buffer_dict)
        self.REs["PolarTargetPos"].call_predict(u, buffer_dict)

        if new_step:
            self.meas_updates(buffer_dict)

        self.REs["PolarTargetPos"].call_update_with_active_interconnection(self.AIs["TriangulationAI"], buffer_dict)
        
        return buffer_dict

    def compute_action(self, gradients):
        decay = 0.9
        gradient_action = decay * self.last_action - 1e-2 * gradients["PolarGoToTarget"]
        # control
        obs = self.env.get_observation()
        gradient_action[2] = 1.0 * obs["target_offset_angle"] + 0.01 * obs["target_offset_angle_dot"]
        return gradient_action
    
    def print_states(self, buffer_dict=None):
        obs = self.env.get_reality()
        print("--------------------------------------------------------------------")
        self.print_state("PolarTargetPos", buffer_dict=buffer_dict, print_cov=2)
        # TODO: observations can be None now
        print(f"True PolarTargetPos: [{obs['target_distance']:.3f}, {obs['target_offset_angle']:.3f}, {obs['target_distance_dot']:.3f}, {obs['target_offset_angle_dot']:.3f}, {obs['target_radius']:.3f}]")
        print("--------------------------------------------------------------------")
        self.print_state("RobotVel", buffer_dict=buffer_dict) 
        print(f"True RobotVel: [{self.env.robot.vel[0]}, {self.env.robot.vel[1]}, {self.env.robot.vel_rot}]")
        print("--------------------------------------------------------------------")

    def custom_reset(self):
        self.goals["PolarGoToTarget"].desired_distance = self.env.target.distance
