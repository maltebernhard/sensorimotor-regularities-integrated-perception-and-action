from typing import Dict
import numpy as np
import torch

from components.aicon import DroneEnvAICON as AICON
from components.helpers import rotate_vector_2d
from components.instances.estimators import Robot_Vel_Estimator_Vel
from components.instances.measurement_models import Robot_Vel_MM, Angle_MM
from components.instances.goals import PolarGoToTargetGoal

from experiment_base.active_interconnections import Triangulation_AI
from experiment_base.estimators import Polar_Pos_Estimator_Vel

# ========================================================================================================

class BaseAICON(AICON):
    def __init__(self, env_config):
        self.type = "Base"
        super().__init__(**env_config)

    def define_estimators(self):
        estimators = {
            "RobotVel":         Robot_Vel_Estimator_Vel(),
            "PolarTargetPos":   Polar_Pos_Estimator_Vel(),
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
            return decay * self.last_action - 1e-2 * gradients["PolarGoToTarget"]
    
    def print_states(self, buffer_dict=None):
        env_state = self.env.get_state()
        print("--------------------------------------------------------------------")
        self.print_state("PolarTargetPos", buffer_dict=buffer_dict, print_cov=2)
        print(f"True PolarTargetPos: [{env_state['target_distance']:.3f}, {env_state['target_offset_angle']:.3f}, {env_state['target_distance_dot']:.3f}, {env_state['target_offset_angle_dot']:.3f}]")
        rtf_vel = rotate_vector_2d(env_state["target_offset_angle"], torch.tensor([env_state["vel_frontal"], env_state["vel_lateral"]]))
        print(f"True Global TargetVel: [{env_state['target_distance_dot']+rtf_vel[0]:.3f}, {env_state['target_offset_angle_dot']+env_state['vel_rot']:.3f}]")
        print("--------------------------------------------------------------------")
        self.print_state("RobotVel", buffer_dict=buffer_dict) 
        print(f"True RobotVel: [{self.env.robot.vel[0]:.3f}, {self.env.robot.vel[1]:.3f}, {self.env.robot.vel_rot:.3f}]")
        print("--------------------------------------------------------------------")

    def custom_reset(self):
        self.goals["PolarGoToTarget"].desired_distance = self.env.target.distance

    
