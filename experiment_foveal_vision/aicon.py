import numpy as np
import torch
from typing import Dict

from components.aicon import DroneEnvAICON as AICON
from components.estimator import RecursiveEstimator
from components.instances.estimators import Robot_Vel_Estimator_Vel #, Polar_Pos_Estimator_Vel
from components.instances.measurement_models import Robot_Vel_MM, Angle_MM
from components.instances.active_interconnections import Triangulation_AI
from components.instances.goals import PolarGoToTargetGoal

from experiment_foveal_vision.estimators import Polar_Pos_Estimator_Vel

# =============================================================================================================================================================

class FovealVisionAICON(AICON):
    def __init__(self, env_config):
        self.type = "FovealVision"
        super().__init__(env_config)

    def define_estimators(self):
        REs: Dict[str, RecursiveEstimator] = {
            "RobotVel":         Robot_Vel_Estimator_Vel(),
            "PolarTargetPos":   Polar_Pos_Estimator_Vel(),
        }
        return REs

    def define_measurement_models(self):
        MMs = {
            "RobotVel":         Robot_Vel_MM(),
            "PolarAngle":       Angle_MM(),
        }
        return MMs

    def define_active_interconnections(self):
        AIs = {
            "PolarDistance":    Triangulation_AI(),
        }
        return AIs

    def define_goals(self):
        goals = {
            "PolarGoToTarget" : PolarGoToTargetGoal(),
        }
        return goals

    def eval_update(self, action: torch.Tensor, new_step: bool, buffer_dict: Dict[str, Dict[str, torch.Tensor]]):
        u = self.get_control_input(action)

        # ----------------------------- predicts -------------------------------------

        self.REs["RobotVel"].call_predict(u, buffer_dict)
        self.REs["PolarTargetPos"].call_predict(u, buffer_dict)

        # ----------------------------- active interconnections -------------------------------------
        
        if new_step:
            self.meas_updates(buffer_dict)

        self.REs["PolarTargetPos"].call_update_with_active_interconnection(self.AIs["PolarDistance"], buffer_dict)

        # ----------------------------- measurements -------------------------------------

        return buffer_dict

    def compute_action(self, gradients):
        action = 0.9 * self.last_action - 1e-1 * gradients["PolarGoToTarget"]
        return action

    def print_estimators(self, print_cov=False):
        """
        print filter and environment states for debugging
        """
        obs = self.env.get_state()
        self.print_estimator("RobotVel", print_cov=print_cov)
        actual_vel = list(self.env.robot.vel)
        actual_vel.append(self.env.robot.vel_rot)
        print(f"True Robot Vel: {[f'{x:.3f}' for x in actual_vel]}")
        # print("--------------------------------------------------------------------")
        self.print_estimator("PolarTargetPos", print_cov=print_cov)
        actual_pos = self.env.rotation_matrix(-self.env.robot.orientation) @ (self.env.target.pos - self.env.robot.pos)
        angle = np.arctan2(actual_pos[1], actual_pos[0])
        dist = np.linalg.norm(actual_pos)
        print(f"True PolarTargetPos: {[f'{x:.3f}' for x in [dist, angle, obs['target_distance_dot'], obs['target_offset_angle_dot'] if obs['target_offset_angle_dot'] else 0.0]]}")
        print("--------------------------------------------------------------------")
    
    def custom_reset(self):
        self.goals["PolarGoToTarget"].desired_distance = self.env.target.distance