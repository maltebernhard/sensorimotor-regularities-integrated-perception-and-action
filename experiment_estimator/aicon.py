from typing import Dict
import numpy as np
import torch

from components.aicon import DroneEnvAICON as AICON
from components.instances.measurement_models import Angle_MM, Vel_MM

from experiment_estimator.active_interconnections import DistanceUpdaterAcc, DistanceUpdaterVel
from experiment_estimator.estimators import Robot_State_Estimator_Acc, Robot_State_Estimator_Vel
from experiment_estimator.goals import SpecificGoToTargetGoal

# ========================================================================================================

class ContingentEstimatorAICON(AICON):
    def __init__(self, env_config):
        self.type = "ContingentEstimator"
        super().__init__(**env_config)

    def define_estimators(self):
        estimators = {}
        estimators["RobotState"] = Robot_State_Estimator_Vel(self.device) if self.vel_control else Robot_State_Estimator_Acc(self.device)
        #estimators["PolarTargetPos"] = Polar_Pos_Estimator_Acc(self.device, "PolarTargetPos") if not self.vel_control else Polar_Pos_Estimator_Vel(self.device, "PolarTargetPos")
        return estimators
    
    def define_measurement_models(self):
        return {
            "VelMM": Vel_MM(self.device),
            "AngleMeasMM": Angle_MM(self.device, "Target"),
        }

    def define_active_interconnections(self):
        active_interconnections = {
            #"TriangulationAI": Triangulation_AI(self.device),
            #"DistanceUpdater": DistanceUpdaterAcc(self.device) if not self.vel_control else DistanceUpdaterVel(self.device),
        }
        return active_interconnections

    def define_goals(self):
        goals = {
            "GoToTarget": SpecificGoToTargetGoal(self.device),
        }
        return goals

    def eval_update(self, action: torch.Tensor, new_step: bool, buffer_dict: Dict[str, Dict[str, torch.Tensor]]):
        u = self.get_control_input(action)
        
        self.REs["RobotState"].call_predict(u, buffer_dict)
        if new_step:
            self.REs["RobotState"].call_update_with_meas_model(self.MMs["VelMM"], buffer_dict, self.get_meas_dict(self.MMs["VelMM"]))
            self.REs["RobotState"].call_update_with_meas_model(self.MMs["AngleMeasMM"], buffer_dict, self.get_meas_dict(self.MMs["AngleMeasMM"]))
        else:
            pass
        return buffer_dict

    def compute_action(self, gradients):
        if not self.vel_control:
            return self.last_action - 1e0 * gradients["GoToTarget"]
        else:
            return self.last_action - 1e-2 * gradients["GoToTarget"]

    def render(self):
        target_mean, target_cov = self.convert_polar_to_cartesian_state(self.REs["RobotState"].state_mean, self.REs["RobotState"].state_cov)
        estimator_means: Dict[str, torch.Tensor] = {"PolarTargetPos": target_mean}
        estimator_covs: Dict[str, torch.Tensor] = {"PolarTargetPos": target_cov}
        return self.env.render(1.0, {key: np.array(mean.cpu()) for key, mean in estimator_means.items()}, {key: np.array(cov.cpu()) for key, cov in estimator_covs.items()})

    def print_states(self, buffer_dict=None):
        obs = self.env.get_reality()
        print("--------------------------------------------------------------------")
        self.print_state("RobotState", buffer_dict=buffer_dict)
        actual_pos = self.env.rotation_matrix(-self.env.robot.orientation) @ (self.env.target.pos - self.env.robot.pos)
        angle = np.arctan2(actual_pos[1], actual_pos[0])
        dist = np.linalg.norm(actual_pos)
        # TODO: observations can be None now
        #print(f"True Polar Target Position: {[f'{x:.3f}' for x in [dist, angle, obs['target_distance_dot'], obs['target_offset_angle_dot']]]}")
        print(f"True Polar Target Position: {[f'{x:.3f}' for x in [dist, angle]]}")
        actual_state = [self.env.robot.vel[0], self.env.robot.vel[1], self.env.robot.vel_rot]
        actual_pos = self.env.rotation_matrix(-self.env.robot.orientation) @ (self.env.target.pos - self.env.robot.pos)
        angle = np.arctan2(actual_pos[1], actual_pos[0])
        dist = np.linalg.norm(actual_pos)
        # TODO: observations can be None now
        #actual_state += [dist, angle, obs['target_distance_dot'], obs['target_offset_angle_dot']]
        actual_state += [dist, angle]
        print(f"True State: {actual_state}")
        print("--------------------------------------------------------------------")

    def custom_reset(self):
        self.goals["GoToTarget"].desired_distance = self.env.target.distance