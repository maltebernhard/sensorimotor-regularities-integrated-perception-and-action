from typing import Dict
import numpy as np
import torch

from components.aicon import DroneEnvAICON as AICON
from models.old.old_component_instances.measurement_models import Angle_MM, Robot_Vel_MM

from models.old.even_older.experiment_estimator.active_interconnections import DistanceUpdaterAcc, DistanceUpdaterVel
from models.old.even_older.experiment_estimator.estimators import Robot_State_Estimator_Acc, Robot_State_Estimator_Vel
from models.old.even_older.experiment_estimator.goals import SpecificGoToTargetGoal

# ========================================================================================================

class ContingentEstimatorAICON(AICON):
    def __init__(self, env_config):
        self.type = "ContingentEstimator"
        super().__init__(env_config)

    def define_estimators(self):
        estimators = {
        "RobotState":       Robot_State_Estimator_Vel() if self.vel_control else Robot_State_Estimator_Acc(),
        #"PolarTargetPos":   Polar_Pos_Estimator_Acc() if not self.vel_control else Polar_Pos_Estimator_Vel()
        }
        return estimators
    
    def define_measurement_models(self):
        return {
            "VelMM":        (Robot_Vel_MM(), ["RobotState"]),
            "AngleMeasMM":  (Angle_MM(), ["RobotState"]),
        }

    def define_active_interconnections(self):
        active_interconnections = {
            #"TriangulationAI": Triangulation_AI(),
            #"DistanceUpdater": DistanceUpdaterAcc() if not self.vel_control else DistanceUpdaterVel(),
        }
        return active_interconnections

    def define_goals(self):
        goals = {
            "GoToTarget":   SpecificGoToTargetGoal(),
        }
        return goals

    def eval_interconnections(self, buffer_dict: Dict[str, Dict[str, torch.Tensor]]):
        # TODO: No AIs?
        return buffer_dict

    def compute_action_from_gradient(self, gradients):
        if not self.vel_control:
            return self.last_action - 1e0 * gradients["GoToTarget"]
        else:
            return self.last_action - 1e-2 * gradients["GoToTarget"]

    def render(self):
        target_mean, target_cov = self.convert_polar_to_cartesian_state(self.REs["RobotState"].state_mean, self.REs["RobotState"].state_cov)
        estimator_means: Dict[str, torch.Tensor] = {"PolarTargetPos": target_mean}
        estimator_covs: Dict[str, torch.Tensor] = {"PolarTargetPos": target_cov}
        return self.env.render(1.0, {key: np.array(mean.cpu()) for key, mean in estimator_means.items()}, {key: np.array(cov.cpu()) for key, cov in estimator_covs.items()})

    def print_estimators(self, buffer_dict=None):
        env_state = self.env.get_state()
        print("--------------------------------------------------------------------")
        self.print_estimator("RobotState", buffer_dict=buffer_dict)
        print(f"True State: [{self.env.robot.vel[0]:.3f}, {self.env.robot.vel[1]:.3f}, {self.env.robot.vel_rot:.3f}, {env_state['target_distance']:.3f}, {env_state['target_offset_angle']:.3f}, {env_state['target_distance_dot']:.3f}, {env_state['target_offset_angle_dot']:.3f}, {env_state['target_radius']:.3f}")
        print("--------------------------------------------------------------------")

    def custom_reset(self):
        self.goals["GoToTarget"].desired_distance = self.env.target.distance