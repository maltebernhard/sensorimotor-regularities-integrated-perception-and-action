import numpy as np
import torch
from typing import Dict

from components.aicon import DroneEnvAICON as AICON
from components.estimator import RecursiveEstimator
from models.old.old_component_instances.estimators import Rad_Estimator, Robot_Vel_Estimator_Vel, Robot_Vel_Estimator_Acc, Polar_Pos_Estimator_Vel, Polar_Pos_Estimator_Acc
from models.old.old_component_instances.measurement_models import Robot_Vel_MM, Pos_Angle_MM#, Angle_MM
#from components.instances.active_interconnections import Triangulation_AI

from models.old.even_older.experiment_visibility.estimators import Target_Visibility_Estimator, Polar_Distance_Estimator, Polar_Angle_Estimator
from models.old.even_older.experiment_visibility.active_interconnections import Radius_Pos_VisAngle_AI, Triangulation_Detached_AI, Visibility_Angle_AI, Triangulation_Visibility_AI, Visibility_Detached_AI
from models.old.even_older.experiment_visibility.goals import AvoidObstacleGoal, PolarGoToTargetGazeFixationGoal, PolarGoToTargetGoal
from models.old.even_older.experiment_visibility.measurement_models import Visibility_MM, Angle_MM

# =============================================================================================================================================================

class VisibilityAICON(AICON):
    def __init__(self, env_config):
        self.type = "Visibility"
        super().__init__(env_config)

    def define_estimators(self):
        REs: Dict[str, RecursiveEstimator] = {
            "RobotVel":             Robot_Vel_Estimator_Vel() if self.vel_control else Robot_Vel_Estimator_Acc(),
            #"PolarTargetPos":      Polar_Pos_Estimator_Vel() if self.vel_control else Polar_Pos_Estimator_Acc(),
            "PolarTargetDistance":  Polar_Distance_Estimator(),
            "PolarTargetAngle":     Polar_Angle_Estimator(),
            "TargetVisibility":     Target_Visibility_Estimator(),
        }
        if not self.vel_control:
            raise NotImplementedError("Only velocity control is implemented for Polar Distance and Angle Estimators")
        return REs

    def define_measurement_models(self):
        MMs = {
            "RobotVel":         (Robot_Vel_MM(),  ["RobotVel"]),
            "PolarTargetAngle": (Angle_MM(),      ["PolarTargetAngle"]),
            "TargetVisiblity":  (Visibility_MM(), ["TargetVisibility"]),
        }
        return MMs

    def define_active_interconnections(self):
        AIs = {
            #"Triangulation":   Triangulation_Visibility_AI(),
            "Triangulation":    Triangulation_Detached_AI(),
            #"TargetVisibility":Visibility_Angle_AI(sensor_angle_rad=self.env_config["robot_sensor_angle"]),
            "TargetVisibility": Visibility_Detached_AI(sensor_angle_rad=self.env_config["robot_sensor_angle"]),
        }
        return AIs

    def define_goals(self):
        goals = {
            "GazeFixation" :    PolarGoToTargetGazeFixationGoal(),
            "PolarGoToTarget" : PolarGoToTargetGoal(),
        }
        return goals

    def render(self):
        polar_state = torch.zeros(2, device=self.device)
        polar_cov = torch.eye(2, device=self.device)
        polar_state[0] = self.REs["PolarTargetDistance"].state_mean[0]
        polar_state[1] = self.REs["PolarTargetAngle"].state_mean[0]
        polar_cov[0, 0] = self.REs["PolarTargetDistance"].state_cov[0, 0]
        polar_cov[1, 1] = self.REs["PolarTargetAngle"].state_cov[0, 0]
        target_mean, target_cov = self.convert_polar_to_cartesian_state(polar_state, polar_cov)
        #target_mean, target_cov = self.convert_polar_to_cartesian_state(self.REs["PolarTargetPos"].state_mean, self.REs["PolarTargetPos"].state_cov)
        estimator_means: Dict[str, torch.Tensor] = {"PolarTargetPos": target_mean}
        estimator_covs: Dict[str, torch.Tensor] = {"PolarTargetPos": target_cov}
        return self.env.render(1.0, {key: np.array(mean.cpu()) for key, mean in estimator_means.items()}, {key: np.array(cov.cpu()) for key, cov in estimator_covs.items()})

    def eval_interconnections(self, buffer_dict: Dict[str, Dict[str, torch.Tensor]]):
        #self.REs["TargetVisibility"].call_update_with_active_interconnection(self.AIs["TargetVisibility"], buffer_dict)
        self.REs["PolarTargetAngle"].call_update_with_active_interconnection(self.AIs["TargetVisibility"], buffer_dict)
        self.REs["PolarTargetDistance"].call_update_with_active_interconnection(self.AIs["Triangulation"], buffer_dict)
        return buffer_dict

    def compute_action_from_gradient(self, gradients):
        goal = "PolarGoToTarget"
        if self.vel_control:
            action = 0.9 * self.last_action - 5e-3 * gradients[goal]
        else:
            action = 0.7 * self.last_action - 5e-2 * gradients[goal]
        return action

    def print_estimators(self, buffer_dict=None):
        """
        print filter and environment states for debugging
        """
        self.print_estimator("TargetVisibility", print_cov=2)
        env_state = self.env.get_state()
        print("--------------------------------------------------------------------")
        self.print_estimator("PolarTargetDistance", buffer_dict=buffer_dict, print_cov=2)
        self.print_estimator("PolarTargetAngle", buffer_dict=buffer_dict, print_cov=2)
        print(f"True PolarTargetPos: [{env_state['target_distance']:.3f}, {env_state['target_offset_angle']:.3f}, {env_state['target_distance_dot']:.3f}, {env_state['target_offset_angle_dot']:.3f}]")
        print("--------------------------------------------------------------------")
        self.print_estimator("RobotVel", buffer_dict=buffer_dict) 
        print(f"True RobotVel: [{self.env.robot.vel[0]:.3f}, {self.env.robot.vel[1]:.3f}, {self.env.robot.vel_rot:.3f}]")
        print("--------------------------------------------------------------------")
    
    def custom_reset(self):
        self.goals["PolarGoToTarget"].desired_distance = self.env.target.distance
        self.goals["GazeFixation"].desired_distance = self.env.target.distance