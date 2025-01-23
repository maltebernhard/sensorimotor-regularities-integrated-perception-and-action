import numpy as np
import torch
from typing import Dict

from components.aicon import DroneEnvAICON as AICON
from components.estimator import RecursiveEstimator
from models.old.old_component_instances.estimators import Robot_Vel_Estimator_Vel, Robot_Vel_Estimator_Acc, Polar_Pos_Estimator_Vel, Polar_Pos_Estimator_Acc, Cartesian_Pos_Estimator, Rad_Estimator
from models.old.old_component_instances.measurement_models import Robot_Vel_MM, Pos_Angle_MM, Angle_MM

from models.old.experiment_general.estimators import Visibility_Estimator
from models.old.experiment_general.active_interconnections import Radius_Pos_VisAngle_AI, Visibility_Angle_AI, Triangulation_Visibility_AI
from models.old.experiment_general.goals import AvoidObstacleGoal, PolarGoToTargetGazeFixationGoal, PolarGoToTargetGoal
from models.old.experiment_general.measurement_models import Visibility_MM

# =============================================================================================================================================================

class GeneralTestAICON(AICON):
    def __init__(self, env_config):
        self.type = "GeneralTest"
        super().__init__(env_config)

    def define_estimators(self):
        REs: Dict[str, RecursiveEstimator] = {
            "RobotVel":         Robot_Vel_Estimator_Vel() if self.vel_control else Robot_Vel_Estimator_Acc(),
            #"CartesianTargetPos": Cartesian_Pos_Estimator(self.device, "CartesianTargetPos"),
            "PolarTargetPos":   Polar_Pos_Estimator_Vel() if self.vel_control else Polar_Pos_Estimator_Acc(),
            "TargetVisibility": Visibility_Estimator(),
        }
        for i in range(1, self.num_obstacles + 1):
            REs[f"PolarObstacle{i}Pos"] =       Polar_Pos_Estimator_Vel(f"Obstacle{i}") if self.vel_control else Polar_Pos_Estimator_Acc(f"Obstacle{i}")
            #REs[f"CartesianObstacle{i}Pos"] =  Cartesian_Pos_Estimator(f"Obstacle{i}")
            REs[f"Obstacle{i}Rad"] =            Rad_Estimator(f"Obstacle{i}")
        return REs

    def define_measurement_models(self):
        MMs = {
            "RobotVel":                     (Robot_Vel_MM(), ["RobotVel"]),
            "PolarTargetAngle":             (Angle_MM(), ["PolarTargetPos"]),
            #"CartesianTargetPos-Angle":    (Pos_Angle_MM(), ["CartesianTargetPos"]),
            "TargetVisiblity":              (Visibility_MM(), ["TargetVisibility"]),
        }
        for i in range(1, self.num_obstacles + 1):
            MMs[f"PolarObstacle{i}Angle"] =             (Angle_MM(f"Obstacle{i}"), [f"PolarObstacle{i}Pos"])
            #MMs[f"CartesianObstacle{i}Pos-Angle"] =    (Pos_Angle_MM(f"Obstacle{i}"), [f"CartesianObstacle{i}Pos"])
        return MMs

    def define_active_interconnections(self):
        AIs = {
            "Triangulation":    Triangulation_Visibility_AI(),
            "TargetVisibility": Visibility_Angle_AI(sensor_angle_rad=self.env_config["robot_sensor_angle"]),
        }
        for i in range(1, self.num_obstacles + 1):
            AIs[f"Obstacle{i}Rad"] = Radius_Pos_VisAngle_AI(f"Obstacle{i}")
        return AIs

    def define_goals(self):
        goals = {
            "GazeFixation" :    PolarGoToTargetGazeFixationGoal(),
            "PolarGoToTarget" : PolarGoToTargetGoal(),
        }
        for i in range(1, self.num_obstacles + 1):
            goals[f"AvoidObstacle{i}"] = AvoidObstacleGoal(i)
        return goals

    def render(self):
        target_mean, target_cov = self.convert_polar_to_cartesian_state(self.REs["PolarTargetPos"].state_mean, self.REs["PolarTargetPos"].state_cov)
        estimator_means: Dict[str, torch.Tensor] = {"PolarTargetPos": target_mean}
        estimator_covs: Dict[str, torch.Tensor] = {"PolarTargetPos": target_cov}
        for i in range(1, self.num_obstacles + 1):
            estimator_means[f"CartesianObstacle{i}Pos"] = self.REs[f"CartesianObstacle{i}Pos"].state_mean
            rad = self.REs[f"Obstacle{i}Rad"].state_mean
            estimator_covs[f"CartesianObstacle{i}Pos"] = torch.tensor([[rad**2, 0], [0, rad**2]])
        return self.env.render(1.0, {key: np.array(mean.cpu()) for key, mean in estimator_means.items()}, {key: np.array(cov.cpu()) for key, cov in estimator_covs.items()})

    def eval_interconnections(self, buffer_dict: Dict[str, Dict[str, torch.Tensor]]):
        self.REs["TargetVisibility"].call_update_with_active_interconnection(self.AIs["TargetVisibility"], buffer_dict)
        self.REs["PolarTargetPos"].call_update_with_active_interconnection(self.AIs["Triangulation"], buffer_dict)
        # for i in range(1, self.num_obstacles + 1):
        #     self.REs[f"Obstacle{i}Rad"].call_update_with_active_interconnection(self.AIs[f"Obstacle{i}Rad"], buffer_dict)
        return buffer_dict

    def compute_action_from_gradient(self, gradients):
        goal = "PolarGoToTarget"
        #goal = "GazeFixation"
        if self.vel_control:
            action = 0.9 * self.last_action - 5e-3 * gradients[goal]
            for i in range(self.num_obstacles):
                action -= 1e-1 * gradients[f"AvoidObstacle{i+1}"] / self.num_obstacles
        else:
            action = 0.7 * self.last_action - 5e-2 * gradients[goal]
            for i in range(self.num_obstacles):
                action -= 1e0 * gradients[f"AvoidObstacle{i+1}"] / self.num_obstacles
        # manual gaze fixation
        # if self.vel_control:
        #     action[2] = 0.05 * self.REs["PolarTargetPos"].state_mean[1] + 0.01 * self.REs["PolarTargetPos"].state_mean[3]
        return action

    def print_estimators(self, buffer_dict=None):
        """
        print filter and environment states for debugging
        """
        env_state = self.env.get_state()
        self.print_estimator("TargetVisibility", print_cov=2)
        print(f"True TargetVisibility: {'target_offset_angle' in self.env.get_observation()[0].keys()}")
        print("--------------------------------------------------------------------")
        self.print_estimator("PolarTargetPos", buffer_dict=buffer_dict, print_cov=2)
        print(f"True PolarTargetPos: [{env_state['target_distance']:.3f}, {env_state['target_offset_angle']:.3f}, {env_state['target_distance_dot']:.3f}, {env_state['target_offset_angle_dot']:.3f}]")
        print("--------------------------------------------------------------------")
        self.print_estimator("RobotVel", buffer_dict=buffer_dict) 
        print(f"True RobotVel: [{self.env.robot.vel[0]:.3f}, {self.env.robot.vel[1]:.3f}, {self.env.robot.vel_rot:.3f}]")
        print("--------------------------------------------------------------------")
        if self.num_obstacles > 0:
            [self.print_estimator(f"CartesianObstacle{i}Pos", print_cov=3) for i in range(1, self.num_obstacles + 1)]
            actual_radii = [self.env.obstacles[i-1].radius for i in range(1, self.num_obstacles + 1)]
            print(f"True Obstacle Radius: {[f'{x:.3f}' for x in actual_radii]}")
            print("--------------------------------------------------------------------")
    
    def custom_reset(self):
        self.goals["PolarGoToTarget"].desired_distance = self.env.target.distance
        self.goals["GazeFixation"].desired_distance = self.env.target.distance