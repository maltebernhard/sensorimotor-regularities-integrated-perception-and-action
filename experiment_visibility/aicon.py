import numpy as np
import torch
from typing import Dict

from components.aicon import DroneEnvAICON as AICON
from components.estimator import RecursiveEstimator
from components.instances.estimators import Robot_Vel_Estimator_Vel, Robot_Vel_Estimator_Acc, Polar_Pos_Estimator_Vel, Polar_Pos_Estimator_Acc, Cartesian_Pos_Estimator
from components.instances.measurement_models import Vel_MM, Pos_Angle_MM#, Angle_MM
#from components.instances.active_interconnections import Triangulation_AI

from experiment_visibility.estimators import Obstacle_Rad_Estimator, Target_Visibility_Estimator, Polar_Distance_Estimator, Polar_Angle_Estimator
from experiment_visibility.active_interconnections import Radius_Pos_VisAngle_AI, Triangulation_Detached_AI, Visibility_Angle_AI, Triangulation_Visibility_AI, Visibility_Detached_AI
from experiment_visibility.goals import AvoidObstacleGoal, PolarGoToTargetGazeFixationGoal, PolarGoToTargetGoal
from experiment_visibility.measurement_models import Visibility_MM, Angle_MM

# =============================================================================================================================================================

class VisibilityAICON(AICON):
    def __init__(self, env_config):
        self.type = "Visibility"
        super().__init__(**env_config)

    def define_estimators(self):
        REs: Dict[str, RecursiveEstimator] = {
            "RobotVel": Robot_Vel_Estimator_Vel(self.device) if self.vel_control else Robot_Vel_Estimator_Acc(self.device),
            # TODO: use this or nah?
            #"CartesianTargetPos": Cartesian_Pos_Estimator(self.device, "CartesianTargetPos"),
            #"PolarTargetPos": Polar_Pos_Estimator_Vel(self.device, "PolarTargetPos") if self.vel_control else Polar_Pos_Estimator_Acc(self.device, "PolarTargetPos"),
            "PolarTargetDistance": Polar_Distance_Estimator(self.device, "PolarTargetDistance"),
            "PolarTargetAngle": Polar_Angle_Estimator(self.device, "PolarTargetAngle"),
            "TargetVisibility": Target_Visibility_Estimator(self.device, "TargetVisibility"),
        }
        if not self.vel_control:
            raise NotImplementedError("Only velocity control is implemented for Polar Distance and Angle Estimators")
        for i in range(1, self.num_obstacles + 1):
            REs[f"PolarObstacle{i}Pos"] = Polar_Pos_Estimator_Vel(self.device, f"PolarObstacle{i}Pos") if self.vel_control else Polar_Pos_Estimator_Acc(self.device, f"PolarObstacle{i}Pos")
            #REs[f"CartesianObstacle{i}Pos"] = Cartesian_Pos_Estimator(self.device, f"CartesianObstacle{i}Pos")
            REs[f"Obstacle{i}Rad"] = Obstacle_Rad_Estimator(self.device, f"Obstacle{i}Rad")
        return REs

    def define_measurement_models(self):
        MMs = {
            "RobotVel": Vel_MM(self.device),
            "PolarTargetAngle": Angle_MM(self.device, "Target"),
            #"CartesianTargetPos-Angle": Pos_Angle_MM(self.device, "Target"),
            "TargetVisiblity": Visibility_MM(self.device, object_name="Target"),
        }
        for i in range(1, self.num_obstacles + 1):
            MMs[f"PolarObstacle{i}Angle"] = Angle_MM(self.device, f"Obstacle{i}")
            #MMs[f"CartesianObstacle{i}Pos-Angle"] = Pos_Angle_MM(self.device, f"Obstacle{i}")
        return MMs

    def define_active_interconnections(self):
        AIs = {
            #"Triangulation": Triangulation_Visibility_AI([self.REs["PolarTargetPos"], self.REs["TargetVisibility"], self.REs["RobotVel"]], self.device),
            "Triangulation": Triangulation_Detached_AI([self.REs["PolarTargetDistance"], self.REs["PolarTargetAngle"], self.REs["TargetVisibility"], self.REs["RobotVel"]], self.device),
            #"TargetVisibility": Visibility_Angle_AI([self.REs["PolarTargetPos"], self.REs["TargetVisibility"]], self.device, object_name="Target", sensor_angle_rad=self.env.robot.sensor_angle),
            "TargetVisibility": Visibility_Detached_AI([self.REs["PolarTargetAngle"], self.REs["TargetVisibility"]], self.device, object_name="Target", sensor_angle_rad=self.env.robot.sensor_angle),
        }
        for i in range(1, self.num_obstacles + 1):
            AIs[f"Obstacle{i}Rad"] = Radius_Pos_VisAngle_AI([self.REs[f"CartesianObstacle{i}Pos"], self.REs[f"Obstacle{i}Rad"], self.obs[f"obstacle{i}_visual_angle"]], self.device, object_name=f"Obstacle{i}")
        return AIs

    def define_goals(self):
        goals = {
            "GazeFixation" : PolarGoToTargetGazeFixationGoal(self.device),
            "PolarGoToTarget" : PolarGoToTargetGoal(self.device),
        }
        for i in range(1, self.num_obstacles + 1):
            goals[f"AvoidObstacle{i}"] = AvoidObstacleGoal(self.device, i)
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
        for i in range(1, self.num_obstacles + 1):
            estimator_means[f"CartesianObstacle{i}Pos"] = self.REs[f"CartesianObstacle{i}Pos"].state_mean
            rad = self.REs[f"Obstacle{i}Rad"].state_mean
            estimator_covs[f"CartesianObstacle{i}Pos"] = torch.tensor([[rad**2, 0], [0, rad**2]], device=self.device)
        return self.env.render(1.0, {key: np.array(mean.cpu()) for key, mean in estimator_means.items()}, {key: np.array(cov.cpu()) for key, cov in estimator_covs.items()})

    def eval_update(self, action: torch.Tensor, new_step: bool, buffer_dict: Dict[str, Dict[str, torch.Tensor]]):
        u = self.get_control_input(action)

        # ----------------------------- predicts -------------------------------------

        self.REs["RobotVel"].call_predict(u, buffer_dict)
        #self.REs["PolarTargetPos"].call_predict(u, buffer_dict)
        self.REs["PolarTargetDistance"].call_predict(u, buffer_dict)
        self.REs["PolarTargetAngle"].call_predict(u, buffer_dict)
        self.REs["TargetVisibility"].call_predict(u, buffer_dict)
        # for i in range(1, self.num_obstacles + 1):
        #     self.REs[f"PolarObstacle{i}Pos"].call_predict(u, buffer_dict)
        #     self.REs[f"CartesianObstacle{i}Pos"].call_predict(u, buffer_dict)
        #     self.REs[f"Obstacle{i}Rad"].call_predict(u, buffer_dict)

        # ----------------------------- measurements -------------------------------------

        if new_step:
            self.meas_updates(buffer_dict)

        # ----------------------------- active interconnections -------------------------------------

        #self.REs["TargetVisibility"].call_update_with_active_interconnection(self.AIs["TargetVisibility"], buffer_dict)
        #self.REs["PolarTargetPos"].call_update_with_active_interconnection(self.AIs["Triangulation"], buffer_dict)
        
        #self.REs["TargetVisibility"].call_update_with_active_interconnection(self.AIs["TargetVisibility"], buffer_dict)
        self.REs["PolarTargetAngle"].call_update_with_active_interconnection(self.AIs["TargetVisibility"], buffer_dict)
        self.REs["PolarTargetDistance"].call_update_with_active_interconnection(self.AIs["Triangulation"], buffer_dict)

        # for i in range(1, self.num_obstacles + 1):
        #     self.REs[f"Obstacle{i}Rad"].call_update_with_active_interconnection(self.AIs[f"Obstacle{i}Rad"], buffer_dict)

        # print("------------------- Post Update -------------------")
        # print(buffer_dict['PolarTargetPos']['state_mean'])

        return buffer_dict

    def compute_action(self, gradients):
        goal = "PolarGoToTarget"
        for g in self.goals.keys():
            print(f"{g} Gradient: ", gradients[g])
        if self.vel_control:
            action = 0.9 * self.last_action - 5e-3 * gradients[goal]
            for i in range(self.num_obstacles):
                action -= 1e-1 * gradients[f"AvoidObstacle{i+1}"] / self.num_obstacles
        else:
            action = 0.7 * self.last_action - 5e-2 * gradients[goal]
            for i in range(self.num_obstacles):
                action -= 1e0 * gradients[f"AvoidObstacle{i+1}"] / self.num_obstacles
        return action

    def print_states(self, buffer_dict=None):
        """
        print filter and environment states for debugging
        """
        self.print_state("TargetVisibility", print_cov=2)
        obs = self.env.get_reality()
        print("--------------------------------------------------------------------")
        self.print_state("PolarTargetDistance", buffer_dict=buffer_dict, print_cov=2)
        self.print_state("PolarTargetAngle", buffer_dict=buffer_dict, print_cov=2)
        # TODO: observations can be None now
        print(f"True PolarTargetPos: [{obs['target_distance']:.3f}, {obs['target_offset_angle']:.3f}, {obs['target_distance_dot']:.3f}, {obs['target_offset_angle_dot']:.3f}]")
        print("--------------------------------------------------------------------")
        self.print_state("RobotVel", buffer_dict=buffer_dict) 
        print(f"True RobotVel: [{self.env.robot.vel[0]:.3f}, {self.env.robot.vel[1]:.3f}, {self.env.robot.vel_rot:.3f}]")
        print("--------------------------------------------------------------------")
        if self.num_obstacles > 0:
            [self.print_state(f"CartesianObstacle{i}Pos", print_cov=3) for i in range(1, self.num_obstacles + 1)]
            actual_radii = [self.env.obstacles[i-1].radius for i in range(1, self.num_obstacles + 1)]
            print(f"True Obstacle Radius: {[f'{x:.3f}' for x in actual_radii]}")
            print("--------------------------------------------------------------------")
        #print("====================================================================")
    
    def custom_reset(self):
        self.goals["PolarGoToTarget"].desired_distance = self.env.target.distance
        self.goals["GazeFixation"].desired_distance = self.env.target.distance