import numpy as np
import torch
from typing import Dict

from components.aicon import DroneEnvAICON as AICON
from components.estimator import RecursiveEstimator
from components.instances.estimators import Robot_Vel_Estimator_Vel, Robot_Vel_Estimator_Acc, Polar_Pos_Estimator_Vel, Polar_Pos_Estimator_Acc, Cartesian_Pos_Estimator
from components.instances.measurement_models import Vel_MM, Pos_Angle_MM, Angle_Meas_MM
from components.instances.active_interconnections import Triangulation_AI

from experiment_general.estimators import Obstacle_Rad_Estimator, Target_Visibility_Estimator
from experiment_general.active_interconnections import Radius_Pos_VisAngle_AI, Visibility_Angle_AI
from experiment_general.measurement_models import Visibility_MM
from experiment_general.goals import AvoidObstacleGoal, PolarGoToTargetGazeFixationGoal, PolarGoToTargetGoal

# =============================================================================================================================================================

class GeneralTestAICON(AICON):
    def __init__(self, env_config):
        self.type = "GeneralTest"
        super().__init__(**env_config)

    def define_estimators(self):
        REs: Dict[str, RecursiveEstimator] = {
            "RobotVel": Robot_Vel_Estimator_Vel(self.device) if self.vel_control else Robot_Vel_Estimator_Acc(self.device),
            # TODO: use this or nah?
            "CartesianTargetPos": Cartesian_Pos_Estimator(self.device, "CartesianTargetPos"),
            "PolarTargetPos": Polar_Pos_Estimator_Vel(self.device, "PolarTargetPos") if self.vel_control else Polar_Pos_Estimator_Acc(self.device, "PolarTargetPos"),
            "TargetVisibility": Target_Visibility_Estimator(self.device, "TargetVisibility"),
        }
        for i in range(1, self.num_obstacles + 1):
            REs[f"CartesianObstacle{i}Pos"] = Cartesian_Pos_Estimator(self.device, f"CartesianObstacle{i}Pos")
            REs[f"Obstacle{i}Rad"] = Obstacle_Rad_Estimator(self.device, f"Obstacle{i}Rad")
        return REs

    def define_measurement_models(self):
        MMs = {
            "RobotVel": Vel_MM(self.device),
            "PolarAngle": Angle_Meas_MM(self.device, "Target"),
            "CartesianTargetPos-Angle": Pos_Angle_MM(self.device, "Target"),
            "TargetVisiblity": Visibility_MM(self.device, object_name="Target"),
        }
        for i in range(1, self.num_obstacles + 1):
            MMs[f"CartesianObstacle{i}Pos-Angle"] = Pos_Angle_MM(self.device, f"Obstacle{i}")
        return MMs

    def define_active_interconnections(self):
        AIs = {
            "PolarDistance": Triangulation_AI([self.REs["PolarTargetPos"], self.REs["RobotVel"]], self.device),
            "TargetVisibility": Visibility_Angle_AI([self.REs["PolarTargetPos"], self.REs["TargetVisibility"]], self.device, object_name="Target", sensor_angle_rad=self.env.robot.sensor_angle),
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
        target_mean, target_cov = self.convert_polar_to_cartesian_state(self.REs["PolarTargetPos"].state_mean, self.REs["PolarTargetPos"].state_cov)
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
        self.REs["PolarTargetPos"].call_predict(u, buffer_dict)
        #self.REs["TargetVisibility"].call_predict(u, buffer_dict)
        
        # for i in range(1, self.num_obstacles + 1):
        #     self.REs[f"CartesianObstacle{i}Pos"].call_predict(u, buffer_dict)
        #     self.REs[f"Obstacle{i}Rad"].call_predict(u, buffer_dict)

        # print("------------------- Post Predict -------------------")
        # print(buffer_dict['PolarTargetPos']['state_mean'])

        # ----------------------------- measurements -------------------------------------

        if new_step:
            self.meas_updates(buffer_dict)

        # ----------------------------- active interconnections -------------------------------------

        self.REs["PolarTargetPos"].call_update_with_active_interconnection(self.AIs["PolarDistance"], buffer_dict)

        #self.REs["PolarTargetPos"].call_update_with_active_interconnection(self.AIs["TargetVisibility"], buffer_dict)
        #self.REs["TargetVisibility"].call_update_with_active_interconnection(self.AIs["TargetVisibility"], buffer_dict)
        
        # for i in range(1, self.num_obstacles + 1):
        #     self.REs[f"Obstacle{i}Rad"].call_update_with_active_interconnection(self.AIs[f"Obstacle{i}Rad"], buffer_dict)

        # print("------------------- Post Update -------------------")
        # print(buffer_dict['PolarTargetPos']['state_mean'])

        return buffer_dict

    def compute_action(self, gradients):
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

    def print_states(self, print_cov=False):
        """
        print filter and environment states for debugging
        """
        self.print_state("TargetVisibility", print_cov=print_cov)
        obs = self.env.get_reality()
        self.print_state("RobotVel", print_cov=print_cov)
        actual_vel = list(self.env.robot.vel)
        actual_vel.append(self.env.robot.vel_rot)
        print(f"True Robot Vel: {[f'{x:.3f}' for x in actual_vel]}")
        # print("--------------------------------------------------------------------")
        self.print_state("PolarTargetPos", print_cov=print_cov)
        actual_pos = self.env.rotation_matrix(-self.env.robot.orientation) @ (self.env.target.pos - self.env.robot.pos)
        angle = np.arctan2(actual_pos[1], actual_pos[0])
        dist = np.linalg.norm(actual_pos)
        print(f"True PolarTargetPos: {[f'{x:.3f}' for x in [dist, angle, obs['target_distance_dot'], obs['target_offset_angle_dot'] if obs['target_offset_angle_dot'] else 0.0]]}")
        print("--------------------------------------------------------------------")
        if self.num_obstacles > 0:
            [self.print_state(f"CartesianObstacle{i}Pos", print_cov=print_cov) for i in range(1, self.num_obstacles + 1)]
            actual_radii = [self.env.obstacles[i-1].radius for i in range(1, self.num_obstacles + 1)]
            print(f"True Obstacle Radius: {[f'{x:.3f}' for x in actual_radii]}")
            print("--------------------------------------------------------------------")
        #print("====================================================================")
    
    def custom_reset(self):
        self.goals["PolarGoToTarget"].desired_distance = self.env.target.distance
        self.goals["GazeFixation"].desired_distance = self.env.target.distance