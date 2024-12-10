import numpy as np
import yaml
import torch
from typing import Dict

from components.aicon import AICON
from environment.gaze_fix_env import GazeFixEnv
from experiment_general.estimators import Obstacle_Rad_Estimator, Polar_Pos_Estimator_Vel, Cartesian_Pos_Estimator, RecursiveEstimator, Robot_Vel_Estimator_Vel, Robot_Vel_Estimator_Acc
from experiment_general.active_interconnections import Angle_Meas_AI, Cartesian_Polar_AI, Pos_Angle_AI, Radius_Pos_VisAngle_AI, Triangulation_AI, Vel_AI
from experiment_general.goals import AvoidObstacleGoal, GazeFixationGoal, GoToTargetGoal, PolarGoToTargetGoal

# =============================================================================================================================================================

class GeneralTestAICON(AICON):
    def __init__(self, num_obstacles=0, vel_control=False, moving_target=False, sensor_angle_deg=360):
        self.num_obstacles = num_obstacles
        self.moving_target = moving_target
        self.vel_control = vel_control
        self.sensor_angle_deg = sensor_angle_deg
        super().__init__()

    def define_env(self):
        config = 'environment/env_config.yaml'
        with open(config) as file:
            env_config = yaml.load(file, Loader=yaml.FullLoader)
            env_config["num_obstacles"] = self.num_obstacles
            env_config["action_mode"] = 3 if self.vel_control else 1
            env_config["moving_target"] = self.moving_target
            env_config["robot_sensor_angle"] = self.sensor_angle_deg / 180 * np.pi
        return GazeFixEnv(env_config)

    def define_estimators(self):
        REs: Dict[str, RecursiveEstimator] = {"RobotVel": Robot_Vel_Estimator_Vel(self.device) if self.vel_control else Robot_Vel_Estimator_Acc(self.device)}
        REs["CartesianTargetPos"] = Cartesian_Pos_Estimator(self.device, "CartesianTargetPos")
        REs["PolarTargetPos"] = Polar_Pos_Estimator_Vel(self.device, "PolarTargetPos")
        for i in range(1, self.num_obstacles + 1):
            REs[f"CartesianObstacle{i}Pos"] = Cartesian_Pos_Estimator(self.device, f"CartesianObstacle{i}Pos")
            REs[f"Obstacle{i}Rad"] = Obstacle_Rad_Estimator(self.device, f"Obstacle{i}Rad")
        return REs

    def define_active_interconnections(self):
        AIs = {
            "RobotVel": Vel_AI([self.REs["RobotVel"], self.obs["vel_frontal"], self.obs["vel_lateral"], self.obs["vel_rot"]], self.device),
            #"CartPolarPos": Cartesian_Polar_AI([REs["CartesianTargetPos"], REs["PolarTargetPos"]], device),
        }
        AIs["PolarAngle"] = Angle_Meas_AI([self.REs["PolarTargetPos"], self.obs["target_offset_angle"], self.obs["del_target_offset_angle"]], self.device)
        AIs["PolarDistance"] = Triangulation_AI([self.REs["PolarTargetPos"], self.REs["RobotVel"]], self.device)
        AIs["CartesianTargetPos-Angle"] = Pos_Angle_AI([self.REs["CartesianTargetPos"], self.REs["RobotVel"], self.obs["target_offset_angle"]], self.device)
        for i in range(1, self.num_obstacles + 1):
            AIs[f"CartesianObstacle{i}Pos-Angle"] = Pos_Angle_AI([self.REs[f"CartesianObstacle{i}Pos"], self.REs["RobotVel"], self.obs[f"obstacle{i}_offset_angle"]], self.device, object_name=f"Obstacle{i}")
            AIs[f"Obstacle{i}Rad"] = Radius_Pos_VisAngle_AI([self.REs[f"CartesianObstacle{i}Pos"], self.REs[f"Obstacle{i}Rad"], self.obs[f"obstacle{i}_visual_angle"]], self.device, object_name=f"Obstacle{i}")
        return AIs

    def define_goals(self):
        goals = {
            "Go-To-Target" : GoToTargetGoal(self.device),
            "PolarGo-To-Target" : PolarGoToTargetGoal(self.device),
            "GazeFixation" : GazeFixationGoal(self.device),
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

    def eval_step(self, action: torch.Tensor, new_step = False):
        self.update_observations()
        buffer_dict = {key: estimator.set_buffer_dict() for key, estimator in list(self.REs.items()) + list(self.obs.items())}

        u = self.get_control_input(action)

        # ----------------------------- update vel -------------------------------------
        self.REs["RobotVel"].call_predict(u, buffer_dict)
        if new_step:
            self.REs["RobotVel"].call_update_with_specific_meas(self.AIs["RobotVel"], buffer_dict)

        # ----------------------------- update and predict -------------------------------------

        #self.REs["CartesianTargetPos"].call_predict(u, buffer_dict)
        self.REs["PolarTargetPos"].call_predict(u, buffer_dict)
        for i in range(1, self.num_obstacles + 1):
            self.REs[f"CartesianObstacle{i}Pos"].call_predict(u, buffer_dict)
            self.REs[f"Obstacle{i}Rad"].call_predict(u, buffer_dict)

        if new_step:
            self.REs["PolarTargetPos"].call_update_with_specific_meas(self.AIs["PolarAngle"], buffer_dict)
            self.REs["PolarTargetPos"].call_update_with_specific_meas(self.AIs["PolarDistance"], buffer_dict)
            self.REs["CartesianTargetPos"].call_update_with_specific_meas(self.AIs["CartesianTargetPos-Angle"], buffer_dict)
            for i in range(1, self.num_obstacles + 1):
                self.REs[f"CartesianObstacle{i}Pos"].call_update_with_specific_meas(self.AIs[f"CartesianObstacle{i}Pos-Angle"], buffer_dict)
                self.REs[f"Obstacle{i}Rad"].call_update_with_specific_meas(self.AIs[f"Obstacle{i}Rad"], buffer_dict)
        else:
            # TODO: instable with acc control
            if self.vel_control:
                self.REs["PolarTargetPos"].call_update_with_specific_meas(self.AIs["PolarDistance"], buffer_dict)
        
        return buffer_dict
    
    def get_control_input(self, action: torch.Tensor):
        env_action = torch.empty_like(action)
        env_action[:2] = (action[:2] / action[:2].norm() if action[:2].norm() > 1.0 else action[:2]) * (self.env.robot.max_vel if self.vel_control else self.env.robot.max_acc)
        env_action[2] = action[2] * (self.env.robot.max_vel_rot if self.vel_control else self.env.robot.max_acc_rot)
        return torch.concat([torch.tensor([0.05], device=self.device), env_action])

    def compute_action(self, gradients):
        if self.vel_control:
            action = 0.9 * self.last_action - 5e-2 * gradients["PolarGo-To-Target"]
            for i in range(self.num_obstacles):
                action -= 1e-1 * gradients[f"AvoidObstacle{i+1}"] / self.num_obstacles
        else:
            action = 0.7 * self.last_action - 5e-3 * gradients["PolarGo-To-Target"]
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
        obs = self.env.get_observation()
        self.print_state("RobotVel", print_cov=print_cov)
        actual_vel = list(self.env.robot.vel)
        actual_vel.append(self.env.robot.vel_rot)
        print(f"True Robot Vel: {[f'{x:.3f}' for x in actual_vel]}")
        # print("--------------------------------------------------------------------")
        self.print_state("PolarTargetPos", print_cov=print_cov)
        actual_pos = self.env.rotation_matrix(-self.env.robot.orientation) @ (self.env.target.pos - self.env.robot.pos)
        angle = np.arctan2(actual_pos[1], actual_pos[0])
        dist = np.linalg.norm(actual_pos)
        print(f"True PolarTargetPos: {[f'{x:.3f}' for x in [dist, angle, obs['del_robot_target_distance'], obs['del_target_offset_angle']]]}")
        print("--------------------------------------------------------------------")
        if self.num_obstacles > 0:
            [self.print_state(f"CartesianObstacle{i}Pos", print_cov=print_cov) for i in range(1, self.num_obstacles + 1)]
            actual_radii = [self.env.obstacles[i-1].radius for i in range(1, self.num_obstacles + 1)]
            print(f"True Obstacle Radius: {[f'{x:.3f}' for x in actual_radii]}")
            print("--------------------------------------------------------------------")
        print("====================================================================")

    def convert_polar_to_cartesian_state(self, polar_mean, polar_cov):
        mean = torch.stack([
            polar_mean[0] * torch.cos(polar_mean[1]),
            polar_mean[0] * torch.sin(polar_mean[1])
        ])
        cov = torch.zeros((2, 2), device=self.device)
        cov[0, 0] = polar_cov[0, 0] * torch.cos(polar_mean[1])**2 + polar_cov[1, 1] * torch.sin(polar_mean[1])**2
        cov[1, 1] = polar_cov[0, 0] * torch.sin(polar_mean[1])**2 + polar_cov[1, 1] * torch.cos(polar_mean[1])**2
        cov[0, 1] = (polar_cov[0, 0] - polar_cov[1, 1]) * torch.cos(polar_mean[1]) * torch.sin(polar_mean[1])
        cov[1, 0] = cov[0, 1]
        return mean, cov
    
    def custom_reset(self):
        self.update_observations()
        self.goals["PolarGo-To-Target"].desired_distance = self.obs["target_distance"].state_mean.item()