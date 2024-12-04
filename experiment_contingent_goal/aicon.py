import numpy as np
import yaml
import torch
from typing import Dict

from components.aicon import AICON
from environment.gaze_fix_env import GazeFixEnv
from experiment_contingent_goal.active_interconnections import Angle_Meas_AI, Gaze_Fixation_AI, Triangulation_AI, Triangulation_AI_Decoupled, Vel_AI
from experiment_contingent_goal.estimators import Polar_Pos_Estimator_Acc, Polar_Pos_Estimator_Vel, Robot_Vel_Estimator_Acc, Robot_Vel_Estimator_Vel
from experiment_contingent_goal.goals import GoToTargetGoal

# ========================================================================================================

class ContingentGoalAICON(AICON):
    def __init__(self):
        super().__init__()

    def define_env(self):
        config = 'config/env_config.yaml'
        with open(config) as file:
            env_config = yaml.load(file, Loader=yaml.FullLoader)
            # vel control:
            env_config["action_mode"] = 3
        return GazeFixEnv(env_config)

    def define_estimators(self):
        estimators = {}
        estimators["RobotVel"] = Robot_Vel_Estimator_Vel(self.device)
        estimators["PolarTargetPos"] = Polar_Pos_Estimator_Vel(self.device, "PolarTargetPos")
        return estimators

    def define_active_interconnections(self):
        active_interconnections = {
            "VelAI": Vel_AI([self.REs["RobotVel"], self.obs["vel_frontal"], self.obs["vel_lateral"], self.obs["vel_rot"]], self.device),
            "AngleMeasAI": Angle_Meas_AI([self.REs["PolarTargetPos"], self.obs["target_offset_angle"], self.obs["del_target_offset_angle"]], self.device),
            "TriangulationAI": Triangulation_AI([self.REs["PolarTargetPos"], self.REs["RobotVel"]], self.device),
        }
        return active_interconnections

    def define_goals(self):
        goals = {
            "GoToTarget": GoToTargetGoal(self.device),
        }
        return goals

    def eval_step(self, action, new_step = False):
        self.update_observations()
        buffer_dict = {key: estimator.set_buffer_dict() for key, estimator in list(self.REs.items()) + list(self.obs.items())}

        u = self.get_control_input(action)

        if new_step:
            self.REs["RobotVel"].call_predict(u, buffer_dict)
            self.REs["RobotVel"].call_update_with_specific_meas(self.AIs["VelAI"], buffer_dict)

            self.REs["PolarTargetPos"].call_predict(u, buffer_dict)
            self.REs["PolarTargetPos"].call_update_with_specific_meas(self.AIs["AngleMeasAI"], buffer_dict)
            self.REs["PolarTargetPos"].call_update_with_specific_meas(self.AIs["TriangulationAI"], buffer_dict)

        else:
            self.REs["RobotVel"].call_predict(u, buffer_dict)
            self.REs["PolarTargetPos"].call_predict(u, buffer_dict)
            self.REs["PolarTargetPos"].call_update_with_specific_meas(self.AIs["TriangulationAI"], buffer_dict)

        return buffer_dict

    def get_control_input(self, action):
        env_action = torch.empty_like(action)
        env_action[:2] = (action[:2] / action[:2].norm() if action[:2].norm() > 1.0 else action[:2]) * self.env.robot.max_vel
        env_action[2] = action[2] * self.env.robot.max_vel_rot
        return torch.concat([torch.tensor([0.05], device=self.device), env_action])

    def render(self):
        estimator_means = {"PolarTargetPos": np.array(torch.stack([
            self.REs["PolarTargetPos"].state_mean[0] * torch.cos(self.REs["PolarTargetPos"].state_mean[1]),
            self.REs["PolarTargetPos"].state_mean[0] * torch.sin(self.REs["PolarTargetPos"].state_mean[1])
        ]).cpu())}
        cart_cov = torch.zeros((2, 2), device=self.device)
        cart_cov[0, 0] = self.REs["PolarTargetPos"].state_cov[0, 0] * torch.cos(self.REs["PolarTargetPos"].state_mean[1])**2 + self.REs["PolarTargetPos"].state_cov[1, 1] * torch.sin(self.REs["PolarTargetPos"].state_mean[1])**2
        cart_cov[0, 1] = (self.REs["PolarTargetPos"].state_cov[0, 0] - self.REs["PolarTargetPos"].state_cov[1, 1]) * torch.cos(self.REs["PolarTargetPos"].state_mean[1]) * torch.sin(self.REs["PolarTargetPos"].state_mean[1])
        cart_cov[1, 0] = cart_cov[0, 1]
        cart_cov[1, 1] = self.REs["PolarTargetPos"].state_cov[0, 0] * torch.sin(self.REs["PolarTargetPos"].state_mean[1])**2 + self.REs["PolarTargetPos"].state_cov[1, 1] * torch.cos(self.REs["PolarTargetPos"].state_mean[1])**2
        estimator_covs = {"PolarTargetPos": np.array(cart_cov.cpu())}
        return self.env.render(1.0, estimator_means, estimator_covs)

    def compute_action(self, gradients):
            decay = 0.9
            return decay * self.last_action - 1e-2 * gradients["GoToTarget"]
    
    def print_states(self, buffer_dict=None):
        obs = self.env.get_observation()
        print("--------------------------------------------------------------------")
        self.print_state("PolarTargetPos", buffer_dict=buffer_dict)
        actual_pos = self.env.rotation_matrix(-self.env.robot.orientation) @ (self.env.target.pos - self.env.robot.pos)
        angle = np.arctan2(actual_pos[1], actual_pos[0])
        dist = np.linalg.norm(actual_pos)
        print(f"True Polar Target Position: {[f'{x:.3f}' for x in [dist, angle, obs['del_robot_target_distance'], obs['del_target_offset_angle']]]}")
        print("--------------------------------------------------------------------")
        self.print_state("RobotVel", buffer_dict=buffer_dict)
        actual_vel = [self.env.robot.vel[0], self.env.robot.vel[1], self.env.robot.vel_rot] 
        print(actual_vel)
        print("--------------------------------------------------------------------")

    def custom_reset(self):
        self.update_observations()
        self.goals["GoToTarget"].desired_distance = self.obs["target_distance"].state_mean.item()
