import numpy as np
import yaml
import torch
from typing import Dict

from components.aicon import AICON
from environment.gaze_fix_env import GazeFixEnv
from experiment_contingent_interconnection.active_interconnections import Angle_Meas_AI, Triangulation_AI, Vel_AI, Gaze_Fixation_AI
from experiment_contingent_interconnection.estimators import Polar_Pos_Estimator_Acc, Polar_Pos_Estimator_Vel, Robot_State_Estimator_Acc, Robot_State_Estimator_Vel
from experiment_contingent_interconnection.goals import GoToTargetGoal

# ========================================================================================================

class ContingentInterconnectionAICON(AICON):
    def __init__(self, vel_control=False):
        self.vel_control = vel_control
        super().__init__()

    def define_env(self):
        config = 'environment/env_config.yaml'
        with open(config) as file:
            env_config = yaml.load(file, Loader=yaml.FullLoader)
            if self.vel_control:
                env_config["action_mode"] = 3
            else:
                env_config["action_mode"] = 1
        return GazeFixEnv(env_config)

    def define_estimators(self):
        estimators = {}
        estimators["RobotState"] = Robot_State_Estimator_Acc(self.device) if not self.vel_control else Robot_State_Estimator_Vel(self.device)
        estimators["PolarTargetPos"] = Polar_Pos_Estimator_Acc(self.device, "PolarTargetPos") if not self.vel_control else Polar_Pos_Estimator_Vel(self.device, "PolarTargetPos")
        return estimators

    def define_measurement_models(self):
        return {}

    def define_active_interconnections(self):
        active_interconnections = {
            "VelAI": Vel_AI([self.REs["RobotState"], self.obs["vel_frontal"], self.obs["vel_lateral"], self.obs["vel_rot"]], self.device),
            "AngleMeasAI": Angle_Meas_AI([self.REs["PolarTargetPos"], self.obs["target_offset_angle"], self.obs["del_target_offset_angle"]], self.device),
            "TriangulationAI": Triangulation_AI([self.REs["PolarTargetPos"], self.REs["RobotState"]], self.device),
            "GazeFixation": Gaze_Fixation_AI([self.REs["PolarTargetPos"], self.REs["RobotState"]], self.device),
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
            self.REs["RobotState"].call_predict(u, buffer_dict)
            self.REs["RobotState"].call_update_with_active_interconnection(self.AIs["VelAI"], buffer_dict)
            self.REs["PolarTargetPos"].call_predict(u, buffer_dict)
            self.REs["PolarTargetPos"].call_update_with_active_interconnection(self.AIs["AngleMeasAI"], buffer_dict)
            self.REs["PolarTargetPos"].call_update_with_active_interconnection(self.AIs["TriangulationAI"], buffer_dict)
        else:
            self.REs["RobotState"].call_predict(u, buffer_dict)
            self.REs["PolarTargetPos"].call_predict(u, buffer_dict)
            self.REs["PolarTargetPos"].call_update_with_active_interconnection(self.AIs["GazeFixation"], buffer_dict)
            self.REs["RobotState"].call_update_with_active_interconnection(self.AIs["GazeFixation"], buffer_dict)
            self.REs["PolarTargetPos"].call_update_with_active_interconnection(self.AIs["TriangulationAI"], buffer_dict)

        return buffer_dict

    def get_control_input(self, action: torch.Tensor):
        env_action = torch.empty_like(action)
        env_action[:2] = (action[:2] / action[:2].norm() if action[:2].norm() > 1.0 else action[:2]) * (self.env.robot.max_vel if self.vel_control else self.env.robot.max_acc)
        env_action[2] = action[2] * (self.env.robot.max_vel_rot if self.vel_control else self.env.robot.max_acc_rot)
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
        #decay = 1.0
        decay = 0.8
        if not self.vel_control:
            action = decay * self.last_action - 1e0 * gradients["GoToTarget"]
        else:
            action = decay * self.last_action - 5e-2 * gradients["GoToTarget"]
        # manual gaze fixation
        # if self.vel_control:
        #     action[2] = 0.8 * self.REs["PolarTargetPos"].state_mean[1] - 0.05 * self.REs["PolarTargetPos"].state_mean[3]
        return action
    
    def print_states(self, buffer_dict=None):
        obs = self.env.get_observation()
        print("--------------------------------------------------------------------")
        self.print_state("RobotState", buffer_dict=buffer_dict)
        actual_pos = self.env.rotation_matrix(-self.env.robot.orientation) @ (self.env.target.pos - self.env.robot.pos)
        angle = np.arctan2(actual_pos[1], actual_pos[0])
        dist = np.linalg.norm(actual_pos)
        #TODO: Consider observations that are None
        #print(f"True Polar Target Position: {[f'{x:.3f}' for x in [dist, angle, obs['del_robot_target_distance'], obs['del_target_offset_angle']]]}")
        print(f"True Polar Target Position: {[f'{x:.3f}' for x in [dist, angle]]}")
        actual_state = [self.env.robot.vel[0], self.env.robot.vel[1], self.env.robot.vel_rot]
        actual_pos = self.env.rotation_matrix(-self.env.robot.orientation) @ (self.env.target.pos - self.env.robot.pos)
        angle = np.arctan2(actual_pos[1], actual_pos[0])
        dist = np.linalg.norm(actual_pos)
        #TODO: Consider observations that are None
        #actual_state += [dist, angle, obs['del_robot_target_distance'], obs['del_target_offset_angle']]
        actual_state += [dist, angle]
        print(f"True State: ", end="")
        [print(f'{x:.3f} ', end="") for x in actual_state]
        print()
        print("--------------------------------------------------------------------")

    def custom_reset(self):
        self.update_observations()
        self.goals["GoToTarget"].desired_distance = self.obs["target_distance"].state_mean.item()