import numpy as np
import yaml
import torch
from typing import Dict

from components.aicon import AICON
from environment.gaze_fix_env import GazeFixEnv
from experiment_contingent_estimator.active_interconnections import Vel_AI
from experiment_contingent_estimator.estimators import Robot_State_Estimator_Acc, Robot_State_Estimator_Vel
from experiment_contingent_estimator.goals import GoToTargetGoal

# ========================================================================================================

class ContingentEstimatorAICON(AICON):
    def __init__(self, vel_control=False):
        self.vel_control = vel_control
        super().__init__()

    def define_env(self):
        config = 'config/env_config.yaml'
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
        return estimators

    def define_active_interconnections(self):
        active_interconnections = {
            "VelAI": Vel_AI([self.REs["RobotState"], self.obs["vel_frontal"], self.obs["vel_lateral"], self.obs["vel_rot"]], self.device),
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

        if self.vel_control:
            vel = action * torch.tensor([self.env.robot.max_vel, self.env.robot.max_vel, self.env.robot.max_vel_rot])
            u = torch.concat([torch.tensor([0.05], device=self.device), vel])
        else:
            acc = action * torch.tensor([self.env.robot.max_acc, self.env.robot.max_acc, self.env.robot.max_acc_rot], device=self.device, dtype=self.dtype)
            u = torch.concat([torch.tensor([0.05], device=self.device), acc])

        self.REs["RobotState"].call_update_with_specific_meas(self.AIs["VelAI"], buffer_dict)

        print(f"After Meas Update: {buffer_dict['RobotState']['state_mean']}")

        self.REs["RobotState"].call_predict(u, buffer_dict)

        print(f"After Predict 1: {buffer_dict['RobotState']['state_mean']}")

        self.REs["RobotState"].call_predict(u, buffer_dict)

        print(f"After Predict 2: {buffer_dict['RobotState']['state_mean']}")

        print("===================================================================")

        return buffer_dict

    def render(self):
        self.env.render(1.0)

    def compute_action(self, gradients):
        return self.last_action - 1.0 * gradients["GoToTarget"]
    
    def print_states(self):
        self.print_state("RobotState")