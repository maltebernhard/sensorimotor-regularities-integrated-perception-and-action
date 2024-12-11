import numpy as np
import torch

from components.aicon import DroneEnvAICON as AICON
from components.instances.measurement_models import Angle_Meas_MM, Vel_MM

from experiment_contingent_estimator.active_interconnections import DistanceUpdaterAcc, DistanceUpdaterVel
from experiment_contingent_estimator.estimators import Robot_State_Estimator_Acc, Robot_State_Estimator_Vel
from experiment_contingent_estimator.goals import SpecificGoToTargetGoal

# ========================================================================================================

class ContingentEstimatorAICON(AICON):
    def __init__(self, vel_control=True, moving_target=False, sensor_angle_deg=360, num_obstacles=0):
        super().__init__(vel_control, moving_target, sensor_angle_deg, num_obstacles)

    def define_estimators(self):
        estimators = {}
        estimators["RobotState"] = Robot_State_Estimator_Acc(self.device) if not self.vel_control else Robot_State_Estimator_Vel(self.device)
        #estimators["PolarTargetPos"] = Polar_Pos_Estimator_Acc(self.device, "PolarTargetPos") if not self.vel_control else Polar_Pos_Estimator_Vel(self.device, "PolarTargetPos")
        return estimators
    
    def define_measurement_models(self):
        return {
            "VelMM": Vel_MM(self.device),
            "AngleMeasMM": Angle_Meas_MM(self.device, "Target"),
        }

    def define_active_interconnections(self):
        active_interconnections = {
            #"TriangulationAI": Triangulation_AI([self.REs["PolarTargetPos"], self.REs["RobotState"]], self.device),
            #"DistanceUpdater": DistanceUpdaterAcc([self.REs["PolarTargetPos"], self.REs["RobotState"]], self.device) if not self.vel_control else DistanceUpdaterVel([self.REs["PolarTargetPos"], self.REs["RobotState"]], self.device),
        }
        return active_interconnections

    def define_goals(self):
        goals = {
            "GoToTarget": SpecificGoToTargetGoal(self.device),
        }
        return goals

    def eval_step(self, action, new_step = False):
        self.update_observations()
        buffer_dict = {key: estimator.set_buffer_dict() for key, estimator in list(self.REs.items()) + list(self.obs.items())}
        u = self.get_control_input(action)
        self.REs["RobotState"].call_predict(u, buffer_dict)
        if new_step:
            self.REs["RobotState"].call_update_with_meas_model(self.MMs["VelMM"], buffer_dict, self.get_meas_dict(self.MMs["VelMM"]))
            self.REs["RobotState"].call_update_with_meas_model(self.MMs["AngleMeasMM"], buffer_dict, self.get_meas_dict(self.MMs["AngleMeasMM"]))
        else:
            pass
        return buffer_dict

    def get_control_input(self, action: torch.Tensor):
        env_action = torch.empty_like(action)
        env_action[:2] = (action[:2] / action[:2].norm() if action[:2].norm() > 1.0 else action[:2]) * (self.env.robot.max_vel if self.vel_control else self.env.robot.max_acc)
        env_action[2] = action[2] * (self.env.robot.max_vel_rot if self.vel_control else self.env.robot.max_acc_rot)
        return torch.concat([torch.tensor([0.05], device=self.device), env_action])

    def render(self):
        estimator_means = {"RobotState": np.array(torch.stack([
            self.REs["RobotState"].state_mean[3] * torch.cos(self.REs["RobotState"].state_mean[4]),
            self.REs["RobotState"].state_mean[3] * torch.sin(self.REs["RobotState"].state_mean[4])
        ]).cpu())}
        cart_cov = torch.zeros((2, 2), device=self.device)
        cart_cov[0, 0] = self.REs["RobotState"].state_cov[3, 3] * torch.cos(self.REs["RobotState"].state_mean[4])**2 + self.REs["RobotState"].state_cov[4, 4] * torch.sin(self.REs["RobotState"].state_mean[4])**2
        cart_cov[0, 1] = (self.REs["RobotState"].state_cov[3, 3] - self.REs["RobotState"].state_cov[4, 4]) * torch.cos(self.REs["RobotState"].state_mean[4]) * torch.sin(self.REs["RobotState"].state_mean[4])
        cart_cov[1, 0] = cart_cov[0, 1]
        cart_cov[1, 1] = self.REs["RobotState"].state_cov[3, 3] * torch.sin(self.REs["RobotState"].state_mean[4])**2 + self.REs["RobotState"].state_cov[4, 4] * torch.cos(self.REs["RobotState"].state_mean[4])**2
        estimator_covs = {"RobotState": np.array(cart_cov.cpu())}
        return self.env.render(1.0, estimator_means, estimator_covs)

    def compute_action(self, gradients):
        if not self.vel_control:
            return self.last_action - 1e0 * gradients["GoToTarget"]
        else:
            return self.last_action - 1e-2 * gradients["GoToTarget"]
    
    def print_states(self, buffer_dict=None):
        obs = self.env.get_observation()
        print("--------------------------------------------------------------------")
        self.print_state("RobotState", buffer_dict=buffer_dict)
        actual_pos = self.env.rotation_matrix(-self.env.robot.orientation) @ (self.env.target.pos - self.env.robot.pos)
        angle = np.arctan2(actual_pos[1], actual_pos[0])
        dist = np.linalg.norm(actual_pos)
        # TODO: observations can be None now
        #print(f"True Polar Target Position: {[f'{x:.3f}' for x in [dist, angle, obs['del_robot_target_distance'], obs['del_target_offset_angle']]]}")
        print(f"True Polar Target Position: {[f'{x:.3f}' for x in [dist, angle]]}")
        actual_state = [self.env.robot.vel[0], self.env.robot.vel[1], self.env.robot.vel_rot]
        actual_pos = self.env.rotation_matrix(-self.env.robot.orientation) @ (self.env.target.pos - self.env.robot.pos)
        angle = np.arctan2(actual_pos[1], actual_pos[0])
        dist = np.linalg.norm(actual_pos)
        # TODO: observations can be None now
        #actual_state += [dist, angle, obs['del_robot_target_distance'], obs['del_target_offset_angle']]
        actual_state += [dist, angle]
        print(f"True State: {actual_state}")
        print("--------------------------------------------------------------------")

    def custom_reset(self):
        self.update_observations()
        self.goals["GoToTarget"].desired_distance = self.obs["target_distance"].state_mean.item()