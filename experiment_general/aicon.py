import numpy as np
import torch
from typing import Dict

from components.aicon import DroneEnvAICON as AICON
from components.estimator import RecursiveEstimator
from components.instances.estimators import Robot_Vel_Estimator_Vel, Robot_Vel_Estimator_Acc, Polar_Pos_Estimator_Vel, Polar_Pos_Estimator_Acc, Cartesian_Pos_Estimator
from components.instances.measurement_models import Vel_MM, Pos_Angle_MM, Angle_Meas_MM
from components.instances.active_interconnections import Triangulation_AI
from components.instances.goals import GazeFixationGoal, CartesianGoToTargetGoal

from experiment_general.estimators import Obstacle_Rad_Estimator, Target_Visibility_Estimator
from experiment_general.active_interconnections import Radius_Pos_VisAngle_AI, Visibility_Angle_AI, Scaled_Triangulation_AI
from experiment_general.measurement_models import Visibility_MM
from experiment_general.goals import AvoidObstacleGoal, PolarGoToTargetGazeFixationGoal, PolarGoToTargetGoal

# =============================================================================================================================================================

class GeneralTestAICON(AICON):
    def __init__(self, vel_control=True, moving_target=False, sensor_angle_deg=360, num_obstacles=0):
        super().__init__(vel_control, moving_target, sensor_angle_deg, num_obstacles)

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
            #"PolarDistance": Scaled_Triangulation_AI([self.REs["PolarTargetPos"], self.REs["RobotVel"]], self.device, max_vel=self.env.robot.max_vel),
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

    def eval_step(self, action: torch.Tensor, new_step = False):
        self.update_observations()
        buffer_dict = {key: estimator.set_buffer_dict() for key, estimator in list(self.REs.items())}

        u = self.get_control_input(action)

        # ----------------------------- predicts -------------------------------------

        self.REs["RobotVel"].call_predict(u, buffer_dict)
        self.REs["PolarTargetPos"].call_predict(u, buffer_dict)
        self.REs["TargetVisibility"].call_predict(u, buffer_dict)
        
        for i in range(1, self.num_obstacles + 1):
            self.REs[f"CartesianObstacle{i}Pos"].call_predict(u, buffer_dict)
            self.REs[f"Obstacle{i}Rad"].call_predict(u, buffer_dict)

        # print("------------------- Post Predict -------------------")
        # print(buffer_dict['PolarTargetPos']['state_mean'])

        # ----------------------------- active interconnections -------------------------------------
        
        self.REs["PolarTargetPos"].call_update_with_active_interconnection(self.AIs["PolarDistance"], buffer_dict)

        self.REs["PolarTargetPos"].call_update_with_active_interconnection(self.AIs["TargetVisibility"], buffer_dict)
        #self.REs["TargetVisibility"].call_update_with_active_interconnection(self.AIs["TargetVisibility"], buffer_dict)
        
        for i in range(1, self.num_obstacles + 1):
            self.REs[f"Obstacle{i}Rad"].call_update_with_active_interconnection(self.AIs[f"Obstacle{i}Rad"], buffer_dict)

        # print("------------------- Post Update -------------------")
        # print(buffer_dict['PolarTargetPos']['state_mean'])

        # ----------------------------- measurements -------------------------------------

        if new_step:
            self.meas_updates(buffer_dict)

        return buffer_dict
    
    def meas_updates(self, buffer_dict):
        for model_key, meas_model in self.MMs.items():
            meas_dict = self.get_meas_dict(self.MMs[model_key])
            if len(meas_dict["means"]) == len(meas_model.observations):
                self.REs[meas_model.estimator].call_update_with_meas_model(meas_model, buffer_dict, meas_dict)
            else:
                print(f"Missing measurements for {model_key}")
    
    def get_control_input(self, action: torch.Tensor):
        env_action = torch.empty_like(action)
        env_action[:2] = (action[:2] / action[:2].norm() if action[:2].norm() > 1.0 else action[:2]) * (self.env.robot.max_vel if self.vel_control else self.env.robot.max_acc)
        env_action[2] = action[2] * (self.env.robot.max_vel_rot if self.vel_control else self.env.robot.max_acc_rot)
        return torch.concat([torch.tensor([0.05], device=self.device), env_action])

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
        # manual slow spinning
        # action[0] = 0.1
        # action[1] = 0.0
        # action[2] = 0.1
        return action

    def print_states(self, print_cov=False):
        """
        print filter and environment states for debugging
        """
        self.print_state("TargetVisibility", print_cov=print_cov)
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
        print(f"True PolarTargetPos: {[f'{x:.3f}' for x in [dist, angle, obs['del_robot_target_distance'], obs['del_target_offset_angle'] if obs['del_target_offset_angle'] else 0.0]]}")
        print("--------------------------------------------------------------------")
        if self.num_obstacles > 0:
            [self.print_state(f"CartesianObstacle{i}Pos", print_cov=print_cov) for i in range(1, self.num_obstacles + 1)]
            actual_radii = [self.env.obstacles[i-1].radius for i in range(1, self.num_obstacles + 1)]
            print(f"True Obstacle Radius: {[f'{x:.3f}' for x in actual_radii]}")
            print("--------------------------------------------------------------------")
        #print("====================================================================")

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
        self.goals["PolarGoToTarget"].desired_distance = self.obs["target_distance"].state_mean.item()