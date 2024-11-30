import numpy as np
import yaml
import torch
from typing import Dict

from components.aicon import AICON
from environment.gaze_fix_env import GazeFixEnv
from experiment_general.estimators import Obstacle_Rad_Estimator, Polar_Pos_Estimator_External_Vel, Polar_Pos_Estimator_Internal_Vel_Vel_Control, Pos_Estimator_External_Vel, Pos_Estimator_Internal_Vel, RecursiveEstimator, Robot_Vel_Estimator
from experiment_general.active_interconnections import Angle_Meas_AI, Cartesian_Polar_AI, Pos_Angle_AI, Radius_Pos_VisAngle_AI, Triangulation_AI, Vel_AI
from experiment_general.goals import AvoidObstacleGoal, GazeFixationGoal, GoToTargetGoal, PolarGoToTargetGoal, StopGoal

# =============================================================================================================================================================

class GeneralTestAICON(AICON):
    def __init__(self, num_obstacles=0, internal_vel=False, vel_control=False):
        self.internal_vel = internal_vel
        self.num_obstacles = num_obstacles
        self.vel_control = vel_control
        super().__init__()

    def define_env(self):
        config = 'config/env_config.yaml'
        with open(config) as file:
            env_config = yaml.load(file, Loader=yaml.FullLoader)
            env_config["num_obstacles"] = self.num_obstacles
            if self.vel_control:
                env_config["action_mode"] = 3
            else:
                env_config["action_mode"] = 1
        return GazeFixEnv(env_config)

    def define_estimators(self):
        REs: Dict[str, RecursiveEstimator] = {"RobotVel": Robot_Vel_Estimator(self.device)}
        if self.internal_vel:
            #REs["TargetPos"] = Pos_Estimator_Internal_Vel(device, "TargetPos")
            REs["PolarTargetPos"] = Polar_Pos_Estimator_Internal_Vel_Vel_Control(self.device, "PolarTargetPos")
            for i in range(1, self.num_obstacles + 1):
                REs[f"Obstacle{i}Pos"] = Pos_Estimator_Internal_Vel(self.device, f"Obstacle{i}Pos")
                REs[f"Obstacle{i}Rad"] = Obstacle_Rad_Estimator(self.device, f"Obstacle{i}Rad")
        else:
            #REs["TargetPos"] = Pos_Estimator_External_Vel(device, "TargetPos")
            REs["PolarTargetPos"] = Polar_Pos_Estimator_External_Vel(self.device, "PolarTargetPos")
            for i in range(1, self.num_obstacles + 1):
                REs[f"Obstacle{i}Pos"] = Pos_Estimator_External_Vel(self.device, f"Obstacle{i}Pos")
                REs[f"Obstacle{i}Rad"] = Obstacle_Rad_Estimator(self.device, f"Obstacle{i}Rad")
        return REs

    def define_active_interconnections(self):
        AIs = {
            "RobotVel": Vel_AI([self.REs["RobotVel"], self.obs["vel_frontal"], self.obs["vel_lateral"], self.obs["vel_rot"]], self.device),
            #"CartPolarPos": Cartesian_Polar_AI([REs["TargetPos"], REs["PolarTargetPos"]], device, estimate_vel=internal_vel),
        }
        if self.internal_vel:
            AIs["PolarAngle"] = Angle_Meas_AI([self.REs["PolarTargetPos"], self.obs["target_offset_angle"], self.obs["del_target_offset_angle"]], self.device, estimate_vel=self.internal_vel)
            AIs["PolarDistance"] = Triangulation_AI([self.REs["PolarTargetPos"], self.REs["RobotVel"]], self.device, estimate_vel=self.internal_vel)
            #AIs["TargetPos-Angle"] = Pos_Angle_AI([REs["TargetPos"], REs["RobotVel"], self.obs["target_offset_angle"]], device, estimate_vel=internal_vel)
        else:
            AIs["PolarAngle"] = Angle_Meas_AI([self.REs["PolarTargetPos"], self.obs["target_offset_angle"]], self.device, estimate_vel=self.internal_vel)
            AIs["PolarDistance"] = Triangulation_AI([self.REs["PolarTargetPos"], self.REs["RobotVel"], self.obs["del_target_offset_angle"]], self.device, estimate_vel=self.internal_vel)
            #AIs["TargetPos-Angle"] = Pos_Angle_AI([REs["TargetPos"], self.obs["target_offset_angle"]], device, estimate_vel=internal_vel)
        for i in range(1, self.num_obstacles + 1):
            AIs[f"Obstacle{i}Pos-Angle"] = Pos_Angle_AI([self.REs[f"Obstacle{i}Pos"], self.REs["RobotVel"], self.obs[f"obstacle{i}_offset_angle"]], self.device, object_name=f"Obstacle{i}", estimate_vel=self.internal_vel)
            AIs[f"Obstacle{i}Rad"] = Radius_Pos_VisAngle_AI([self.REs[f"Obstacle{i}Pos"], self.REs[f"Obstacle{i}Rad"], self.obs[f"obstacle{i}_visual_angle"]], self.device, obstacle_id=i)
        return AIs

    def define_goals(self):
        goals = {
            #"Go-To-Target" : GoToTargetGoal(device),
            "PolarGo-To-Target" : PolarGoToTargetGoal(self.device),
            "Stop"         : StopGoal(self.device),
            "GazeFixation" : GazeFixationGoal(self.device),
        }
        for i in range(1, self.num_obstacles + 1):
            goals[f"AvoidObstacle{i}"] = AvoidObstacleGoal(self.REs[f"Obstacle{i}Pos"])
        return goals

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

    def eval_step(self, action: torch.Tensor, new_step = False):
        # Use a copy of the state to avoid modifying the actual state
        self.update_observations()
        buffer_dict = {key: estimator.set_buffer_dict() for key, estimator in list(self.REs.items()) + list(self.obs.items())}

        # print("EVAL Action: ", action)
        # print("PRE Update: ", buffer_dict["PolarTargetPos"]["state_mean"])

        # ----------------------------- update vel -------------------------------------
        if self.vel_control:
            vel = action * torch.tensor([self.env.robot.max_vel, self.env.robot.max_vel, self.env.robot.max_vel_rot])
            buffer_dict["RobotVel"]["state_mean"] = vel
            buffer_dict["RobotVel"]["state_cov"] = torch.eye(3) * 1e-3
        else:
            timestep = torch.tensor([0.05], device=self.device)
            acc = action * torch.tensor([self.env.robot.max_acc, self.env.robot.max_acc, self.env.robot.max_acc_rot])
            u_robot_vel = torch.concat([acc, timestep])
            self.REs["RobotVel"].call_predict(u_robot_vel, buffer_dict)
            if new_step:
                self.REs["RobotVel"].call_update_with_specific_meas(self.AIs["RobotVel"], buffer_dict)

        # ----------------------------- update and predict -------------------------------------

        if (self.internal_vel and self.vel_control) or not self.internal_vel:
            u_pos = torch.concat([
                torch.atleast_1d(torch.tensor(self.env.timestep)),
                buffer_dict["RobotVel"]["state_mean"],
            ]).squeeze()
        else:
            u_pos = torch.concat([
                torch.atleast_1d(torch.tensor(self.env.timestep)),
                acc,
            ]).squeeze()

        if new_step:
            self.call_predicts(u_pos, buffer_dict)
            self.call_updates(buffer_dict)
        else:
            #self.call_updates(buffer_dict)
            #TODO: breaks everythang
            #self.REs["PolarTargetPos"].call_update_with_specific_meas(self.AIs["PolarDistance"], buffer_dict)
            self.call_predicts(u_pos, buffer_dict)
            #self.call_predicts(u_pos, buffer_dict)

        return buffer_dict

    def call_updates(self, buffer_dict):
        self.REs["PolarTargetPos"].call_update_with_specific_meas(self.AIs["PolarAngle"], buffer_dict)
        self.REs["PolarTargetPos"].call_update_with_specific_meas(self.AIs["PolarDistance"], buffer_dict)
        for i in range(1, self.num_obstacles + 1):
            self.REs[f"Obstacle{i}Pos"].call_update_with_specific_meas(self.AIs[f"Obstacle{i}Pos-Angle"], buffer_dict)
            self.REs[f"Obstacle{i}Rad"].call_update_with_specific_meas(self.AIs[f"Obstacle{i}Rad"], buffer_dict)
        #print("POST Measurement: ", buffer_dict["PolarTargetPos"]["state_mean"])#, "\n", buffer_dict["PolarTargetPos"]["state_cov"])

    def call_predicts(self, u_pos, buffer_dict):
        self.REs["PolarTargetPos"].call_predict(u_pos, buffer_dict)
        for i in range(1, self.num_obstacles + 1):
            self.REs[f"Obstacle{i}Pos"].call_predict(u_pos, buffer_dict)
            self.REs[f"Obstacle{i}Rad"].call_predict(u_pos, buffer_dict)
        #print("POST Prediction: ", buffer_dict["PolarTargetPos"]["state_mean"])#, "\n", buffer_dict["PolarTargetPos"]["state_cov"])

    def compute_action(self, gradients):
        if self.vel_control and self.internal_vel:
            action = self.last_action - 5e-2 * gradients["PolarGo-To-Target"]
            for i in range(self.num_obstacles):
                action -= 1e-2 * gradients[f"AvoidObstacle{i+1}"] / self.num_obstacles
        elif self.vel_control and not self.internal_vel:
            action = self.last_action - 5e-4 * gradients["PolarGo-To-Target"]
            for i in range(self.num_obstacles):
                action -= 1e-2 * gradients[f"AvoidObstacle{i+1}"] / self.num_obstacles
        elif not self.vel_control and self.internal_vel:
            action = self.last_action - 5e-2 * gradients["PolarGo-To-Target"]
            for i in range(self.num_obstacles):
                action -= 1e0 * gradients[f"AvoidObstacle{i+1}"] / self.num_obstacles
        elif not self.vel_control and not self.internal_vel:
            action = self.last_action - 5e-2 * gradients["PolarGo-To-Target"]
            for i in range(self.num_obstacles):
                action -= 1e-2 * gradients[f"AvoidObstacle{i+1}"] / self.num_obstacles

        # manual gaze fixation
        #action[2] = 0.05 * self.REs["PolarTargetPos"].state_mean[1] + 0.01 * self.REs["PolarTargetPos"].state_mean[3]
        return action

    def print_states(self, print_cov=False):
        """
        print filter and environment states for debugging
        """
        obs = self.env.get_observation()
        # self.print_state("RobotVel", print_cov=print_cov)
        # actual_vel = list(self.env.robot.vel)
        # actual_vel.append(self.env.robot.vel_rot)
        # print(f"True Robot Vel: {[f'{x:.3f}' for x in actual_vel]}")
        # print("--------------------------------------------------------------------")
        self.print_state("PolarTargetPos", print_cov=True)
        actual_pos = self.env.rotation_matrix(-self.env.robot.orientation) @ (self.env.target.pos - self.env.robot.pos)
        angle = np.arctan2(actual_pos[1], actual_pos[0])
        dist = np.linalg.norm(actual_pos)
        print(f"True Polar Target Position: {[f'{x:.3f}' for x in [dist, angle, obs['del_robot_target_distance'], obs['del_target_offset_angle']]]}")
        print("--------------------------------------------------------------------")
        if self.num_obstacles > 0:
            [self.print_state(f"Obstacle{i}Pos", print_cov=print_cov) for i in range(1, self.num_obstacles + 1)]
            actual_radii = [self.env.obstacles[i-1].radius for i in range(1, self.num_obstacles + 1)]
            print(f"True Obstacle Radius: {[f'{x:.3f}' for x in actual_radii]}")
            print("--------------------------------------------------------------------")
        print("====================================================================")