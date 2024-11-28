import numpy as np
import yaml
import torch
from typing import Dict

from components.aicon import AICON
from environment.gaze_fix_env import GazeFixEnv
from experiment_general.estimators import Obstacle_Rad_Estimator, Polar_Pos_Estimator_External_Vel, Polar_Pos_Estimator_Internal_Vel, Pos_Estimator_External_Vel, Pos_Estimator_Internal_Vel, RecursiveEstimator, Robot_Vel_Estimator
from experiment_general.active_interconnections import Angle_Meas_AI, Cartesian_Polar_AI, Pos_Angle_AI, Radius_Pos_VisAngle_AI, Triangulation_AI, Vel_AI
from experiment_general.goals import AvoidObstacleGoal, GazeFixationGoal, GoToTargetGoal, PolarGoToTargetGoal, StopGoal

# =============================================================================================================================================================

class GeneralTestAICON(AICON):
    def __init__(self, num_obstacles=0, internal_vel=False, vel_control=False):
        super().__init__()

        self.internal_vel = internal_vel
        self.num_obstacles = num_obstacles
        self.vel_control = vel_control

        config = 'config/env_config.yaml'
        with open(config) as file:
            env_config = yaml.load(file, Loader=yaml.FullLoader)
            env_config["num_obstacles"] = num_obstacles
            if vel_control:
                env_config["action_mode"] = 3
            else:
                env_config["action_mode"] = 1
        self.set_env(GazeFixEnv(env_config))

        REs: Dict[str, RecursiveEstimator] = {"RobotVel": Robot_Vel_Estimator(self.device)}
        if internal_vel:
            REs["TargetPos"] = Pos_Estimator_Internal_Vel(self.device, "TargetPos")
            REs["PolarTargetPos"] = Polar_Pos_Estimator_Internal_Vel(self.device, "PolarTargetPos")
            for i in range(1, num_obstacles + 1):
                REs[f"Obstacle{i}Pos"] = Pos_Estimator_Internal_Vel(self.device, f"Obstacle{i}Pos")
                REs[f"Obstacle{i}Rad"] = Obstacle_Rad_Estimator(self.device, f"Obstacle{i}Rad")
        else:
            REs["TargetPos"] = Pos_Estimator_External_Vel(self.device, "TargetPos")
            REs["PolarTargetPos"] = Polar_Pos_Estimator_External_Vel(self.device, "PolarTargetPos")
            for i in range(1, num_obstacles + 1):
                REs[f"Obstacle{i}Pos"] = Pos_Estimator_External_Vel(self.device, f"Obstacle{i}Pos")
                REs[f"Obstacle{i}Rad"] = Obstacle_Rad_Estimator(self.device, f"Obstacle{i}Rad")
        self.set_estimators(REs)

        AIs = {
            "RobotVel": Vel_AI([self.REs["RobotVel"], self.obs["vel_frontal"], self.obs["vel_lateral"], self.obs["vel_rot"]], self.device),
            "CartPolarPos": Cartesian_Polar_AI([REs["TargetPos"], REs["PolarTargetPos"]], self.device, estimate_vel=internal_vel),
        }
        if self.internal_vel:
            AIs["PolarAngle"] = Angle_Meas_AI([self.REs["PolarTargetPos"], self.obs["target_offset_angle"], self.obs["del_target_offset_angle"]], self.device, estimate_vel=internal_vel)
            AIs["PolarDistance"] = Triangulation_AI([self.REs["PolarTargetPos"], self.REs["RobotVel"]], self.device, estimate_vel=internal_vel)
            AIs["TargetPos-Angle"] = Pos_Angle_AI([REs["TargetPos"], REs["RobotVel"], self.obs["target_offset_angle"]], self.device, estimate_vel=internal_vel)
        else:
            AIs["PolarAngle"] = Angle_Meas_AI([self.REs["PolarTargetPos"], self.obs["target_offset_angle"]], self.device, estimate_vel=internal_vel)
            AIs["PolarDistance"] = Triangulation_AI([self.REs["PolarTargetPos"], self.REs["RobotVel"], self.obs["del_target_offset_angle"]], self.device, estimate_vel=internal_vel)
            AIs["TargetPos-Angle"] = Pos_Angle_AI([REs["TargetPos"], self.obs["target_offset_angle"]], self.device, estimate_vel=internal_vel)
        for i in range(1, num_obstacles + 1):
            AIs[f"Obstacle{i}Pos-Angle"] = Pos_Angle_AI([REs[f"Obstacle{i}Pos"], REs["RobotVel"], self.obs[f"obstacle{i}_offset_angle"]], self.device, object_name=f"Obstacle{i}", estimate_vel=internal_vel)
            AIs[f"Obstacle{i}Rad"] = Radius_Pos_VisAngle_AI([REs[f"Obstacle{i}Pos"], REs[f"Obstacle{i}Rad"], self.obs[f"obstacle{i}_visual_angle"]], self.device, obstacle_id=i)
        self.set_active_interconnections(AIs)

        goals = {
            "Go-To-Target" : GoToTargetGoal(self.device),
            "PolarGo-To-Target" : PolarGoToTargetGoal(self.device),
            "Stop"         : StopGoal(self.device),
            "GazeFixation" : GazeFixationGoal(self.device),
        }
        for i in range(1, num_obstacles + 1):
            goals[f"AvoidObstacle{i}"] = AvoidObstacleGoal(REs[f"Obstacle{i}Pos"])
        self.set_goals(goals)

    def reset(self):
        self.REs["RobotVel"].set_state(torch.tensor([0.1, 0.1, 0.1]), torch.eye(3)*1e1)
        self.REs["RobotVel"].set_static_motion_noise(torch.eye(3)*1e-1)

        if self.internal_vel:
            self.REs["TargetPos"].set_state(torch.tensor([20.0, 0.0, 0.0, 0.0, 0.0]), torch.eye(5)*1e3)
            self.REs["TargetPos"].set_static_motion_noise(torch.eye(5)*1e-1)
            self.REs["PolarTargetPos"].set_state(torch.tensor([20.0, 0.0, 0.0, 0.0]), torch.eye(4)*1e3)
            self.REs["PolarTargetPos"].set_static_motion_noise(torch.eye(4)*1e-1)
        else:
            self.REs["TargetPos"].set_state(torch.tensor([20.0, 0.0]), torch.eye(2)*1e3)
            self.REs["TargetPos"].set_static_motion_noise(torch.eye(2)*1e-1)
            self.REs["PolarTargetPos"].set_state(torch.tensor([20.0, 0.0]), torch.eye(2)*1e3)
            self.REs["PolarTargetPos"].set_static_motion_noise(torch.eye(2)*1e-1)

        for i in range(1, self.num_obstacles + 1):
            if self.internal_vel:
                self.REs[f"Obstacle{i}Pos"].set_state(torch.tensor([10.0, 0.0, 0.0, 0.0, 0.0]), torch.eye(5)*1e3)
                self.REs[f"Obstacle{i}Pos"].set_static_motion_noise(torch.eye(5)*1e-1)
            else:
                self.REs[f"Obstacle{i}Pos"].set_state(torch.tensor([10.0, 0.0]), torch.eye(2)*1e3)
                self.REs[f"Obstacle{i}Pos"].set_static_motion_noise(torch.eye(2)*1e-1)
            self.REs[f"Obstacle{i}Rad"].set_state(torch.tensor([1.0]), torch.eye(1)*1e3)
            self.REs[f"Obstacle{i}Rad"].set_static_motion_noise(torch.eye(1)*1e-3)

    def render(self):
        # estimator_means = {key: np.array(self.REs[key].state_mean.cpu()) for key in ["TargetPos"] + [f"Obstacle{i}Pos" for i in range(1, self.num_obstacles + 1)] if key in self.REs.keys()}
        # estimator_covs = {key: np.array(self.REs[key].state_cov.cpu()) for key in ["TargetPos"] + [f"Obstacle{i}Pos" for i in range(1, self.num_obstacles + 1)] if key in self.REs.keys()}
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

    def eval_step(self, action, new_step = False):
        # Use a copy of the state to avoid modifying the actual state
        self.update_observations()
        buffer_dict = {key: estimator.set_buffer_dict() for key, estimator in list(self.REs.items()) + list(self.obs.items())}

        # ----------------- prediction updates -----------------------

        #print("EVAL Action: ", action)

        if self.vel_control:
            buffer_dict["RobotVel"]["state_mean"] = torch.concat([action[:2]*self.env.robot.max_vel, torch.tensor([action[2]*self.env.robot.max_vel_rot])])
            buffer_dict["RobotVel"]["state_cov"] = torch.eye(3) * 1e-3
        else:
            u_robot_vel = torch.concat([action[:2]*self.env.robot.max_acc, torch.tensor([action[2]*self.env.robot.max_acc_rot]), torch.tensor([self.env.timestep])])
            self.REs["RobotVel"].call_predict(u_robot_vel, buffer_dict)

        u_pos = torch.concat([
            buffer_dict["RobotVel"]["state_mean"],
            torch.atleast_1d(torch.tensor(self.env.timestep)),
        ]).squeeze() if not self.internal_vel else torch.tensor(self.env.timestep)

        # ----------------- measurement updates -----------------------

        #print("PRE Measurement: ", buffer_dict["PolarTargetPos"]["state_mean"])

        self.REs["TargetPos"].call_update_with_specific_meas(self.AIs["TargetPos-Angle"], buffer_dict)

        self.REs["PolarTargetPos"].call_update_with_specific_meas(self.AIs["PolarAngle"], buffer_dict)
        self.REs["PolarTargetPos"].call_update_with_specific_meas(self.AIs["PolarDistance"], buffer_dict)

        # TODO: updating each other appears stupid
        #self.REs["PolarTargetPos"].call_update_with_specific_meas(self.AIs["CartPolarPos"], buffer_dict)
        self.REs["TargetPos"].call_update_with_specific_meas(self.AIs["CartPolarPos"], buffer_dict)
        
        for i in range(1, self.num_obstacles + 1):
            self.REs[f"Obstacle{i}Pos"].call_update_with_specific_meas(self.AIs[f"Obstacle{i}Pos-Angle"], buffer_dict)
            self.REs[f"Obstacle{i}Rad"].call_update_with_specific_meas(self.AIs[f"Obstacle{i}Rad"], buffer_dict)

        if not self.vel_control:
            self.REs["RobotVel"].call_update_with_specific_meas(self.AIs["RobotVel"], buffer_dict)

        #print("POST Measurement: ", buffer_dict["PolarTargetPos"]["state_mean"])

        self.REs["TargetPos"].call_predict(u_pos, buffer_dict)
        for i in range(1, self.num_obstacles + 1):
            self.REs[f"Obstacle{i}Pos"].call_predict(u_pos, buffer_dict)
            self.REs[f"Obstacle{i}Rad"].call_predict(u_pos, buffer_dict)
        self.REs["PolarTargetPos"].call_predict(u_pos, buffer_dict)

        #print("POST Prediction: ", buffer_dict["PolarTargetPos"]["state_mean"])

        return buffer_dict

    def compute_action(self, gradients):
        if self.vel_control:
            action = self.last_action - 1e-3 * gradients["PolarGo-To-Target"]# - gradients["GazeFixation"]
            for i in range(self.num_obstacles):
                action -= 1e-3 * gradients[f"AvoidObstacle{i+4}"] / self.num_obstacles
        else:
            action = self.last_action - 10.0 * gradients["PolarGo-To-Target"]# - gradients["GazeFixation"]
            for i in range(self.num_obstacles):
                action -= gradients[f"AvoidObstacle{i+4}"] / self.num_obstacles
        # manual gaze fixation
        #action[2] = 0.05 * self.REs["PolarTargetPos"].state_mean[1] + 0.01 * self.REs["PolarTargetPos"].state_mean[3]
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
        print("--------------------------------------------------------------------")
        self.print_state("TargetPos", print_cov=print_cov)
        actual_pos = list(self.env.rotation_matrix(-self.env.robot.orientation) @ (self.env.target.pos - self.env.robot.pos))
        for v in actual_vel: actual_pos.append(-v)
        print(f"True Target Position: {[f'{x:.3f}' for x in actual_pos]}")
        print("--------------------------------------------------------------------")
        if self.num_obstacles > 0:
            [self.print_state(f"Obstacle{i}Pos", print_cov=print_cov) for i in range(1, self.num_obstacles + 1)]
            actual_radii = [self.env.obstacles[i-1].radius for i in range(1, self.num_obstacles + 1)]
            print(f"True Obstacle Radius: {[f'{x:.3f}' for x in actual_radii]}")
            print("--------------------------------------------------------------------")
        self.print_state("PolarTargetPos", print_cov=print_cov)
        actual_pos = self.env.rotation_matrix(-self.env.robot.orientation) @ (self.env.target.pos - self.env.robot.pos)
        angle = np.arctan2(actual_pos[1], actual_pos[0])
        dist = np.linalg.norm(actual_pos)
        print(f"True Polar Target Position: {[f'{x:.3f}' for x in [dist, angle, obs['del_robot_target_distance'], obs['del_target_offset_angle']]]}")
        print("====================================================================")

# =============================================================================================================================================================

class SimpleVelTestAICON(AICON):
    def __init__(self):
        super().__init__()

        config = 'config/env_config.yaml'
        with open(config) as file:
            env_config = yaml.load(file, Loader=yaml.FullLoader)
            env_config["action_mode"] = 3
        self.set_env(GazeFixEnv(env_config))

        estimators = {
            "RobotVel": Robot_Vel_Estimator(device=self.device),
            "PolarTargetPos": Polar_Pos_Estimator_External_Vel(device=self.device, id='PolarTargetPos'),
        }
        self.set_estimators(estimators)

        active_interconnections = {
            "PolarAngle": Angle_Meas_AI([self.REs["PolarTargetPos"], self.obs["target_offset_angle"]], self.device, estimate_vel=False),
            "PolarDistance": Triangulation_AI([self.REs["PolarTargetPos"], self.REs["RobotVel"], self.obs["del_target_offset_angle"]], self.device, estimate_vel=False),
        }
        self.set_active_interconnections(active_interconnections)

        self.set_goals({
            "GazeFixation": GazeFixationGoal(device=self.device),
            "PolarGoToTarget": PolarGoToTargetGoal(device=self.device)
        })

        self.reset()

    def reset(self):
        self.REs["RobotVel"].set_state(torch.tensor([0.0, 0.0, 0.0]), torch.eye(3) * 0.01)
        self.REs["RobotVel"].set_static_motion_noise(torch.eye(3) * 0.01)
        self.REs["PolarTargetPos"].set_state(torch.tensor([10.0, 0.0]), torch.eye(2) * 1e3)
        self.REs["PolarTargetPos"].set_static_motion_noise(torch.eye(2) * 0.01)

    def eval_predict(self, action, buffer_dict):
        vel = action * torch.tensor([self.env.robot.max_vel, self.env.robot.max_vel, self.env.robot.max_vel_rot])
        timestep = torch.tensor([0.05], device=self.device)
        self.REs["PolarTargetPos"].call_predict(torch.concat([vel, timestep]), buffer_dict)
        buffer_dict["RobotVel"]["state_mean"] = vel
        buffer_dict["RobotVel"]["state_cov"] = torch.eye(3) * 1e-3
        return buffer_dict

    def eval_step(self, action, new_step = False):
        self.update_observations()
        buffer_dict = {key: estimator.set_buffer_dict() for key, estimator in list(self.REs.items()) + list(self.obs.items())}

        if not new_step:
            return self.eval_predict(action, buffer_dict)

        vel = action * torch.tensor([self.env.robot.max_vel, self.env.robot.max_vel, self.env.robot.max_vel_rot])
        timestep = torch.tensor([0.05], device=self.device)

        if new_step:
            print(
                f"------------------- Pre Predict: -------------------\n",
                f"Polar Pos: {buffer_dict['PolarTargetPos']['state_mean']}",
            )

        self.REs["PolarTargetPos"].call_predict(torch.concat([vel, timestep]), buffer_dict)

        buffer_dict["RobotVel"]["state_mean"] = vel
        buffer_dict["RobotVel"]["state_cov"] = torch.eye(3) * 1e-3

        if new_step:
            print(
                f"------------------- Post Predict: -------------------\n",
                f"Polar Pos: {buffer_dict['PolarTargetPos']['state_mean']}",
            )

        self.REs["PolarTargetPos"].call_update_with_specific_meas(self.AIs["PolarAngle"], buffer_dict)
        self.REs["PolarTargetPos"].call_update_with_specific_meas(self.AIs["PolarDistance"], buffer_dict)

        if new_step:
            print(
                f"------------------ Post Measurement: ------------------\n",
                f"Polar Pos: {buffer_dict['PolarTargetPos']['state_mean']}",
            )

        return buffer_dict
    
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
        """
        CAN be implemented by user. Computes the action based on the gradients
        """
        return self.last_action - 1.0 * gradients["PolarGoToTarget"]
    
    def print_states(self):
        obs = self.env._get_observation()
        print("==========================================")
        self.print_state("PolarTargetPos", False)
        print(f"True PolarTargetPos: [{obs['robot_target_distance']:.3f}, {obs['target_offset_angle']:.3f}]")
        print("==========================================")