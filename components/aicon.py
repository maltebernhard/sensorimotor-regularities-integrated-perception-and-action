import numpy as np
import torch
from torch.func import jacrev, functional_call
from abc import ABC, abstractmethod
from typing import Dict, List

import yaml
#from components.active_interconnection import ActiveInterconnection
from components.active_interconnection import ActiveInterconnection, Pos_Angle_AI, Pos_Angle_Vel_AI, Radius_Pos_VisAngle_AI, Vel_AI
from components.estimator import Obstacle_Rad_Estimator, Polar_Pos_Estimator_Internal_Vel, Pos_Estimator_External_Vel, Pos_Estimator_Internal_Vel, RecursiveEstimator, Robot_Vel_Estimator, State
from components.goal import AvoidObstacleGoal, GoToTargetGoal, Goal, StopGoal
from components.measurement_model import ImplicitMeasurementModel, Polar_Pos_Vel_MM, Pos_Vel_MM, Pos_MM, Radius_MM, Robot_Vel_MM
from environment.gaze_fix_env import GazeFixEnv

# ========================================================================================

class AICON(ABC):
    def __init__(self):
        device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        dtype = torch.float64
        torch.set_default_dtype(dtype)
        self.device = device
        self.dtype = dtype
        self.env: GazeFixEnv = None
        self.REs: Dict[str, RecursiveEstimator] = None
        self.AIs: Dict[str, ImplicitMeasurementModel] = None
        self.goals: Dict[str, Goal] = None

    def set_env(self, env):
        self.env = env
        self.obs: Dict[str, State] = {}
        observations: dict = self.env.get_observation()
        for key, value in observations.items():
            self.obs[key] = State(key, 1, self.device)
            self.obs[key].set_state(torch.tensor([value], device=self.device), torch.zeros((1,1), device=self.device))

    def set_estimators(self, REs: Dict[str, RecursiveEstimator]):
        self.REs = REs

    def set_active_interconnections(self, AIs: Dict[str, ActiveInterconnection]):
        self.AIs = AIs

    def set_goals(self, goals: Dict[str, Goal]):
        self.goals = goals

    def update_observations(self):
        observations: dict = self.env.get_observation_unnormalized()
        for key, value in observations.items():
            # TODO: consider measurement noise (possibly on env level)
            self.obs[key].set_state(torch.tensor([value], device=self.device, dtype=self.dtype), torch.zeros((1,1), device=self.device, dtype=self.dtype))

    def step(self, action):
        gain = 0.025 / self.env.timestep**2
        env_action = action * gain
        if env_action[:2].norm() > 1.0:
            env_action[:2] = env_action[:2] / env_action[:2].norm()
        if env_action[2] > 1.0:
            env_action[2] = 1.0
        #print(f"Action: {env_action}")
        self.env.step(np.array(env_action.cpu()))
        buffers = self.eval_step(env_action)
        for key, buffer_dict in buffers.items():
            self.REs[key].load_state_dict(buffer_dict) if key in self.REs.keys() else self.obs[key].load_state_dict(buffer_dict)

    def _eval_goal_with_aux(self, action, goal):
        buffer_dict = self.eval_step(action)
        loss = goal.loss_function_from_buffer(buffer_dict)
        return loss, loss

    def compute_goal_action_gradient(self, goal):
        action = torch.tensor([0.0, 0.0, 0.0], device=self.device)
        jacobian, step_eval = jacrev(
            self._eval_goal_with_aux,
            argnums=0, # x and meas_dict (all measurements within the dict!)
            has_aux=True)(action, goal)
        return jacobian

    def compute_action_gradients(self):
        gradients: List[torch.Tensor] = []
        for goal in self.goals.values():
            gradients.append(self.compute_goal_action_gradient(goal))
        print("Action gradients:")
        for i, gradient in enumerate(gradients):
            print(f"{list(self.goals.keys())[i]}: {[f'{x:.3f}' for x in gradient.tolist()]}")
        return gradients

    def get_steepest_gradient(self, gradients):
        steepest_gradient = gradients[0]
        for gradient in gradients:
            if gradient.norm() > steepest_gradient.norm():
                steepest_gradient = gradient
        return steepest_gradient

    def run(self, timesteps, env_seed, render = True, prints = 0, step_by_step = True, record_video = False):
        """
        Runs AICON on the environment for a given number of timesteps
        Args:
            timesteps:    number of timesteps to run
            env_seed:     seed for the environment
            render:       render the environment
            prints:       print the states every "prints" timesteps
            step_by_step: wait for user input after each step
        """
        assert self.env is not None, "Environment not set"
        assert self.REs is not None, "Estimators not set"
        assert self.AIs is not None, "Active Interconnections not set"
        assert self.goals is not None, "Goals not set"
        self.reset()
        self.env.reset(seed=env_seed, video_path="test_vid.mp4")
        if render:
            self.render()
        input("Press Enter to continue...")
        for step in range(timesteps):
            action_gradients = self.compute_action_gradients()
            action = self.compute_action(action_gradients)
            self.step(action)
            if render:
                self.render()
            step += 1
            if prints > 0 and step % prints == 0:
                print(f"============================ Step {step} ================================")
                self.print_states()

            if step_by_step:
                input("Press Enter to continue...")
        self.env.reset()

    @abstractmethod
    def reset(self) -> None:
        """
        MUST be implemented by user. Sets initial states of all estimators.
        """
        raise NotImplementedError

    @abstractmethod
    def eval_step(self, action):
        """
        MUST be implemented by user. Evaluates one step of the environment and returns the buffer_dict
        """
        raise NotImplementedError

    def compute_action(self, gradients):
        """
        CAN be implemented by user. Computes the action based on the gradients
        """
        return - self.get_steepest_gradient(gradients)

    def print_states(self):
        """
        CAN be implemented by user, only necessary if "prints" option is used in run()
        """
        raise NotImplementedError

    def render(self, estimator_means: List[torch.Tensor], estimator_covs: List[torch.Tensor], playback_speed=1.0):
        """
        CAN be implemented by user, overwrite to include custom estimator means and covariances into rendering
        """
        return self.env.render()

# ========================================================================================================

class MinimalAICON(AICON):
    def __init__(self, num_obstacles=0, internal_vel=False):
        super().__init__()

        self.internal_vel = internal_vel
        self.num_obstacles = num_obstacles

        assert num_obstacles in [0, 1, 2], "Number of obstacles must be 0, 1 or 2"
        config = {
            0: 'config/env_config_zero_obst.yaml',
            1: 'config/env_config_one_obst.yaml',
            2: 'config/env_config_two_obst.yaml'
        }
        with open(config[num_obstacles]) as file:
            env_config = yaml.load(file, Loader=yaml.FullLoader)
        self.set_env(GazeFixEnv(env_config))

        REs: Dict[str, RecursiveEstimator] = {"RobotVel": Robot_Vel_Estimator(self.device)}
        if internal_vel:
            REs["TargetPos"] = Pos_Estimator_Internal_Vel(self.device, "TargetPos")
            if num_obstacles > 0:
                REs["Obstacle1Pos"] = Pos_Estimator_Internal_Vel(self.device, "Obstacle1Pos")
                REs["Obstacle1Rad"] = Obstacle_Rad_Estimator(self.device, "Obstacle1Rad")
            if num_obstacles > 1:
                REs["Obstacle2Pos"] = Pos_Estimator_Internal_Vel(self.device, "Obstacle2Pos")
                REs["Obstacle2Rad"] = Obstacle_Rad_Estimator(self.device, "Obstacle2Rad")
        else:
            REs["TargetPos"] = Pos_Estimator_External_Vel(self.device, "TargetPos")
            if num_obstacles > 0:
                REs["Obstacle1Pos"] = Pos_Estimator_External_Vel(self.device, "Obstacle1Pos")
                REs["Obstacle1Rad"] = Obstacle_Rad_Estimator(self.device, "Obstacle1Rad")
            if num_obstacles > 1:
                REs["Obstacle2Pos"] = Pos_Estimator_External_Vel(self.device, "Obstacle2Pos")
                REs["Obstacle2Rad"] = Obstacle_Rad_Estimator(self.device, "Obstacle2Rad")
        REs["PolarTargetPos"] = Polar_Pos_Estimator_Internal_Vel(self.device, "PolarTargetPos")
        self.set_estimators(REs)

        AIs = {"RobotVel": Vel_AI([self.REs["RobotVel"], self.obs["vel_frontal"], self.obs["vel_lateral"], self.obs["vel_rot"]], self.device)}
        if internal_vel:
            AIs["TargetPos-Angle"] = Pos_Angle_Vel_AI([REs["TargetPos"], REs["RobotVel"], self.obs["target_offset_angle"]], "Target", self.device)
            if num_obstacles > 0:
                AIs["Obstacle1Pos"] = Pos_Angle_Vel_AI([REs["Obstacle1Pos"], REs["RobotVel"], self.obs["obstacle1_offset_angle"]], "Obstacle1", self.device)
                AIs["Obstacle1Rad"] = Radius_Pos_VisAngle_AI([REs["Obstacle1Pos"], REs["Obstacle1Rad"], self.obs["obstacle1_visual_angle"]], 1, self.device)
            if num_obstacles > 1:
                AIs["Obstacle2Pos"] = Pos_Angle_Vel_AI([REs["Obstacle2Pos"], REs["RobotVel"], self.obs["obstacle2_offset_angle"]], "Obstacle2", self.device)
                AIs["Obstacle2Rad"] = Radius_Pos_VisAngle_AI([REs["Obstacle2Pos"], REs["Obstacle2Rad"], self.obs["obstacle2_visual_angle"]], 2, self.device)
        else:
            AIs["TargetPos-Angle"] = Pos_Angle_AI([REs["TargetPos", self.obs["target_offset_angle"]]], "Target", self.device)
            if num_obstacles > 0:
                AIs["Obstacle1Pos"] = Pos_Angle_AI([REs["Obstacle1Pos"], self.obs["obstacle1_offset_angle"]], "Obstacle1", self.device)
                AIs["Obstacle1Rad"] = Radius_Pos_VisAngle_AI([REs["Obstacle1Pos"], REs["Obstacle1Rad"], self.obs["obstacle1_visual_angle"]], 1, self.device)
            if num_obstacles > 1:
                AIs["Obstacle2Pos"] = Pos_Angle_AI([REs["Obstacle2Pos"], self.obs["obstacle2_offset_angle"]], "Obstacle2", self.device)
                AIs["Obstacle2Rad"] = Radius_Pos_VisAngle_AI([REs["Obstacle2Pos"], REs["Obstacle2Rad"], self.obs["obstacle2_visual_angle"]], 2, self.device)
        AIs["PolarTargetPos"] = Polar_Pos_Vel_MM(self.device)
        self.set_active_interconnections(AIs)

        goals = {
            "Go-To-Target"     : GoToTargetGoal(REs["TargetPos"]),
            "Stop"             : StopGoal(REs["RobotVel"]),
        }
        if num_obstacles > 0:
            goals["AvoidObstacle1"] = AvoidObstacleGoal(REs["Obstacle1Pos"])
        if num_obstacles > 1:
            goals["AvoidObstacle2"] = AvoidObstacleGoal(REs["Obstacle2Pos"])
        self.set_goals(goals)

    def reset(self):
        self.REs["RobotVel"].set_state(torch.tensor([0.0, 0.0, 0.0], device=self.device), torch.eye(3, device=self.device)*0.01)
        self.REs["RobotVel"].set_static_motion_noise(torch.eye(3, device=self.device)*1e-1)

        if self.internal_vel:
            self.REs["TargetPos"].set_state(torch.tensor([20.0, 0.0, 0.0, 0.0, 0.0], device=self.device), torch.eye(5, device=self.device)*1e3)
            self.REs["TargetPos"].set_static_motion_noise(torch.eye(5, device=self.device)*1e-1)
        else:
            self.REs["TargetPos"].set_state(torch.tensor([20.0, 0.0], device=self.device), torch.eye(2, device=self.device)*1e3)
            self.REs["TargetPos"].set_static_motion_noise(torch.eye(2, device=self.device)*1e-1)

        if self.num_obstacles > 0:
            if self.internal_vel:
                self.REs["Obstacle1Pos"].set_state(torch.tensor([10.0, 0.0, 0.0, 0.0, 0.0], device=self.device), torch.eye(5, device=self.device)*1e3)
                self.REs["Obstacle1Pos"].set_static_motion_noise(torch.eye(5, device=self.device)*1e-1)
            else:
                self.REs["Obstacle1Pos"].set_state(torch.tensor([10.0, 0.0], device=self.device), torch.eye(2, device=self.device)*1e3)
                self.REs["Obstacle1Pos"].set_static_motion_noise(torch.eye(2, device=self.device)*1e-1)
            self.REs["Obstacle1Rad"].set_state(torch.tensor([1.0], device=self.device), torch.eye(1, device=self.device)*1e3)
            self.REs["Obstacle1Rad"].set_static_motion_noise(torch.eye(1, device=self.device)*1e-3)

        if self.num_obstacles > 1:
            if self.internal_vel:
                self.REs["Obstacle2Pos"].set_state(torch.tensor([10.0, 0.0, 0.0, 0.0, 0.0], device=self.device), torch.eye(5, device=self.device)*1e3)
                self.REs["Obstacle2Pos"].set_static_motion_noise(torch.eye(5, device=self.device)*1e-1)
            else:
                self.REs["Obstacle2Pos"].set_state(torch.tensor([10.0, 0.0], device=self.device), torch.eye(2, device=self.device)*1e3)
                self.REs["Obstacle2Pos"].set_static_motion_noise(torch.eye(2, device=self.device)*1e-1)
            self.REs["Obstacle2Rad"].set_state(torch.tensor([1.0], device=self.device), torch.eye(1, device=self.device)*1e3)
            self.REs["Obstacle2Rad"].set_static_motion_noise(torch.eye(1, device=self.device)*1e-3)

        self.REs["PolarTargetPos"].set_state(torch.tensor([0.0, 20.0, 0.0, 0.0], device=self.device), torch.eye(4, device=self.device)*1e3)
        self.REs["PolarTargetPos"].set_static_motion_noise(torch.eye(4, device=self.device)*1e-1)

    def render(self):
        estimator_means = {key: np.array(self.REs[key].state_mean.cpu()) for key in ["TargetPos", "Obstacle1Pos", "Obstacle2Pos"] if key in self.REs.keys()}
        estimator_covs = {key: np.array(self.REs[key].state_cov.cpu()) for key in ["TargetPos", "Obstacle1Pos", "Obstacle2Pos"] if key in self.REs.keys()}
        return self.env.render(1.0, estimator_means, estimator_covs)

    def eval_step(self, action):
        # Use a copy of the state to avoid modifying the actual state
        self.update_observations()
        buffer_dict = {key: estimator.set_buffer_dict() for key, estimator in list(self.REs.items()) + list(self.obs.items())}

        # ----------------- prediction updates -----------------------

        u_robot_vel = torch.concat([action[:2]*self.env.robot.max_acc, torch.tensor([action[2]*self.env.robot.max_acc_rot], device=self.device), torch.tensor([self.env.timestep], device=self.device)])
        self.REs["RobotVel"].call_predict(u_robot_vel, buffer_dict)

        u_pos = torch.stack([
            buffer_dict["RobotVel"]["state_mean"][0],
            buffer_dict["RobotVel"]["state_mean"][1],
            buffer_dict["RobotVel"]["state_mean"][2],
            torch.tensor(self.env.timestep, dtype=self.dtype, device=self.device),
        ]).squeeze() if not self.internal_vel else torch.tensor(self.env.timestep, dtype=self.dtype, device=self.device)

        self.REs["TargetPos"].call_predict(u_pos, buffer_dict)
        if self.num_obstacles > 0:
            self.REs["Obstacle1Pos"].call_predict(u_pos, buffer_dict)
            self.REs["Obstacle1Rad"].call_predict(u_pos, buffer_dict)
        if self.num_obstacles > 1:
            self.REs["Obstacle2Pos"].call_predict(u_pos, buffer_dict)
            self.REs["Obstacle2Rad"].call_predict(u_pos, buffer_dict)
        self.REs["PolarTargetPos"].call_predict(u_pos, buffer_dict)

        # ----------------- measurement updates -----------------------

        self.REs["RobotVel"].call_update_with_specific_meas(self.AIs["RobotVel"], buffer_dict)
        self.REs["TargetPos"].call_update_with_specific_meas(self.AIs["TargetPos-Angle"], buffer_dict)
        if self.num_obstacles > 0:
            self.REs["Obstacle1Pos"].call_update_with_specific_meas(self.AIs["Obstacle1Pos"], buffer_dict)
            self.REs["Obstacle1Rad"].call_update_with_specific_meas(self.AIs["Obstacle1Rad"], buffer_dict)
        if self.num_obstacles > 1:
            self.REs["Obstacle2Pos"].call_update_with_specific_meas(self.AIs["Obstacle2Pos"], buffer_dict)
            self.REs["Obstacle2Rad"].call_update_with_specific_meas(self.AIs["Obstacle2Rad"], buffer_dict)
        return buffer_dict

    def compute_action(self, gradients):
        action = - (gradients[0])
        if self.num_obstacles > 0:
            action -= gradients[2]
        if self.num_obstacles > 1:
            action -= gradients[3]
        return action

    def print_states(self):
        """
        print filter and environment states for debugging
        """
        print(f"Robot Vel Estimate: {[f'{x:.3f}' for x in self.REs['RobotVel'].state_mean.tolist()]}")
        actual_vel = list(self.env.robot.vel)
        actual_vel.append(self.env.robot.vel_rot)
        print(f"True Robot Vel:     {[f'{x:.3f}' for x in actual_vel]}")
        print("--------------------------------------------------------------------")
        print(f"Target Position Estimate: {[f'{x:.3f}' for x in self.REs['TargetPos'].state_mean.tolist()]}")
        actual_pos = list(self.env.rotation_matrix(-self.env.robot.orientation) @ (self.env.target.pos - self.env.robot.pos))
        for v in actual_vel: actual_pos.append(-v)
        print(f"True Target Position:      {[f'{x:.3f}' for x in actual_pos]}")
        print("--------------------------------------------------------------------")
        if self.num_obstacles > 1:
            print(f"Obstacle Radius Estimates: {[f'{x:.3f}' for x in self.REs['Obstacle1Rad'].state_mean.tolist()]}, {[f'{x:.3f}' for x in self.REs['Obstacle2Rad'].state_mean.tolist()]}")
            actual_rads = [self.env.obstacles[0].radius, self.env.obstacles[1].radius]
            print(f"True Obstacle Radius:      {[f'{x:.3f}' for x in actual_rads]}")
            print("--------------------------------------------------------------------")
        print(f"Polar Target Position Estimates: {[f'{x:.3f}' for x in self.REs['PolarTargetPos'].state_mean.tolist()]}")
        actual_pos = self.env.rotation_matrix(-self.env.robot.orientation) @ (self.env.target.pos - self.env.robot.pos)
        angle = np.arctan2(actual_pos[1], actual_pos[0])
        dist = np.linalg.norm(actual_pos)
        print(f"True Polar Target Position:      {[angle, dist]}")

# ================================================================================================================================