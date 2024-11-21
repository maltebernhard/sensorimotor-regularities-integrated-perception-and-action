import numpy as np
import torch
from torch.func import jacrev, functional_call
from abc import ABC, abstractmethod
from typing import Dict, List

import yaml
#from components.active_interconnection import ActiveInterconnection
from components.active_interconnection import ActiveInterconnection, Polar_Angle_AI, Polar_Angle_Vel_AI, Polar_Angle_Vel_NonCart_AI, Polar_Distance_AI, Polar_Distance_Vel_AI, Pos_Angle_AI, Pos_Angle_Vel_AI, Radius_Pos_VisAngle_AI, Vel_AI
from components.estimator import Obstacle_Rad_Estimator, Polar_Pos_Estimator_External_Vel, Polar_Pos_Estimator_Internal_Vel, Pos_Estimator_External_Vel, Pos_Estimator_Internal_Vel, RecursiveEstimator, Robot_Vel_Estimator, State
from components.goal import AvoidObstacleGoal, GazeFixationGoal, GoToTargetGoal, Goal, StopGoal
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
        observations: dict = self.env.get_observation()
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
        print(f"Action: {env_action}")
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

    def print_vector(self, vector: torch.Tensor, name = None, trail = "", use_scientific = False):
        if not use_scientific:
            use_scientific = any(abs(x.item()) < 1e-3 or abs(x.item()) > 1e3 for x in vector)
        if name is not None:
            print(f"{name}: [", end="")
        else:
            print("[", end="")
        for i, x in enumerate(vector):
            if i > 0:
                print(" ", end="")
            if 0 <= x.item():
                print(" ", end="")
            if use_scientific:
                print(f"{x.item():.3e}", end="")
            else:
                print(f"{x.item():.3f}", end="")
            if i < vector.shape[0] - 1:
                print(", ", end="")
        else:
            print("]" + trail)

    def print_matrix(self, matrix: torch.Tensor, name = None):
        use_scientific = any(abs(x.item()) < 1e-3 or abs(x.item()) > 1e3 for x in matrix.flatten())
        for i in range(matrix.shape[0]):
            if i == 0:
                if name is not None:
                    print(f"{name}: [", end="")
                else:
                    print("[", end="")
            else:
                if name is not None:
                    print(" " * (len(name) + 3), end="")
                else:
                    print(" ", end="")
            self.print_vector(matrix[i], trail="," if i < matrix.shape[0] - 1 else "]", use_scientific=use_scientific)

    def print_state(self, id, print_cov: bool = False):
        self.print_vector(self.REs[id].state_mean, id + " Mean" if print_cov else id)
        if print_cov:
            self.print_matrix(self.REs[id].state_cov, f"{id} Cov")

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

        config = 'config/env_config.yaml'
        with open(config) as file:
            env_config = yaml.load(file, Loader=yaml.FullLoader)
            env_config["num_obstacles"] = num_obstacles
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

        AIs = {"RobotVel": Vel_AI([self.REs["RobotVel"], self.obs["vel_frontal"], self.obs["vel_lateral"], self.obs["vel_rot"]], self.device)}
        if internal_vel:
            AIs["TargetPos-Angle"] = Pos_Angle_Vel_AI([REs["TargetPos"], REs["RobotVel"], self.obs["target_offset_angle"]], "Target", self.device)
            AIs["PolarDistance"] = Polar_Distance_Vel_AI([REs["PolarTargetPos"], self.REs["TargetPos"]], "Target", self.device)
            AIs["PolarAngle"] = Polar_Angle_Vel_AI([REs["PolarTargetPos"], self.obs["target_offset_angle"], self.obs["del_target_offset_angle"]], "Target", self.device)
            for i in range(1, num_obstacles + 1):
                AIs[f"Obstacle{i}Pos-Angle"] = Pos_Angle_Vel_AI([REs[f"Obstacle{i}Pos"], REs["RobotVel"], self.obs[f"obstacle{i}_offset_angle"]], f"Obstacle{i}", self.device)
                AIs[f"Obstacle{i}Rad"] = Radius_Pos_VisAngle_AI([REs[f"Obstacle{i}Pos"], REs[f"Obstacle{i}Rad"], self.obs[f"obstacle{i}_visual_angle"]], i, self.device)
        else:
            AIs["TargetPos-Angle"] = Pos_Angle_AI([REs["TargetPos"], self.obs["target_offset_angle"]], "Target", self.device)
            AIs["PolarDistance"] = Polar_Distance_AI([REs["PolarTargetPos"], self.REs["TargetPos"]], "Target", self.device)
            AIs["PolarAngle"] = Polar_Angle_AI([REs["PolarTargetPos"], self.obs["target_offset_angle"]], "Target", self.device)
            for i in range(1, num_obstacles + 1):
                AIs[f"Obstacle{i}Pos-Angle"] = Pos_Angle_AI([REs[f"Obstacle{i}Pos"], self.obs[f"obstacle{i}_offset_angle"]], f"Obstacle{i}", self.device)
                AIs[f"Obstacle{i}Rad"] = Radius_Pos_VisAngle_AI([REs[f"Obstacle{i}Pos"], REs[f"Obstacle{i}Rad"], self.obs[f"obstacle{i}_visual_angle"]], i, self.device)
        
        if not internal_vel:
            raise Exception("TESTING PURPOSE: use internal vel")
        AIs["PolarNoncart"] = Polar_Angle_Vel_NonCart_AI([REs["PolarTargetPos"], self.REs["RobotVel"], self.obs["target_offset_angle"], self.obs["del_target_offset_angle"]], "Target", self.device)
        AIs["PolarNoncart"].set_static_measurement_noise("target_offset_angle", torch.eye(1, device=self.device)*1e-4)
        AIs["PolarNoncart"].set_static_measurement_noise("del_target_offset_angle", torch.eye(1, device=self.device)*1e-4)
        AIs["PolarNoncart"].set_static_measurement_noise("RobotVel", torch.eye(3, device=self.device)*1e-4)

        self.set_active_interconnections(AIs)

        goals = {
            "Go-To-Target" : GoToTargetGoal(REs["TargetPos"]),
            "PolarGo-To-Target" : GoToTargetGoal(REs["PolarTargetPos"]),
            "Stop"         : StopGoal(REs["RobotVel"]),
            "GazeFixation" : GazeFixationGoal(REs["PolarTargetPos"]),
        }
        for i in range(1, num_obstacles + 1):
            goals[f"AvoidObstacle{i}"] = AvoidObstacleGoal(REs[f"Obstacle{i}Pos"])
        self.set_goals(goals)

    def reset(self):
        self.REs["RobotVel"].set_state(torch.tensor([0.1, 0.1, 0.1], device=self.device), torch.eye(3, device=self.device)*1e1)
        self.REs["RobotVel"].set_static_motion_noise(torch.eye(3, device=self.device)*1e-1)

        if self.internal_vel:
            self.REs["TargetPos"].set_state(torch.tensor([20.0, 0.0, 0.0, 0.0, 0.0], device=self.device), torch.eye(5, device=self.device)*1e3)
            self.REs["TargetPos"].set_static_motion_noise(torch.eye(5, device=self.device)*1e-1)
            self.REs["PolarTargetPos"].set_state(torch.tensor([20.0, 0.0, 0.0, 0.0], device=self.device), torch.eye(4, device=self.device)*1e3)
            self.REs["PolarTargetPos"].set_static_motion_noise(torch.eye(4, device=self.device)*1e-1)
        else:
            self.REs["TargetPos"].set_state(torch.tensor([20.0, 0.0], device=self.device), torch.eye(2, device=self.device)*1e3)
            self.REs["TargetPos"].set_static_motion_noise(torch.eye(2, device=self.device)*1e-1)
            self.REs["PolarTargetPos"].set_state(torch.tensor([20.0, 0.0], device=self.device), torch.eye(2, device=self.device)*1e3)
            self.REs["PolarTargetPos"].set_static_motion_noise(torch.eye(2, device=self.device)*1e-1)

        for i in range(1, self.num_obstacles + 1):
            if self.internal_vel:
                self.REs[f"Obstacle{i}Pos"].set_state(torch.tensor([10.0, 0.0, 0.0, 0.0, 0.0], device=self.device), torch.eye(5, device=self.device)*1e3)
                self.REs[f"Obstacle{i}Pos"].set_static_motion_noise(torch.eye(5, device=self.device)*1e-1)
            else:
                self.REs[f"Obstacle{i}Pos"].set_state(torch.tensor([10.0, 0.0], device=self.device), torch.eye(2, device=self.device)*1e3)
                self.REs[f"Obstacle{i}Pos"].set_static_motion_noise(torch.eye(2, device=self.device)*1e-1)
            self.REs[f"Obstacle{i}Rad"].set_state(torch.tensor([1.0], device=self.device), torch.eye(1, device=self.device)*1e3)
            self.REs[f"Obstacle{i}Rad"].set_static_motion_noise(torch.eye(1, device=self.device)*1e-3)

    def render(self):
        estimator_means = {key: np.array(self.REs[key].state_mean.cpu()) for key in ["TargetPos"] + [f"Obstacle{i}Pos" for i in range(1, self.num_obstacles + 1)] if key in self.REs.keys()}
        estimator_covs = {key: np.array(self.REs[key].state_cov.cpu()) for key in ["TargetPos"] + [f"Obstacle{i}Pos" for i in range(1, self.num_obstacles + 1)] if key in self.REs.keys()}
        return self.env.render(1.0, estimator_means, estimator_covs)

    def eval_step(self, action):
        # Use a copy of the state to avoid modifying the actual state
        self.update_observations()
        buffer_dict = {key: estimator.set_buffer_dict() for key, estimator in list(self.REs.items()) + list(self.obs.items())}

        # ----------------- prediction updates -----------------------

        u_robot_vel = torch.concat([action[:2]*self.env.robot.max_acc, torch.tensor([action[2]*self.env.robot.max_acc_rot], device=self.device), torch.tensor([self.env.timestep], device=self.device)])
        self.REs["RobotVel"].call_predict(u_robot_vel, buffer_dict)

        u_pos = torch.concat([
            buffer_dict["RobotVel"]["state_mean"],
            torch.tensor(self.env.timestep, dtype=self.dtype, device=self.device),
        ]).squeeze() if not self.internal_vel else torch.tensor(self.env.timestep, dtype=self.dtype, device=self.device)

        self.REs["TargetPos"].call_predict(u_pos, buffer_dict)
        for i in range(1, self.num_obstacles + 1):
            self.REs[f"Obstacle{i}Pos"].call_predict(u_pos, buffer_dict)
            self.REs[f"Obstacle{i}Rad"].call_predict(u_pos, buffer_dict)

        # ----------------- measurement updates -----------------------

        self.REs["RobotVel"].call_update_with_specific_meas(self.AIs["RobotVel"], buffer_dict)
        self.REs["TargetPos"].call_update_with_specific_meas(self.AIs["TargetPos-Angle"], buffer_dict)
        # self.REs["PolarTargetPos"].call_update_with_specific_meas(self.AIs["PolarDistance"], buffer_dict)
        # self.REs["PolarTargetPos"].call_update_with_specific_meas(self.AIs["PolarAngle"], buffer_dict)
        self.REs["PolarTargetPos"].call_update_with_specific_meas(self.AIs["PolarNoncart"], buffer_dict)
        for i in range(1, self.num_obstacles + 1):
            self.REs[f"Obstacle{i}Pos"].call_update_with_specific_meas(self.AIs[f"Obstacle{i}Pos-Angle"], buffer_dict)
            self.REs[f"Obstacle{i}Rad"].call_update_with_specific_meas(self.AIs[f"Obstacle{i}Rad"], buffer_dict)

        self.REs["PolarTargetPos"].call_predict(u_pos, buffer_dict)

        return buffer_dict

    def compute_action(self, gradients):
        action = - gradients[1] - gradients[3]
        for i in range(self.num_obstacles):
            action -= gradients[i+4] / self.num_obstacles
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

# ================================================================================================================================