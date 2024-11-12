import numpy as np
import torch
from torch.func import jacrev, functional_call
from abc import ABC, abstractmethod
from typing import Dict, List

import yaml
from components.active_interconnection import ActiveInterconnection
from components.estimator import Obstacle_Rad_Estimator, Pos_Estimator_External_Vel, Pos_Estimator_Internal_Vel, RecursiveEstimator, Robot_Vel_Estimator
from components.goal import AvoidObstacleGoal, GoToTargetGoal, Goal, StopGoal
from components.measurement_model import Pos_Vel_MM, Pos_MM, Radius_MM, Robot_Vel_MM
from environment.gaze_fix_env import GazeFixEnv

# ========================================================================================

class AICON(ABC):
    def __init__(self, device, env, REs, AIs, goals):
        self.device = device
        self.env = env
        self.REs: Dict[str, RecursiveEstimator] = REs
        self.AIs: Dict[str, ActiveInterconnection] = AIs
        self.goals: Dict[str, Goal] = goals
        self.reset()

    def call_predict(self, estimator, u, buffer_dict):
        args_to_be_passed = ('predict',)
        kwargs = {'u': u}
        return functional_call(self.REs[estimator], buffer_dict, args_to_be_passed, kwargs)

    def call_update_with_specific_meas(self, estimator, specific_meas_model, meas_dict: Dict[str, torch.Tensor], buffer_dict):
        args_to_be_passed = ('update_with_specific_meas', specific_meas_model)
        kwargs = {'meas_dict': meas_dict}
        return functional_call(self.REs[estimator], buffer_dict, args_to_be_passed, kwargs)

    def step(self, action):
        gain = 10.0
        env_action = action * gain
        if env_action[:2].norm() > 1.0:
            env_action[:2] = env_action[:2] / env_action[:2].norm()
        if env_action[2] > 1.0:
            env_action[2] = 1.0
        #print(f"Action: {env_action}")
        self.env.step(np.array(env_action.cpu()))
        buffers = self.eval_step(env_action)
        for key, buffer_dict in buffers.items():
            self.REs[key].load_state_dict(buffer_dict)

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
    
    def run(self, timesteps, env_seed, render = True, prints = 0, step_by_step = True):
        """
        Runs AICON on the environment for a given number of timesteps
        Args:
            timesteps:    number of timesteps to run
            env_seed:     seed for the environment
            render:       render the environment
            prints:       print the states every "prints" timesteps
            step_by_step: wait for user input after each step
        """
        self.reset()
        self.env.reset(seed=env_seed)
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
        assert num_obstacles in [0, 1, 2], "Number of obstacles must be 0, 1 or 2"

        self.internal_vel = internal_vel
        self.num_obstacles = num_obstacles
        device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        
        config = {
            0: 'config/env_config_zero_obst.yaml',
            1: 'config/env_config_one_obst.yaml',
            2: 'config/env_config_two_obst.yaml'
        }
        with open(config[num_obstacles]) as file:
            env_config = yaml.load(file, Loader=yaml.FullLoader)
        env = GazeFixEnv(env_config)

        REs: Dict[str, RecursiveEstimator] = {"RobotVel": Robot_Vel_Estimator(device)}
        if internal_vel:
            REs["TargetPos"] = Pos_Estimator_Internal_Vel(device, "TargetPos")
            if num_obstacles > 0:
                REs["Obstacle1Pos"] = Pos_Estimator_Internal_Vel(device, "Obstacle1Pos")
                REs["Obstacle1Rad"] = Obstacle_Rad_Estimator(device)
            if num_obstacles > 1:
                REs["Obstacle2Pos"] = Pos_Estimator_Internal_Vel(device, "Obstacle2Pos")
                REs["Obstacle2Rad"] = Obstacle_Rad_Estimator(device)
        else:
            REs["TargetPos"] = Pos_Estimator_External_Vel(device, "TargetPos")
            if num_obstacles > 0:
                REs["Obstacle1Pos"] = Pos_Estimator_External_Vel(device, "Obstacle1Pos")
                REs["Obstacle1Rad"] = Obstacle_Rad_Estimator(device)
            if num_obstacles > 1:
                REs["Obstacle2Pos"] = Pos_Estimator_External_Vel(device, "Obstacle2Pos")
                REs["Obstacle2Rad"] = Obstacle_Rad_Estimator(device)

        AIs = {"RobotVel": Robot_Vel_MM(device)}
        if internal_vel:
            AIs["TargetPos"] = Pos_Vel_MM(device)
            if num_obstacles > 0:
                AIs["Obstacle1Pos"] = Pos_Vel_MM(device)
                AIs["Obstacle1Rad"] = Radius_MM(device)
            if num_obstacles > 1:
                AIs["Obstacle2Pos"] = Pos_Vel_MM(device)
                AIs["Obstacle2Rad"] = Radius_MM(device)
        else:
            AIs["TargetPos"] = Pos_MM(device)
            if num_obstacles > 0:
                AIs["Obstacle1Pos"] = Pos_MM(device)
                AIs["Obstacle1Rad"] = Radius_MM(device)
            if num_obstacles > 1:
                AIs["Obstacle2Pos"] = Pos_MM(device)
                AIs["Obstacle2Rad"] = Radius_MM(device)
    
        goals = {
            "Go-To-Target"     : GoToTargetGoal(REs["TargetPos"]),
            "Stop"             : StopGoal(REs["RobotVel"]),
        }
        if num_obstacles > 0:
            goals["AvoidObstacle1"] = AvoidObstacleGoal(REs["Obstacle1Pos"])
        if num_obstacles > 1:
            goals["AvoidObstacle2"] = AvoidObstacleGoal(REs["Obstacle2Pos"])
        
        super().__init__(device, env, REs, AIs, goals)

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

    def render(self):
        estimator_means = {key: np.array(self.REs[key].state_mean.cpu()) for key in ["TargetPos", "Obstacle1Pos", "Obstacle2Pos"] if key in self.REs.keys()}
        estimator_covs = {key: np.array(self.REs[key].state_cov.cpu()) for key in ["TargetPos", "Obstacle1Pos", "Obstacle2Pos"] if key in self.REs.keys()}
        return self.env.render(1.0, estimator_means, estimator_covs)

    def eval_step(self, action):
        observations: dict = self.env.get_observation()

        # Use a copy of the state to avoid modifying the actual state
        buffer_dict = {key: estimator.set_buffer_dict() for key, estimator in self.REs.items()}
        
        u_robot_vel = torch.concat([action[:2]*self.env.robot.max_acc, torch.tensor([action[2]*self.env.robot.max_acc_rot], device=self.device), torch.tensor([self.env.timestep], device=self.device)])
        self.call_predict("RobotVel", u_robot_vel, buffer_dict["RobotVel"])
        
        u_pos = torch.stack([
            buffer_dict["RobotVel"]["state_mean"][0],
            buffer_dict["RobotVel"]["state_mean"][1],
            buffer_dict["RobotVel"]["state_mean"][2],
            torch.tensor(self.env.timestep, dtype=torch.float32, device=self.device),
        ]).squeeze() if not self.internal_vel else torch.tensor(self.env.timestep, dtype=torch.float32, device=self.device)

        self.call_predict("TargetPos", u_pos, buffer_dict["TargetPos"])
        if self.num_obstacles > 0:
            self.call_predict("Obstacle1Pos", u_pos, buffer_dict["Obstacle1Pos"])
            self.call_predict("Obstacle1Rad", u_pos, buffer_dict["Obstacle1Rad"])
        if self.num_obstacles > 1:
            self.call_predict("Obstacle2Pos", u_pos, buffer_dict["Obstacle2Pos"])
            self.call_predict("Obstacle2Rad", u_pos, buffer_dict["Obstacle2Rad"])
        
        
        self.call_update_with_specific_meas(
            "RobotVel",
            self.AIs["RobotVel"],
            {
                "robot_vel": torch.tensor([
                    observations["vel_frontal"]*self.env.robot.max_vel,
                    observations["vel_lateral"]*self.env.robot.max_vel,
                    observations["vel_rot"]*self.env.robot.max_vel_rot
                ], device=self.device, dtype=torch.float32)
            },
            buffer_dict["RobotVel"]
        )

        target_pos_meas_dict = {"offset_angle": torch.tensor(observations["target_offset_angle"], device=self.device, dtype=torch.float32)}
        if self.internal_vel: target_pos_meas_dict["robot_vel"] = buffer_dict["RobotVel"]["state_mean"]
        self.call_update_with_specific_meas("TargetPos", self.AIs["TargetPos"], target_pos_meas_dict, buffer_dict["TargetPos"])

        if self.num_obstacles > 0:
            obst1_meas_dict = {"offset_angle": torch.tensor(observations["obstacle1_offset_angle"], device=self.device, dtype=torch.float32)}
            if self.internal_vel: obst1_meas_dict["robot_vel"] = buffer_dict["RobotVel"]["state_mean"]
            self.call_update_with_specific_meas("Obstacle1Pos", self.AIs["Obstacle1Pos"], obst1_meas_dict, buffer_dict["Obstacle1Pos"])
            self.call_update_with_specific_meas("Obstacle1Rad", self.AIs["Obstacle1Rad"], {"pos": buffer_dict["Obstacle1Pos"]["state_mean"][:2], "visual_angle": torch.tensor(observations["obstacle1_visual_angle"], device=self.device, dtype=torch.float32)}, buffer_dict["Obstacle1Rad"])
        if self.num_obstacles > 1:
            obst2_meas_dict = {"offset_angle": torch.tensor(observations["obstacle2_offset_angle"], device=self.device, dtype=torch.float32)}
            if self.internal_vel: obst2_meas_dict["robot_vel"] = buffer_dict["RobotVel"]["state_mean"]
            self.call_update_with_specific_meas("Obstacle2Pos", self.AIs["Obstacle2Pos"], obst2_meas_dict, buffer_dict["Obstacle2Pos"])
            self.call_update_with_specific_meas("Obstacle2Rad", self.AIs["Obstacle2Rad"], {"pos": buffer_dict["Obstacle2Pos"]["state_mean"][:2], "visual_angle": torch.tensor(observations["obstacle2_visual_angle"], device=self.device, dtype=torch.float32)}, buffer_dict["Obstacle2Rad"])
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
        print(f"Target Position Estimates: {[f'{x:.3f}' for x in self.REs['TargetPos'].state_mean.tolist()]}")
        actual_pos = list(self.env.rotation_matrix(-self.env.robot.orientation) @ (self.env.target.pos - self.env.robot.pos))
        for v in actual_vel: actual_pos.append(-v)
        print(f"True Target Position:      {[f'{x:.3f}' for x in actual_pos]}")
        print("--------------------------------------------------------------------")
        if self.num_obstacles > 1:
            print(f"Obstacle Radius Estimates: {[f'{x:.3f}' for x in self.REs['Obstacle1Rad'].state_mean.tolist()]}, {[f'{x:.3f}' for x in self.REs['Obstacle2Rad'].state_mean.tolist()]}")
            actual_rads = [self.env.obstacles[0].radius, self.env.obstacles[1].radius]
            print(f"True Obstacle Radius:      {[f'{x:.3f}' for x in actual_rads]}")
            print("--------------------------------------------------------------------")

# ================================================================================================================================