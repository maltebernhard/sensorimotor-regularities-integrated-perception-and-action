from typing import Dict
import numpy as np
import torch
import yaml
from components.aicon import AICON
from components.estimator import Obstacle_Pos_Estimator, RecursiveEstimator, Robot_Vel_Estimator, Target_Pos_Estimator
from components.measurement_model import Obstacle_Pos_MM, Robot_Vel_MM, Target_Pos_MM
from components.goal import AvoidObstacleGoal, GoToTargetGoal, StopGoal
from environment.gaze_fix_env import GazeFixEnv
from torch.func import functional_call
from torch.func import jacrev


class MinimalAICON(AICON):
    def __init__(self):
        device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        
        #with open('config/env_config_zero_obst.yaml') as file:
        with open('config/env_config_one_obst.yaml') as file:
            env_config = yaml.load(file, Loader=yaml.FullLoader)
        env = GazeFixEnv(env_config)

        REs: Dict[str, RecursiveEstimator] = {
            "RobotVel" : Robot_Vel_Estimator(device),
            "TargetPos" : Target_Pos_Estimator(device),
            "ObstaclePos" : Obstacle_Pos_Estimator(device)
        }

        AIs = {
            "RobotVel" : Robot_Vel_MM(device),
            "TargetPos" : Target_Pos_MM(device),
            "ObstaclePos" : Obstacle_Pos_MM(device)
        }

        goals = [
            GoToTargetGoal(REs["TargetPos"]),
            StopGoal(REs["RobotVel"]),
            AvoidObstacleGoal(REs["ObstaclePos"])
        ]
        super().__init__(device, env, REs, AIs, goals)
        self.reset()

    def reset(self):
        self.REs["TargetPos"].set_state(torch.tensor([20.0, 0.0], device=self.device), torch.eye(2, device=self.device)*1e3)
        self.REs["TargetPos"].set_static_motion_noise(torch.eye(2, device=self.device)*1e-1)

        self.REs["TargetPos"].set_state(torch.tensor([10.0, 0.0], device=self.device), torch.eye(2, device=self.device)*1e3)
        self.REs["TargetPos"].set_static_motion_noise(torch.eye(2, device=self.device)*1e-1)

        self.REs["RobotVel"].set_state(torch.tensor([0.0, 0.0, 0.0], device=self.device), torch.eye(3, device=self.device)*0.01)
        self.REs["RobotVel"].set_static_motion_noise(torch.eye(3, device=self.device)*1e-1)

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
        print(f"Action: {env_action}")
        buffers = self.eval_step(env_action)
        for key, buffer_dict in buffers.items():
            self.REs[key].load_state_dict(buffer_dict)
        self.env.step(np.array(env_action.cpu()))
        self.env.render(1.0, np.array(self.REs["TargetPos"].state_mean.cpu()), np.array(self.REs["TargetPos"].state_cov.cpu()))

    def eval_step(self, action):
        observations: dict = self.env.get_observation()
        
        # Use a copy of the state to avoid modifying the actual state
        buffer_dict = {key: estimator.set_buffer_dict() for key, estimator in self.REs.items()}
        
        #u_robot_vel = torch.concat([action, torch.tensor([self.env.timestep], device=self.device)])
        u_robot_vel = torch.concat([action[:2]*self.env.robot.max_acc, torch.tensor([action[2]*self.env.robot.max_acc_rot], device=self.device), torch.tensor([self.env.timestep], device=self.device)])

        self.call_predict("RobotVel", u_robot_vel, buffer_dict["RobotVel"])
        
        u_target_pos = torch.stack([
            buffer_dict["RobotVel"]["state_mean"][0],
            buffer_dict["RobotVel"]["state_mean"][1],
            buffer_dict["RobotVel"]["state_mean"][2],
            torch.tensor(self.env.timestep, dtype=torch.float32, device=self.device),
        ]).squeeze()
        self.call_predict("TargetPos", u_target_pos, buffer_dict["TargetPos"])
        self.call_predict("ObstaclePos", u_target_pos, buffer_dict["ObstaclePos"])
        
        self.call_update_with_specific_meas("TargetPos", self.AIs["TargetPos"], {key: torch.tensor(val, device=self.device, dtype=torch.float32) for key, val in observations.items() if key in self.AIs["TargetPos"].meas_config.keys()}, buffer_dict["TargetPos"])
        self.call_update_with_specific_meas("RobotVel", self.AIs["RobotVel"], {"robot_vel": torch.tensor([observations["vel_frontal"]*self.env.robot.max_vel, observations["vel_lateral"]*self.env.robot.max_vel, observations["vel_rot"]*self.env.robot.max_vel_rot], device=self.device, dtype=torch.float32)}, buffer_dict["RobotVel"])
        self.call_update_with_specific_meas("ObstaclePos", self.AIs["ObstaclePos"], {"obstacle1_offset_angle": torch.tensor(observations["obstacle1_offset_angle"], device=self.device, dtype=torch.float32)}, buffer_dict["ObstaclePos"])
        return buffer_dict
    
    # def eval_goal(self, goal, buffer_dict):
    #     return goal.loss_function_from_buffer(buffer_dict)
    
    def _eval_goal_with_aux(self, action, goal):
        buffer_dict = self.eval_step(action)
        loss = goal.loss_function_from_buffer(buffer_dict)
        return loss, loss

    def compute_goal_action_jacobian(self, goal):
        action = torch.tensor([0.0, 0.0, 0.0], device=self.device)
        jacobian, step_eval = jacrev(
            self._eval_goal_with_aux,
            argnums=0, # x and meas_dict (all measurements within the dict!)
            has_aux=True)(action, goal)
        return jacobian

    def get_steepest_gradient_action(jacobian):
        return -jacobian

if __name__ == "__main__":
    aicon = MinimalAICON()
    seed = 15


    for run in range(10):
        aicon.reset()
        step = 0
        aicon.env.reset(seed=seed+run)

        for _ in range(100):
            
            jacobians = []
            for goal in aicon.goals:
                grad = aicon.compute_goal_action_jacobian(goal)
                if len(jacobians) == 0:
                    steepest_grad = grad
                else:
                    if grad.norm() > steepest_grad.norm():
                        steepest_grad = grad
                jacobians.append(grad)

            #action = -steepest_grad
            action = - (jacobians[0] + jacobians[2])


            # jac = aicon.compute_goal_action_jacobian(aicon.goals[0])
            # action = -jac
            
            aicon.step(action)
            step += 1
            if step % 100 == 0:
                print(f"============================ Step {step} ================================")
                print(f"Robot Vel Estimate: {aicon.REs['RobotVel'].state_mean.tolist()}")
                actual_vel = list(aicon.env.robot.vel)
                actual_vel.append(aicon.env.robot.vel_rot)
                print(f"True Robot Vel:     {actual_vel}")

            #input("Press Enter to continue...\n")