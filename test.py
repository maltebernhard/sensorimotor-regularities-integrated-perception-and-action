from typing import Dict
import numpy as np
import torch
import yaml
from components.aicon import AICON
from components.estimator import Obstacle_Pos_Estimator, RecursiveEstimator, Robot_Vel_Estimator, Target_Pos_Estimator
from components.measurement_model import Robot_Vel_MM, Pos_MM
from components.goal import AvoidObstacleGoal, GoToTargetGoal, StopGoal
from environment.gaze_fix_env import GazeFixEnv

# ==================================================================================

class MinimalAICON(AICON):
    def __init__(self):
        device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        
        #with open('config/env_config_zero_obst.yaml') as file:
        with open('config/env_config_two_obst.yaml') as file:
            env_config = yaml.load(file, Loader=yaml.FullLoader)
        env = GazeFixEnv(env_config)

        REs: Dict[str, RecursiveEstimator] = {
            "RobotVel" : Robot_Vel_Estimator(device),
            "TargetPos" : Target_Pos_Estimator(device),
            "Obstacle1Pos" : Obstacle_Pos_Estimator(device),
            "Obstacle2Pos" : Obstacle_Pos_Estimator(device),
        }

        AIs = {
            "RobotVel" : Robot_Vel_MM(device),
            "TargetPos" : Pos_MM(device, "target_offset_angle"),
            "Obstacle1Pos" : Pos_MM(device, "obstacle1_offset_angle"),
            "Obstacle2Pos" : Pos_MM(device, "obstacle2_offset_angle"),
        }

        goals = [
            GoToTargetGoal(REs["TargetPos"]),
            StopGoal(REs["RobotVel"]),
            AvoidObstacleGoal(REs["Obstacle1Pos"]),
            AvoidObstacleGoal(REs["Obstacle2Pos"]),
        ]
        super().__init__(device, env, REs, AIs, goals)

    def reset(self):
        self.REs["TargetPos"].set_state(torch.tensor([20.0, 0.0], device=self.device), torch.eye(2, device=self.device)*1e3)
        self.REs["TargetPos"].set_static_motion_noise(torch.eye(2, device=self.device)*1e-1)

        self.REs["Obstacle1Pos"].set_state(torch.tensor([10.0, 0.0], device=self.device), torch.eye(2, device=self.device)*1e3)
        self.REs["Obstacle1Pos"].set_static_motion_noise(torch.eye(2, device=self.device)*1e-1)

        self.REs["Obstacle2Pos"].set_state(torch.tensor([10.0, 0.0], device=self.device), torch.eye(2, device=self.device)*1e3)
        self.REs["Obstacle2Pos"].set_static_motion_noise(torch.eye(2, device=self.device)*1e-1)

        self.REs["RobotVel"].set_state(torch.tensor([0.0, 0.0, 0.0], device=self.device), torch.eye(3, device=self.device)*0.01)
        self.REs["RobotVel"].set_static_motion_noise(torch.eye(3, device=self.device)*1e-1)

    def render(self):
        estimator_means = {key: np.array(self.REs[key].state_mean.cpu()) for key in ["TargetPos", "Obstacle1Pos", "Obstacle2Pos"]}
        estimator_covs = {key: np.array(self.REs[key].state_cov.cpu()) for key in ["TargetPos", "Obstacle1Pos", "Obstacle2Pos"]}
        return self.env.render(1.0, estimator_means, estimator_covs)

    def eval_step(self, action):
        observations: dict = self.env.get_observation()
        
        # Use a copy of the state to avoid modifying the actual state
        buffer_dict = {key: estimator.set_buffer_dict() for key, estimator in self.REs.items()}
        
        u_robot_vel = torch.concat([action[:2]*self.env.robot.max_acc, torch.tensor([action[2]*self.env.robot.max_acc_rot], device=self.device), torch.tensor([self.env.timestep], device=self.device)])
        self.call_predict("RobotVel", u_robot_vel, buffer_dict["RobotVel"])
        
        u_target_pos = torch.stack([
            buffer_dict["RobotVel"]["state_mean"][0],
            buffer_dict["RobotVel"]["state_mean"][1],
            buffer_dict["RobotVel"]["state_mean"][2],
            torch.tensor(self.env.timestep, dtype=torch.float32, device=self.device),
        ]).squeeze()

        self.call_predict("TargetPos", u_target_pos, buffer_dict["TargetPos"])
        self.call_predict("Obstacle1Pos", u_target_pos, buffer_dict["Obstacle1Pos"])
        self.call_predict("Obstacle2Pos", u_target_pos, buffer_dict["Obstacle2Pos"])
        
        self.call_update_with_specific_meas("TargetPos", self.AIs["TargetPos"], {key: torch.tensor(val, device=self.device, dtype=torch.float32) for key, val in observations.items() if key in self.AIs["TargetPos"].meas_config.keys()}, buffer_dict["TargetPos"])
        self.call_update_with_specific_meas("RobotVel", self.AIs["RobotVel"], {"robot_vel": torch.tensor([observations["vel_frontal"]*self.env.robot.max_vel, observations["vel_lateral"]*self.env.robot.max_vel, observations["vel_rot"]*self.env.robot.max_vel_rot], device=self.device, dtype=torch.float32)}, buffer_dict["RobotVel"])
        self.call_update_with_specific_meas("Obstacle1Pos", self.AIs["Obstacle1Pos"], {"obstacle1_offset_angle": torch.tensor(observations["obstacle1_offset_angle"], device=self.device, dtype=torch.float32)}, buffer_dict["Obstacle1Pos"])
        self.call_update_with_specific_meas("Obstacle2Pos", self.AIs["Obstacle2Pos"], {"obstacle2_offset_angle": torch.tensor(observations["obstacle2_offset_angle"], device=self.device, dtype=torch.float32)}, buffer_dict["Obstacle2Pos"])
        return buffer_dict
    
    def compute_action(self, gradients):
        ret = - (gradients[0] + gradients[2])
        return ret

    def print_states(self):
        """
        print filter and environment states for debugging
        """
        print(f"Robot Vel Estimate: {[f'{x:.3f}' for x in self.REs['RobotVel'].state_mean.tolist()]}")
        actual_vel = list(self.env.robot.vel)
        actual_vel.append(self.env.robot.vel_rot)
        print(f"True Robot Vel:     {[f'{x:.3f}' for x in actual_vel]}")
        print("--------------------------------------------------------------------")
        

# ==================================================================================

if __name__ == "__main__":

    aicon = MinimalAICON()
    # seed for one obstacle example | demonstrates estimation error when switching offset angle from pi/2 to -pi/2
    #seed = 19
    # seed for two obstacle example
    seed = 7

    for run in range(10):
        aicon.run(150, seed+run, render=True, prints=1, step_by_step=True)