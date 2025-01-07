from typing import Dict
import numpy as np
import torch

from components.aicon import DroneEnvAICON as AICON
from components.instances.estimators import Polar_Pos_Estimator_Acc, Polar_Pos_Estimator_Vel
from components.instances.measurement_models import Angle_Meas_MM, Vel_MM
from components.instances.active_interconnections import Triangulation_AI

from experiment_interconnection.active_interconnections import Gaze_Fixation_AI, Gaze_Fixation_AI2
from experiment_interconnection.estimators import Robot_State_Estimator_Acc, Robot_State_Estimator_Vel
from experiment_interconnection.goals import PolarGoToTargetGoal, PolarGoToTargetGazeFixationGoal

# ========================================================================================================

class ContingentInterconnectionAICON(AICON):
    def __init__(self, vel_control=True, moving_target=False, sensor_angle_deg=360, num_obstacles=0, timestep=0.05):
        self.type = "ContingentInterconnection"
        super().__init__(vel_control, moving_target, sensor_angle_deg, num_obstacles, timestep)

    def define_estimators(self):
        estimators = {}
        estimators["RobotVel"] = Robot_State_Estimator_Acc(self.device) if not self.vel_control else Robot_State_Estimator_Vel(self.device)
        estimators["PolarTargetPos"] = Polar_Pos_Estimator_Acc(self.device, "PolarTargetPos") if not self.vel_control else Polar_Pos_Estimator_Vel(self.device, "PolarTargetPos")
        return estimators

    def define_measurement_models(self):
        return {
            "VelMM": Vel_MM(self.device),
            "AngleMeasMM": Angle_Meas_MM(self.device, "Target"),
        }

    def define_active_interconnections(self):
        active_interconnections = {
            "TriangulationAI": Triangulation_AI([self.REs["PolarTargetPos"], self.REs["RobotVel"]], self.device),
            "GazeFixation": Gaze_Fixation_AI([self.REs["PolarTargetPos"], self.REs["RobotVel"]], self.device),
        }
        return active_interconnections

    def define_goals(self):
        goals = {
            "GoToTarget": PolarGoToTargetGoal(self.device),
            #"GoToTarget": PolarGoToTargetGazeFixationGoal(self.device),
        }
        return goals

    def eval_update(self, action: torch.Tensor, new_step: bool, buffer_dict: Dict[str, Dict[str, torch.Tensor]]):
        u = self.get_control_input(action)

        self.REs["RobotVel"].call_predict(u, buffer_dict)
        self.REs["PolarTargetPos"].call_predict(u, buffer_dict)

        if new_step:
            self.REs["RobotVel"].call_update_with_meas_model(self.MMs["VelMM"], buffer_dict, self.get_meas_dict(self.MMs["VelMM"]))
            self.REs["PolarTargetPos"].call_update_with_meas_model(self.MMs["AngleMeasMM"], buffer_dict, self.get_meas_dict(self.MMs["AngleMeasMM"]))
            self.REs["PolarTargetPos"].call_update_with_active_interconnection(self.AIs["TriangulationAI"], buffer_dict)
        else:
            # NOTE: both connections individually have the same effect
            self.REs["PolarTargetPos"].call_update_with_active_interconnection(self.AIs["GazeFixation"], buffer_dict)
            #self.REs["RobotVel"].call_update_with_active_interconnection(self.AIs["GazeFixation"], buffer_dict)

            # NOTE: completely chaotic behavior
            # self.REs["RobotVel"].call_update_with_active_interconnection(self.AIs["GazeFixation"], buffer_dict)
            # self.REs["PolarTargetPos"].call_update_with_active_interconnection(self.AIs["GazeFixation"], buffer_dict)

            self.REs["PolarTargetPos"].call_update_with_active_interconnection(self.AIs["TriangulationAI"], buffer_dict)

        return buffer_dict

    def get_control_input(self, action: torch.Tensor):
        env_action = torch.empty_like(action)
        env_action[:2] = (action[:2] / action[:2].norm() if action[:2].norm() > 1.0 else action[:2]) * (self.env.robot.max_vel if self.vel_control else self.env.robot.max_acc)
        env_action[2] = action[2] * (self.env.robot.max_vel_rot if self.vel_control else self.env.robot.max_acc_rot)
        return torch.concat([torch.tensor([0.05], device=self.device), env_action])

    def render(self):
        target_mean, target_cov = self.convert_polar_to_cartesian_state(self.REs["PolarTargetPos"].state_mean, self.REs["PolarTargetPos"].state_cov)
        estimator_means: Dict[str, torch.Tensor] = {"PolarTargetPos": target_mean}
        estimator_covs: Dict[str, torch.Tensor] = {"PolarTargetPos": target_cov}
        return self.env.render(1.0, {key: np.array(mean.cpu()) for key, mean in estimator_means.items()}, {key: np.array(cov.cpu()) for key, cov in estimator_covs.items()})

    def compute_action(self, gradients):
        #decay = 1.0
        decay = 0.8
        self.print_vector(gradients["GoToTarget"], "GoToTarget Gradient")
        action = decay * self.last_action - 5e-2 * gradients["GoToTarget"]
        return action
    
    def print_states(self, buffer_dict=None):
        obs = self.env.get_reality()
        print("--------------------------------------------------------------------")
        self.print_state("RobotVel", buffer_dict=buffer_dict)
        actual_pos = self.env.rotation_matrix(-self.env.robot.orientation) @ (self.env.target.pos - self.env.robot.pos)
        angle = np.arctan2(actual_pos[1], actual_pos[0])
        dist = np.linalg.norm(actual_pos)
        #TODO: Consider observations that are None
        #print(f"True Polar Target Position: {[f'{x:.3f}' for x in [dist, angle, obs['target_distance_dot'], obs['target_offset_angle_dot']]]}")
        print(f"True Polar Target Position: {[f'{x:.3f}' for x in [dist, angle]]}")
        actual_state = [self.env.robot.vel[0], self.env.robot.vel[1], self.env.robot.vel_rot]
        actual_pos = self.env.rotation_matrix(-self.env.robot.orientation) @ (self.env.target.pos - self.env.robot.pos)
        angle = np.arctan2(actual_pos[1], actual_pos[0])
        dist = np.linalg.norm(actual_pos)
        #TODO: Consider observations that are None
        #actual_state += [dist, angle, obs['target_distance_dot'], obs['target_offset_angle_dot']]
        actual_state += [dist, angle]
        print(f"True State: ", end="")
        [print(f'{x:.3f} ', end="") for x in actual_state]
        print()
        print("--------------------------------------------------------------------")

    def custom_reset(self):
        self.goals["GoToTarget"].desired_distance = self.env.target.distance