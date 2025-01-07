from typing import Dict
import numpy as np
import torch

from components.aicon import DroneEnvAICON as AICON
from components.instances.estimators import Robot_Vel_Estimator_Vel, Polar_Pos_Estimator_Vel
from components.instances.measurement_models import Angle_Meas_MM, Vel_MM
from components.instances.active_interconnections import Triangulation_AI

from experiment_goal.goals import GoToTargetGoal

# ========================================================================================================

class ContingentGoalAICON(AICON):
    def __init__(self, vel_control=True, moving_target=False, sensor_angle_deg=360, num_obstacles=0, timestep = 0.05):
        self.type = "ContingentGoal"
        super().__init__(vel_control, moving_target, sensor_angle_deg, num_obstacles, timestep)

    def define_estimators(self):
        estimators = {
            "RobotVel": Robot_Vel_Estimator_Vel(self.device),
            "PolarTargetPos": Polar_Pos_Estimator_Vel(self.device, "PolarTargetPos"),
        }
        return estimators

    def define_measurement_models(self):
        return {
            "VelMM": Vel_MM(self.device),
            "AngleMeasMM": Angle_Meas_MM(self.device, "Target"),
        }

    def define_active_interconnections(self):
        active_interconnections = {
            "TriangulationAI": Triangulation_AI([self.REs["PolarTargetPos"], self.REs["RobotVel"]], self.device),
        }
        return active_interconnections

    def define_goals(self):
        goals = {
            "GoToTarget": GoToTargetGoal(self.device),
        }
        return goals

    def eval_update(self, action: torch.Tensor, new_step: bool, buffer_dict: Dict[str, Dict[str, torch.Tensor]]):
        u = self.get_control_input(action)

        self.REs["RobotVel"].call_predict(u, buffer_dict)
        self.REs["PolarTargetPos"].call_predict(u, buffer_dict)

        if new_step:
            self.REs["RobotVel"].call_update_with_meas_model(self.MMs["VelMM"], buffer_dict, self.get_meas_dict(self.MMs["VelMM"]))
            meas_dict = self.get_meas_dict(self.MMs["AngleMeasMM"])
            if len(meas_dict["means"]) == len(self.MMs["AngleMeasMM"].observations):
                self.REs["PolarTargetPos"].call_update_with_meas_model(self.MMs["AngleMeasMM"], buffer_dict, meas_dict)
            else:
                print("No angle measurement.")
        self.REs["PolarTargetPos"].call_update_with_active_interconnection(self.AIs["TriangulationAI"], buffer_dict)
        
        return buffer_dict

    def render(self):
        target_mean, target_cov = self.convert_polar_to_cartesian_state(self.REs["PolarTargetPos"].state_mean, self.REs["PolarTargetPos"].state_cov)
        estimator_means: Dict[str, torch.Tensor] = {"PolarTargetPos": target_mean}
        estimator_covs: Dict[str, torch.Tensor] = {"PolarTargetPos": target_cov}
        return self.env.render(1.0, {key: np.array(mean.cpu()) for key, mean in estimator_means.items()}, {key: np.array(cov.cpu()) for key, cov in estimator_covs.items()})

    def compute_action(self, gradients):
            decay = 0.9
            return decay * self.last_action - 1e-2 * gradients["GoToTarget"]
    
    def print_states(self, buffer_dict=None):
        obs = self.env.get_reality()
        print("--------------------------------------------------------------------")
        self.print_state("PolarTargetPos", buffer_dict=buffer_dict, print_cov=2)
        actual_pos = self.env.rotation_matrix(-self.env.robot.orientation) @ (self.env.target.pos - self.env.robot.pos)
        angle = np.arctan2(actual_pos[1], actual_pos[0])
        dist = np.linalg.norm(actual_pos)
        # TODO: observations can be None now
        print(f"True PolarTargetPos: [{dist:.3f}, {angle:.3f}, {obs['target_distance_dot']:.3f}, {obs['target_offset_angle_dot']:.3f}]")
        print("--------------------------------------------------------------------")
        self.print_state("RobotVel", buffer_dict=buffer_dict) 
        print(f"True RobotVel: [{self.env.robot.vel[0]}, {self.env.robot.vel[1]}, {self.env.robot.vel_rot}]")
        print("--------------------------------------------------------------------")

    def custom_reset(self):
        self.goals["GoToTarget"].desired_distance = self.env.target.distance
