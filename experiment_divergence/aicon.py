from typing import Dict
import numpy as np
import torch

from components.aicon import DroneEnvAICON as AICON
from components.instances.estimators import Robot_Vel_Estimator_Vel
from components.instances.measurement_models import Vel_MM

from experiment_divergence.estimators import Polar_Pos_Rad_Estimator
from experiment_divergence.active_interconnections import Triangulation_AI
from experiment_divergence.measurement_models import Vis_Angle_MM, Angle_Meas_MM
from experiment_divergence.goals import PolarGoToTargetGoal

# ========================================================================================================

class DivergenceAICON(AICON):
    def __init__(self, vel_control=True, moving_target=False, sensor_angle_deg=360, num_obstacles=0, timestep=0.05):
        self.type = "Divergence"
        super().__init__(vel_control, moving_target, sensor_angle_deg, num_obstacles, timestep)

    def define_estimators(self):
        estimators = {
            "RobotVel": Robot_Vel_Estimator_Vel(self.device),
            "PolarTargetPosRadius": Polar_Pos_Rad_Estimator(self.device, "PolarTargetPosRadius"),
        }
        return estimators

    def define_measurement_models(self):
        return {
            "VelMM": Vel_MM(self.device),
            "AngleMeasMM": Angle_Meas_MM(self.device),
            "VisAngleMM": Vis_Angle_MM(self.device),
        }

    def define_active_interconnections(self):
        active_interconnections = {
            "TriangulationAI": Triangulation_AI([self.REs["PolarTargetPosRadius"], self.REs["RobotVel"]], self.device),
        }
        return active_interconnections

    def define_goals(self):
        goals = {
            "PolarGoToTarget": PolarGoToTargetGoal(self.device),
        }
        return goals

    def eval_update(self, action: torch.Tensor, new_step: bool, buffer_dict: Dict[str, Dict[str, torch.Tensor]]):
        u = self.get_control_input(action)

        self.REs["RobotVel"].call_predict(u, buffer_dict)
        self.REs["PolarTargetPosRadius"].call_predict(u, buffer_dict)

        self.REs["PolarTargetPosRadius"].call_update_with_active_interconnection(self.AIs["TriangulationAI"], buffer_dict)

        if new_step:
            self.meas_updates(buffer_dict)
        
        return buffer_dict

    def render(self):
        target_mean, target_cov = self.convert_polar_to_cartesian_state(self.REs["PolarTargetPosRadius"].state_mean, self.REs["PolarTargetPosRadius"].state_cov)
        estimator_means: Dict[str, torch.Tensor] = {"PolarTargetPosRadius": target_mean}
        estimator_covs: Dict[str, torch.Tensor] = {"PolarTargetPosRadius": target_cov}
        return self.env.render(1.0, {key: np.array(mean.cpu()) for key, mean in estimator_means.items()}, {key: np.array(cov.cpu()) for key, cov in estimator_covs.items()})

    def compute_action(self, gradients):
        decay = 0.9
        return decay * self.last_action - 1e-2 * gradients["PolarGoToTarget"]
    
    def print_states(self, buffer_dict=None):
        obs = self.env.get_reality()
        print("--------------------------------------------------------------------")
        self.print_state("PolarTargetPosRadius", buffer_dict=buffer_dict, print_cov=2)
        # TODO: observations can be None now
        print(f"True PolarTargetPosRadius: [{obs['target_distance']:.3f}, {obs['target_offset_angle']:.3f}, {obs['target_distance_dot']:.3f}, {obs['target_offset_angle_dot']:.3f}, {obs['target_radius']:.3f}]")
        print("--------------------------------------------------------------------")
        self.print_state("RobotVel", buffer_dict=buffer_dict) 
        print(f"True RobotVel: [{self.env.robot.vel[0]}, {self.env.robot.vel[1]}, {self.env.robot.vel_rot}]")
        print("--------------------------------------------------------------------")

    def custom_reset(self):
        self.goals["PolarGoToTarget"].desired_distance = self.env.target.distance
