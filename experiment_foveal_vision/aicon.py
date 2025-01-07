import numpy as np
import torch
from typing import Dict

from components.aicon import DroneEnvAICON as AICON
from components.estimator import RecursiveEstimator
from components.instances.estimators import Robot_Vel_Estimator_Vel #, Polar_Pos_Estimator_Vel
from components.instances.measurement_models import Vel_MM, Angle_Meas_MM
from components.instances.active_interconnections import Triangulation_AI
from components.instances.goals import PolarGoToTargetGoal

from experiment_foveal_vision.estimators import Polar_Pos_Estimator_Vel

# =============================================================================================================================================================

class FovealVisionAICON(AICON):
    def __init__(self, vel_control=True, moving_target=False, sensor_angle_deg=360, num_obstacles=0, timestep=0.05):
        self.type = "FovealVision"
        super().__init__(vel_control, moving_target, sensor_angle_deg, num_obstacles, timestep)

    def define_estimators(self):
        REs: Dict[str, RecursiveEstimator] = {
            "RobotVel": Robot_Vel_Estimator_Vel(self.device),
            "PolarTargetPos": Polar_Pos_Estimator_Vel(self.device, "PolarTargetPos"),
        }
        return REs

    def define_measurement_models(self):
        MMs = {
            "RobotVel": Vel_MM(self.device),
            "PolarAngle": Angle_Meas_MM(self.device, "Target"),
        }
        return MMs

    def define_active_interconnections(self):
        AIs = {
            "PolarDistance": Triangulation_AI([self.REs["PolarTargetPos"], self.REs["RobotVel"]], self.device),
        }
        return AIs

    def define_goals(self):
        goals = {
            "PolarGoToTarget" : PolarGoToTargetGoal(self.device),
        }
        return goals

    def render(self):
        target_mean, target_cov = self.convert_polar_to_cartesian_state(self.REs["PolarTargetPos"].state_mean, self.REs["PolarTargetPos"].state_cov)
        estimator_means: Dict[str, torch.Tensor] = {"PolarTargetPos": target_mean}
        estimator_covs: Dict[str, torch.Tensor] = {"PolarTargetPos": target_cov}
        return self.env.render(1.0, {key: np.array(mean.cpu()) for key, mean in estimator_means.items()}, {key: np.array(cov.cpu()) for key, cov in estimator_covs.items()})

    def eval_update(self, action: torch.Tensor, new_step: bool, buffer_dict: Dict[str, Dict[str, torch.Tensor]]):
        u = self.get_control_input(action)

        # ----------------------------- predicts -------------------------------------

        self.REs["RobotVel"].call_predict(u, buffer_dict)
        self.REs["PolarTargetPos"].call_predict(u, buffer_dict)

        print("------------------- Post Predict -------------------")
        print(buffer_dict['PolarTargetPos']['state_mean'])
        print(buffer_dict['PolarTargetPos']['state_cov'])

        # ----------------------------- active interconnections -------------------------------------
        
        if new_step:
            self.meas_updates(buffer_dict)

        self.REs["PolarTargetPos"].call_update_with_active_interconnection(self.AIs["PolarDistance"], buffer_dict)

        print("------------------- Post Update -------------------")
        print(buffer_dict['PolarTargetPos']['state_mean'])
        print(buffer_dict['PolarTargetPos']['state_cov'])

        # ----------------------------- measurements -------------------------------------

        return buffer_dict

    def compute_action(self, gradients):
        self.print_vector(gradients["PolarGoToTarget"], "GoToTarget Gradient")
        action = 0.9 * self.last_action - 1e-2 * gradients["PolarGoToTarget"]
        return action

    def print_states(self, print_cov=False):
        """
        print filter and environment states for debugging
        """
        obs = self.env.get_reality()
        self.print_state("RobotVel", print_cov=print_cov)
        actual_vel = list(self.env.robot.vel)
        actual_vel.append(self.env.robot.vel_rot)
        print(f"True Robot Vel: {[f'{x:.3f}' for x in actual_vel]}")
        # print("--------------------------------------------------------------------")
        self.print_state("PolarTargetPos", print_cov=print_cov)
        actual_pos = self.env.rotation_matrix(-self.env.robot.orientation) @ (self.env.target.pos - self.env.robot.pos)
        angle = np.arctan2(actual_pos[1], actual_pos[0])
        dist = np.linalg.norm(actual_pos)
        print(f"True PolarTargetPos: {[f'{x:.3f}' for x in [dist, angle, obs['target_distance_dot'], obs['target_offset_angle_dot'] if obs['target_offset_angle_dot'] else 0.0]]}")
        print("--------------------------------------------------------------------")
    
    def custom_reset(self):
        self.goals["PolarGoToTarget"].desired_distance = self.env.target.distance