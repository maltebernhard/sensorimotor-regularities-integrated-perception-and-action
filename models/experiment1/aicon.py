from typing import Dict
import torch

from components.aicon import DroneEnvAICON as AICON
from components.helpers import rotate_vector_2d
from components.instances.estimators import Robot_Vel_Estimator_Vel, Polar_Pos_Estimator_Vel
from components.instances.active_interconnections import Triangulation_AI
from components.instances.measurement_models import Robot_Vel_MM, Angle_MM
from components.instances.goals import PolarGoToTargetGoal
from models.experiment1.active_interconnections import Gaze_Fixation_AI
from models.experiment1.estimators import Polar_Pos_FovealVision_Estimator_Vel
from models.experiment1.goals import PolarGoToTargetGazeFixationGoal

# ========================================================================================================

class Experiment1AICON(AICON):
    def __init__(self, env_config, aicon_type):
        assert aicon_type in ["Baseline", "Control", "Goal", "FovealVision", "Interconnection"], f"Invalid aicon_type: {aicon_type}"
        self.type = aicon_type
        super().__init__(env_config)

    def define_estimators(self):
        estimators = {
            "RobotVel":         Robot_Vel_Estimator_Vel(),
            "PolarTargetPos":   Polar_Pos_Estimator_Vel() if self.type != "FovealVision" else Polar_Pos_FovealVision_Estimator_Vel(),
        }
        return estimators

    def define_measurement_models(self):
        return {
            "VelMM":        Robot_Vel_MM(),
            "AngleMeasMM":  Angle_MM(),
        }

    def define_active_interconnections(self):
        active_interconnections = {
            "TriangulationAI": Triangulation_AI(),
        }
        if self.type == "Interconnection":
            active_interconnections["GazeFixation"] = Gaze_Fixation_AI()
        return active_interconnections

    def define_goals(self):
        goals = {
            "PolarGoToTarget": PolarGoToTargetGoal() if self.type != "Goal" else PolarGoToTargetGazeFixationGoal(),
        }
        return goals

    def eval_interconnections(self, buffer_dict: Dict[str, Dict[str, torch.Tensor]]):
        if self.type == "Interconnection":
            self.REs["PolarTargetPos"].call_update_with_active_interconnection(self.AIs["GazeFixation"], buffer_dict)
        self.REs["PolarTargetPos"].call_update_with_active_interconnection(self.AIs["TriangulationAI"], buffer_dict)
        return buffer_dict

    def compute_action(self):
        if self.type != "Baseline":
            return super().compute_action()
        else:
            action = torch.zeros(3)
            # go to target control
            vel_radial = 1.0 / 3.0 * (self.REs["PolarTargetPos"].state_mean[0] - self.goals["PolarGoToTarget"].desired_distance)
            # tangential motion control
            vel_tangential = 1.0 / 10.0 * self.REs["PolarTargetPos"].state_cov[0][0]
            action[:2] = rotate_vector_2d(-self.REs["PolarTargetPos"].state_mean[1], torch.tensor([vel_radial, vel_tangential])).squeeze()
            # rotation control
            action[2] = 2.0 * self.REs["PolarTargetPos"].state_mean[1] #+ 0.01 * self.REs["PolarTargetPos"].state_mean[3]
            return action

    def compute_action_from_gradient(self, gradients):
        decay = 0.9
        gradient_action = decay * self.last_action - 3e-2 * gradients["PolarGoToTarget"]
        # control
        if self.type == "Control":
            gradient_action[2] = 2.0 * self.REs["PolarTargetPos"].state_mean[1] + 0.01 * self.REs["PolarTargetPos"].state_mean[3]
        return gradient_action
    
    def print_estimators(self, buffer_dict=None):
        env_state = self.env.get_state()
        print("--------------------------------------------------------------------")
        self.print_estimator("PolarTargetPos", buffer_dict=buffer_dict, print_cov=2)
        print(f"True PolarTargetPos: [{env_state['target_distance']:.3f}, {env_state['target_offset_angle']:.3f}, {env_state['target_distance_dot']:.3f}, {env_state['target_offset_angle_dot']:.3f}, {env_state['target_radius']:.3f}]")
        print("--------------------------------------------------------------------")
        self.print_estimator("RobotVel", buffer_dict=buffer_dict) 
        print(f"True RobotVel: [{self.env.robot.vel[0]}, {self.env.robot.vel[1]}, {self.env.robot.vel_rot}]")
        print("--------------------------------------------------------------------")

    def custom_reset(self):
        self.goals["PolarGoToTarget"].desired_distance = self.env.target.distance
