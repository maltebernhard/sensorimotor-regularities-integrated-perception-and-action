from typing import Dict
import numpy as np
import torch

from components.aicon import DroneEnvAICON as AICON
from components.instances.estimators import Polar_Pos_Estimator_Acc, Polar_Pos_Estimator_Vel, Robot_Vel_Estimator_Acc, Robot_Vel_Estimator_Vel
from components.instances.measurement_models import Angle_Meas_MM, Vel_MM
from components.instances.active_interconnections import Triangulation_AI
#from components.instances.goals import PolarGoToTargetGoal

from experiment_interconnection.active_interconnections import Gaze_Fixation_AI, Gaze_Fixation_Relative_AI, Gaze_Fixation_Constrained_AI
from experiment_interconnection.goals import PolarGoToTargetGoal

# ========================================================================================================

class ContingentInterconnectionAICON(AICON):
    def __init__(self, env_config):
        self.type = "ContingentInterconnection"
        super().__init__(**env_config)

    def define_estimators(self):
        estimators = {}
        estimators["RobotVel"] = Robot_Vel_Estimator_Acc(self.device) if not self.vel_control else Robot_Vel_Estimator_Vel(self.device)
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
            #"GazeFixation": Gaze_Fixation_AI([self.REs["PolarTargetPos"], self.REs["RobotVel"]], self.device),
            #"GazeFixation": Gaze_Fixation_Relative_AI([self.REs["PolarTargetPos"], self.REs["RobotVel"]], self.device),
            "GazeFixation": Gaze_Fixation_Constrained_AI([self.REs["PolarTargetPos"], self.REs["RobotVel"]], self.device),
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

    def compute_action(self, gradients):
        #decay = 1.0
        decay = 0.8
        #self.print_vector(gradients["PolarGoToTarget"], "GoToTarget Gradient")
        action = decay * self.last_action - 5e-2 * gradients["PolarGoToTarget"]
        return action
    
    def print_states(self, buffer_dict=None):
        obs = self.env.get_reality()
        self.print_state("PolarTargetPos", buffer_dict=buffer_dict)
        # TODO: observations can be None now
        print(f"True PolarTargetPos: [{obs['target_distance']:.3f}, {obs['target_offset_angle']:.3f}, {obs['target_distance_dot']:.3f}, {obs['target_offset_angle_dot']:.3f}]")
        self.print_state("RobotVel", buffer_dict=buffer_dict) 
        print(f"True RobotVel: [{self.env.robot.vel[0]:.3f}, {self.env.robot.vel[1]:.3f}, {self.env.robot.vel_rot:.3f}]")
        print("--------------------------------------------------------------------")

    def custom_reset(self):
        self.goals["PolarGoToTarget"].desired_distance = self.env.target.distance