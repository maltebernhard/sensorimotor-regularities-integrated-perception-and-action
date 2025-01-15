from typing import Dict
import numpy as np
import torch

from components.aicon import DroneEnvAICON as AICON
from components.instances.estimators import Polar_Pos_Estimator_Acc, Polar_Pos_Estimator_Vel, Robot_Vel_Estimator_Acc, Robot_Vel_Estimator_Vel
from components.instances.measurement_models import Angle_MM, Robot_Vel_MM
from components.instances.active_interconnections import Triangulation_AI

# works well together:
from models.old.experiment_interconnection.active_interconnections import Gaze_Fixation_Constrained_AI
from components.instances.goals import PolarGoToTargetGoal

# scaled goal necessary for unconstrained offset angle value AI:
from models.old.experiment_interconnection.active_interconnections import Gaze_Fixation_AI, Gaze_Fixation_Relative_AI
#from models.old.experiment_interconnection.goals import PolarGoToTargetGoal

# ========================================================================================================

class InterconnectionAICON(AICON):
    def __init__(self, env_config):
        self.type = "Interconnection"
        super().__init__(env_config)

    def define_estimators(self):
        estimators = {}
        estimators["RobotVel"] =        Robot_Vel_Estimator_Acc() if not self.vel_control else Robot_Vel_Estimator_Vel()
        estimators["PolarTargetPos"] =  Polar_Pos_Estimator_Acc() if not self.vel_control else Polar_Pos_Estimator_Vel()
        return estimators

    def define_measurement_models(self):
        return {
            "VelMM":        Robot_Vel_MM(),
            "AngleMeasMM":  Angle_MM(),
        }

    def define_active_interconnections(self):
        active_interconnections = {
            "TriangulationAI":  Triangulation_AI(),
            #"GazeFixation":    Gaze_Fixation_AI(),
            "GazeFixation":    Gaze_Fixation_Relative_AI(),
            #"GazeFixation":     Gaze_Fixation_Constrained_AI(),
        }
        return active_interconnections

    def define_goals(self):
        goals = {
            "PolarGoToTarget": PolarGoToTargetGoal(),
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
        action = decay * self.last_action - 5e-2 * gradients["PolarGoToTarget"]
        return action
    
    def print_estimators(self, buffer_dict=None):
        env_state = self.env.get_state()
        self.print_estimator("PolarTargetPos", buffer_dict=buffer_dict)
        print(f"True PolarTargetPos: [{env_state['target_distance']:.3f}, {env_state['target_offset_angle']:.3f}, {env_state['target_distance_dot']:.3f}, {env_state['target_offset_angle_dot']:.3f}]")
        self.print_estimator("RobotVel", buffer_dict=buffer_dict) 
        print(f"True RobotVel: [{self.env.robot.vel[0]:.3f}, {self.env.robot.vel[1]:.3f}, {self.env.robot.vel_rot:.3f}]")
        print("--------------------------------------------------------------------")

    def custom_reset(self):
        self.goals["PolarGoToTarget"].desired_distance = self.env.target.distance