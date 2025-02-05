from typing import Dict
import torch

from components.aicon import DroneEnvAICON as AICON
from components.helpers import rotate_vector_2d
from models.experiment1.estimators import Robot_Vel_Estimator, Polar_Pos_Estimator, Polar_Pos_FovealVision_Estimator
from models.experiment1.active_interconnections import Triangulation_AI, Gaze_Fixation_AI
from models.experiment1.measurement_models import Robot_Vel_MM, Angle_MM
from models.experiment1.goals import PolarGoToTargetGoal, PolarGoToTargetFovealVisionGoal, PolarGoToTargetGazeFixationGoal

# ========================================================================================================

class Experiment1AICON(AICON):
    def __init__(self, env_config, aicon_type):
        assert aicon_type in ["Baseline", "Control", "Goal", "FovealVision", "Interconnection"], f"Invalid aicon_type: {aicon_type}"
        self.type = aicon_type
        super().__init__(env_config)

    def define_estimators(self):
        estimators = {
            "RobotVel":         Robot_Vel_Estimator(),
            "PolarTargetPos":   Polar_Pos_Estimator() if self.type != "FovealVision" else Polar_Pos_FovealVision_Estimator(foveal_vision_noise=self.env_config["fv_noise"], sensor_angle=self.env_config["robot_sensor_angle"]),
        }
        return estimators

    def define_measurement_models(self):
        return {
            "VelMM":        (Robot_Vel_MM(), ["RobotVel"]),
            "AngleMeasMM":  (Angle_MM(),     ["PolarTargetPos"]),
        }

    def define_active_interconnections(self):
        active_interconnections = {
            "TriangulationAI": Triangulation_AI(),
        }
        if self.type == "Interconnection":
            active_interconnections["GazeFixation"] = Gaze_Fixation_AI()
        return active_interconnections

    def define_goals(self):
        def select_goal():
            if self.type == "Goal":
                return PolarGoToTargetGazeFixationGoal()
            elif self.type == "FovealVision":
                return PolarGoToTargetFovealVisionGoal()
            else:
                return PolarGoToTargetGoal()
        goals = {
            "PolarGoToTarget": select_goal(),
        }
        
        return goals

    def eval_interconnections(self, buffer_dict: Dict[str, Dict[str, torch.Tensor]]):
        if self.type == "Interconnection":
            self.REs["RobotVel"].call_update_with_active_interconnection(self.AIs["GazeFixation"], buffer_dict)
            #self.REs["PolarTargetPos"].call_update_with_active_interconnection(self.AIs["GazeFixation"], buffer_dict)
        
        print("Pre Triangulation:"), self.print_estimator("PolarTargetPos", buffer_dict=buffer_dict, print_cov=2)
        self.REs["PolarTargetPos"].call_update_with_active_interconnection(self.AIs["TriangulationAI"], buffer_dict)
        print("Post Triangulation:"), self.print_estimator("PolarTargetPos", buffer_dict=buffer_dict, print_cov=2)
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
            action[:2] = rotate_vector_2d(self.REs["PolarTargetPos"].state_mean[1], torch.tensor([vel_radial, vel_tangential])).squeeze()
            # rotation control
            action[2] = 3.0 * self.REs["PolarTargetPos"].state_mean[1]
            return action

    def compute_action_from_gradient(self, gradients):
        # TODO: improve timestep scaling of action generation
        decay = 0.9 ** (self.env_config["timestep"] / 0.05)
        gradient_action = decay * self.last_action - 6e-1 * self.env_config["timestep"] * gradients["PolarGoToTarget"]
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
        print(f"True RobotVel: [{self.env.robot.vel[0]:.3f}, {self.env.robot.vel[1]:.3f}, {self.env.robot.vel_rot:.3f}]")
        print("--------------------------------------------------------------------")

    def custom_reset(self):
        self.goals["PolarGoToTarget"].desired_distance = self.env.target.distance

    def get_observation_update_uncertainty(self, key):
        if "angle" in key or "_rot" in key:
            update_uncertainty: torch.Tensor = 3e-1 * torch.eye(1)
        else:
            update_uncertainty: torch.Tensor = 5e-1 * torch.eye(1)
        return update_uncertainty