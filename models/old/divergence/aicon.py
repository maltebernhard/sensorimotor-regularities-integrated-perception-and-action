from typing import Dict
import torch

from components.aicon import DroneEnvAICON as AICON
from models.old.divergence.estimators import PolarPos_Tri_Div_Estimator
from models.old.divergence.measurement_models import Angle_MM
from models.old.divergence.goals import PolarGoToTargetGoal

# ========================================================================================================

class DivergenceAICON(AICON):
    def __init__(self, env_config):
        self.type = "Divergence"
        super().__init__(env_config)

    def define_estimators(self):
        estimators = {
            "PolarTargetPos":   PolarPos_Tri_Div_Estimator(),
        }
        return estimators

    def define_measurement_models(self):
        return {
            "AngleMeasMM":  (Angle_MM(),     ["PolarTargetPos"]),
        }

    def define_active_interconnections(self):
        active_interconnections = {}
        return active_interconnections

    def define_goals(self):
        goals = {
            "PolarGoToTarget": PolarGoToTargetGoal(),
        }
        return goals

    def eval_interconnections(self, buffer_dict: Dict[str, Dict[str, torch.Tensor]]):
        return buffer_dict

    def compute_action_from_gradient(self, gradients):
        # TODO: improve timestep scaling of action generation
        decay = 0.9 ** (self.env_config["timestep"] / 0.05)
        gradient_action = decay * self.last_action - 1e-1 * self.env_config["timestep"] * gradients["PolarGoToTarget"]
        return gradient_action
    
    def print_estimators(self, buffer_dict=None):
        env_state = self.env.get_state()
        print("--------------------------------------------------------------------")
        self.print_estimator("PolarTargetPos", buffer_dict=buffer_dict, print_cov=2)
        print(f"True PolarTargetPos: [{env_state['target_distance']:.3f}, {env_state['target_offset_angle']:.3f}, {env_state['target_distance_dot']:.3f}, {env_state['target_offset_angle_dot']+env_state['vel_rot']:.3f}, {env_state['target_visual_angle']:.3f}, {env_state['target_visual_angle_dot']:.3f}]")
        print("--------------------------------------------------------------------")

    def custom_reset(self):
        self.goals["PolarGoToTarget"].desired_distance = self.env.target.distance
    
    def adapt_contingent_measurements(self, buffer_dict: dict):
        predicted_angle = buffer_dict['PolarTargetPos']['state_mean'][1]
        buffer_dict['target_offset_angle']['state_mean']     = predicted_angle
        buffer_dict['target_offset_angle_dot']['state_mean'] = buffer_dict['PolarTargetPos']['state_mean'][3]
        buffer_dict['target_visual_angle_dot']['state_mean'] = buffer_dict['PolarTargetPos']['state_mean'][5]
