from typing import Dict
import torch

from components.aicon import DroneEnvAICON as AICON
from models.smc_ais.estimators import Polar_Pos_Estimator, Robot_Vel_Estimator
from models.smc_ais.helpers import get_foveal_noise
from models.smc_ais.measurement_models import Angle_MM
from models.smc_ais.goals import PolarGoToTargetGoal
from models.smc_ais.smcs import SMC_MM

# ========================================================================================================

class SMCAICON(AICON):
    def __init__(self, env_config):
        self.type = "SMC"
        super().__init__(env_config)

    def define_estimators(self):
        estimators = {
            "RobotVel":         Robot_Vel_Estimator(),
            "PolarTargetPos":   Polar_Pos_Estimator(),
        }
        return estimators

    def define_measurement_models(self):
        smcs = ["Triangulation", "Divergence"]
        if len(self.env_config["foveal_vision_noise"]) > 0: smcs += ["FovealVision"]
        return {
            #"AngleMeasMM":      (Angle_MM(),          ["PolarTargetPos"]),
            "TriangulationSMC": (SMC_MM(smcs=smcs, sensor_noise=self.env_config["observation_noise"], foveal_vision_noise=self.env_config["foveal_vision_noise"]), ["PolarTargetPos", "RobotVel"]),
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
        gradient_action = decay * self.last_action - torch.tensor([1e-1, 1e-1, 1e2]) * self.env_config["timestep"] * gradients["PolarGoToTarget"]
        return gradient_action
    
    def print_estimators(self, buffer_dict=None):
        env_state = self.env.get_state()
        self.print_estimator("PolarTargetPos", buffer_dict=buffer_dict, print_cov=2)
        state = buffer_dict['PolarTargetPos']['state_mean'] if buffer_dict is not None else self.REs['PolarTargetPos'].state_mean
        self.print_vector(state - torch.tensor([env_state['target_distance'], env_state['target_offset_angle'], env_state['target_visual_angle']]), "PolarTargetPos Err.")
        #print(f"True PolarTargetPos: [{env_state['target_distance']:.3f}, {env_state['target_offset_angle']:.3f}, {env_state['target_distance_dot']:.3f}, {env_state['target_offset_angle_dot']+env_state['vel_rot']:.3f}]")#, {env_state['target_radius']:.3f}]")

    def custom_reset(self):
        self.goals["PolarGoToTarget"].desired_distance = self.env.target.distance
    
    def get_custom_sensor_noise(self, obs: dict):
        observation_noise = {}
        for key in obs.keys():
            if key in self.env_config["foveal_vision_noise"].keys(): observation_noise[key] = get_foveal_noise(obs["target_offset_angle"], key, self.env_config["foveal_vision_noise"], self.env_config["robot_sensor_angle"])
            else: observation_noise[key] = 0.0
        return {key: val*torch.eye(1) for key, val in observation_noise.items()}
    
    def adapt_contingent_measurements(self, buffer_dict: dict):
        sensor_vals, sensor_covs = self.MMs["TriangulationSMC"][0].transform_state_to_innovation_space(buffer_dict['PolarTargetPos']['state_mean'], buffer_dict['RobotVel']['state_mean'])
        buffer_dict['target_offset_angle']['state_mean']     = sensor_vals[0]
        if "Triangulation" in self.MMs["TriangulationSMC"][0].smcs:
            buffer_dict['target_offset_angle_dot']['state_mean'] = sensor_vals[1]
            if "Divergence" in self.MMs["TriangulationSMC"][0].smcs:
                buffer_dict['target_visual_angle']['state_mean']     = sensor_vals[2]
                buffer_dict['target_visual_angle_dot']['state_mean'] = sensor_vals[3]
        elif "Divergence" in self.MMs["TriangulationSMC"][0].smcs:
            buffer_dict['target_visual_angle']['state_mean']     = sensor_vals[1]
            buffer_dict['target_visual_angle_dot']['state_mean'] = sensor_vals[2]
        if sensor_covs is not None:
            buffer_dict['target_offset_angle']['state_cov']      = sensor_covs[0]
            if "Triangulation" in self.MMs["TriangulationSMC"][0].smcs:
                buffer_dict['target_offset_angle_dot']['state_cov'] = sensor_covs[1]
                if "Divergence" in self.MMs["TriangulationSMC"][0].smcs:
                    buffer_dict['target_visual_angle']['state_cov']     = sensor_covs[2]
                    buffer_dict['target_visual_angle_dot']['state_cov'] = sensor_covs[3]
            elif "Divergence" in self.MMs["TriangulationSMC"][0].smcs:
                buffer_dict['target_visual_angle']['state_cov']     = sensor_covs[1]
                buffer_dict['target_visual_angle_dot']['state_cov'] = sensor_covs[2]
