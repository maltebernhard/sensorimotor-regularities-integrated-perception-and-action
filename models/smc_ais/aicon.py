from typing import Dict
import torch

from components.aicon import DroneEnvAICON as AICON
from components.helpers import rotate_vector_2d
from models.smc_ais.estimators import Polar_Pos_Estimator, Robot_Vel_Estimator
from models.smc_ais.helpers import get_foveal_noise
from models.smc_ais.goals import PolarGoToTargetGoal
from models.smc_ais.smcs import Angle_MM, Distance_MM, Robot_Vel_MM, Triangulation_SMC, Divergence_SMC

# ========================================================================================================

class SMCAICON(AICON):
    def __init__(self, env_config, aicon_type):
        #assert aicon_type in [""], f"Invalid aicon_type: {aicon_type}"
        self.smcs = aicon_type["SMCs"]
        self.distance_sensor = aicon_type["DistanceSensor"]
        self.control = aicon_type["Control"]
        super().__init__(env_config)

    def define_estimators(self):
        estimators = {
            "RobotVel":         Robot_Vel_Estimator(),
            "PolarTargetPos":   Polar_Pos_Estimator(),
        }
        return estimators

    def define_measurement_models(self):
        sensor_noise = self.env_config["observation_noise"]
        foveal_vision_noise = self.env_config["foveal_vision_noise"]
        meas_models = {}
        if self.distance_sensor:
            meas_models["DistanceMM"] = (Distance_MM(sensor_noise=sensor_noise, foveal_vision_noise=foveal_vision_noise), ["PolarTargetPos"])
        meas_models["VelMM"]   = (Robot_Vel_MM(), ["RobotVel"])
        meas_models["AngleMM"] = (Angle_MM(),     ["PolarTargetPos"])
        if "Triangulation" in self.smcs:
            meas_models["TriangulationSMC"] = (Triangulation_SMC(sensor_noise=sensor_noise, foveal_vision_noise=foveal_vision_noise), ["PolarTargetPos", "RobotVel"])
        if "Divergence" in self.smcs:
            meas_models["DivergenceSMC"] = (Divergence_SMC(sensor_noise=sensor_noise, foveal_vision_noise=foveal_vision_noise), ["PolarTargetPos", "RobotVel"])
        return meas_models

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

    def compute_action(self):
        if not self.control:
            return super().compute_action()
        else:
            action = torch.zeros(3)

            # goal control
            vel_radial = 2e-1 * (self.REs["PolarTargetPos"].state_mean[0] - self.goals["PolarGoToTarget"].desired_distance)
            vel_tangential = 0.0

            # smc control
            if "Triangulation" in self.smcs:
                vel_tangential += 5e-1 * self.REs["PolarTargetPos"].state_cov[0][0]
            if "Divergence" in self.smcs:
                vel_radial    += 1e-1 * self.REs["PolarTargetPos"].state_cov[0][0] * vel_radial.sign()

            # rotation control
            if len(self.env.foveal_vision_noise) > 0:
                action[2] = 3.0 * self.REs["PolarTargetPos"].state_mean[1]

            action[:2] = rotate_vector_2d(self.REs["PolarTargetPos"].state_mean[1], torch.tensor([vel_radial, vel_tangential])).squeeze()
            
            return {}, action

    def compute_action_from_gradient(self, gradients):
        # TODO: improve timestep scaling of action generation
        decay = 0.9 ** (self.env_config["timestep"] / 0.05)
        gradient_action = decay * self.last_action - torch.tensor([1e-1, 1e-1, 1e1]) * self.env_config["timestep"] * gradients["PolarGoToTarget"]
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
        for smc in [val[0] for val in self.MMs.values()]:
            sensor_vals, sensor_covs = smc.transform_state_to_innovation_space(buffer_dict[smc.state_component]['state_mean'], buffer_dict['RobotVel']['state_mean'])
            for i, val in enumerate(sensor_vals):
                buffer_dict[smc.required_observations[i]]['state_mean'] = val
                if sensor_covs is not None:
                    buffer_dict[smc.required_observations[i]]['state_cov'] = sensor_covs[i]
