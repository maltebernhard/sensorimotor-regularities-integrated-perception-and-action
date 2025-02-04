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
        self.smcs = aicon_type["smcs"]
        self.distance_sensor = aicon_type["distance_sensor"]
        self.control = aicon_type["control"]
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

    def compute_action_gradients(self):
        if not self.control:
            return super().compute_action_gradients()
        else:
            # goal control
            task_grad = torch.zeros(3)
            task_vel_radial = 2e-1 * (self.REs["PolarTargetPos"].state_mean[0] - self.goals["PolarGoToTarget"].desired_distance)
            task_grad[:2] = - rotate_vector_2d(self.REs["PolarTargetPos"].state_mean[1], torch.tensor([task_vel_radial, 0.0])).squeeze()

            # smc control
            unc_grad = torch.zeros(3)
            unc_vel_tangential = 5e-1 * self.REs["PolarTargetPos"].state_cov[0][0] if "Triangulation" in self.smcs else 0.0
            unc_vel_radial     = 1e-1 * self.REs["PolarTargetPos"].state_cov[0][0] * task_vel_radial.sign() if "Divergence" in self.smcs else 0
            unc_grad[:2] = - rotate_vector_2d(self.REs["PolarTargetPos"].state_mean[1], torch.tensor([unc_vel_radial, unc_vel_tangential])).squeeze()
            unc_grad[2] = - 5e-3 * self.REs["PolarTargetPos"].state_cov[0][0] * self.REs["PolarTargetPos"].state_mean[1].sign() if len(self.env.foveal_vision_noise) > 0 else 0.0
            
            return {"PolarGoToTarget": {
                "task_gradient": task_grad,
                "uncertainty_gradient": unc_grad,
                "total": task_grad + unc_grad
            }}

    def compute_action_from_gradient(self, gradients):
        # TODO: improve timestep scaling of action generation
        decay = 0.9 ** (self.env_config["timestep"] / 0.05)
        gradient_action = decay * self.last_action - torch.tensor([2e0, 2e0, 3e2]) * self.env_config["timestep"] * gradients["PolarGoToTarget"]
        return gradient_action
    
    def print_estimators(self, buffer_dict=None):
        env_state = self.env.get_state()
        self.print_estimator("PolarTargetPos", buffer_dict=buffer_dict, print_cov=2)
        state = buffer_dict['PolarTargetPos']['state_mean'] if buffer_dict is not None else self.REs['PolarTargetPos'].state_mean
        self.print_vector(state - torch.tensor([env_state['target_distance'], env_state['target_offset_angle'], env_state['target_visual_angle']]), "PolarTargetPos Err.")
        #print(f"True PolarTargetPos: [{env_state['target_distance']:.3f}, {env_state['target_offset_angle']:.3f}, {env_state['target_distance_dot']:.3f}, {env_state['target_offset_angle_dot']+env_state['vel_rot']:.3f}]")#, {env_state['target_radius']:.3f}]")

    def custom_reset(self):
        self.goals["PolarGoToTarget"].desired_distance = self.env.target.distance
    
    def get_observation_update_uncertainty(self, key):
        if "angle" in key or "_rot" in key:
            update_uncertainty: torch.Tensor = 5e-2 * torch.eye(1)
        else:
            update_uncertainty: torch.Tensor = 2e-1 * torch.eye(1)
        return update_uncertainty

    def get_custom_sensor_noise(self, obs: dict):
        # get foveal vision noise and estimations
        expected_obs_with_noise = self.get_contingent_measurements()
        for key in obs.keys():
            # get noise scaling with measurement
            if "vel" in key or "distance" in key:
                expected_obs_with_noise[key]['noise'] += self.obs[key].sensor_noise * expected_obs_with_noise[key]['mean']
            else:
                expected_obs_with_noise[key]['noise'] += self.obs[key].sensor_noise
        return {key: expected_obs_with_noise[key]['noise'] for key in obs.keys()}
    
    def get_contingent_measurements(self):
        obs_dict = {}
        for smc_key, smc in [(key, val[0]) for key, val in self.MMs.items()]:
            sensor_vals, sensor_covs = smc.transform_state_to_innovation_space(self.REs[smc.state_component].state_mean, self.REs['RobotVel'].state_mean)
            for i, val in enumerate(sensor_vals):
                obs_dict[smc.required_observations[i]] = {'mean': val, 'noise': 0.0*torch.eye(1)}
                if sensor_covs is not None:
                    obs_dict[smc.required_observations[i]]['noise'] = sensor_covs[i].sqrt()
        return obs_dict

    def adapt_contingent_measurements(self, buffer_dict: dict):
        for smc_key, smc in [(key, val[0]) for key, val in self.MMs.items()]:
            sensor_vals, sensor_covs = smc.transform_state_to_innovation_space(buffer_dict[smc.state_component]['state_mean'], buffer_dict['RobotVel']['state_mean'])
            for i, val in enumerate(sensor_vals):
                buffer_dict[smc.required_observations[i]]['state_mean'] = val
                if sensor_covs is not None:
                    buffer_dict[smc.required_observations[i]]['state_cov'] = sensor_covs[i]
