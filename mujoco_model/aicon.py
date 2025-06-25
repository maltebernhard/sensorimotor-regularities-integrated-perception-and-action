
from typing import Dict
import torch

from components.aicon import AICON
from .environment import MujocoEnv
from .estimators import Polar_Pos_Estimator, Robot_Vel_Estimator
from .goals import PolarGoToTargetGoal
from .smrs import Angle_MM, Distance_MM, DistanceDot_MM, Robot_Vel_MM, Triangulation_SMR

# ========================================================================================================

class MujocoAICON(AICON):
    def __init__(self):
        # self.env_config = env_config
        # self.smrs = aicon_type["smrs"]
        # self.distance_sensor = aicon_type["distance_sensor"]
        super().__init__()

    def define_env(self):
        """
        Sets the environment to the mujoco env for our experiment
        """
        return MujocoEnv('./mujoco_model/two_spheres.xml')

    def define_estimators(self):
        estimators = {
            "RobotVel":         Robot_Vel_Estimator(),
            "PolarTargetPos":   Polar_Pos_Estimator("Target", moving_object=False),
        }
        # for obs in range(self.env.num_obstacles):
        #     moving_obs = self.env_config["obstacles"][obs][0] != "stationary"
        #     estimators[f"PolarObstacle{obs+1}Pos"] = Polar_Pos_Estimator(f"Obstacle{obs+1}", moving_obs)
        return estimators

    def define_measurement_models(self):
        meas_models = {}
        # TODO: take care: Vel meas model is gone, we only improve vel uncertainty through meas models now
        meas_models["VelMM"]   = (Robot_Vel_MM(), ["RobotVel"])
        # ------------------- target -------------------
        meas_models["AngleMM"] = (Angle_MM("Target"), ["PolarTargetPos"])
        #meas_models["DistanceMM"] = (Distance_MM("Target"), ["PolarTargetPos"])
        #meas_models["DistanceMM"] = (DistanceDot_MM("Target", moving_object=False), ["PolarTargetPos"])
        # if "Divergence" in self.smrs:
        #     meas_models["DivergenceSMR"] = (Divergence_SMR("Target", target_config, fv_noise, sensor_angle), ["RobotVel", "PolarTargetPos"])
        meas_models["TriangulationSMR"] = (Triangulation_SMR("Target", moving_object=False), ["RobotVel", "PolarTargetPos"])
        # ------------------- obstacles -------------------
        # for obs in range(self.env.num_obstacles):
        #     moving_obs = self.env_config["obstacles"][obs][0] != "stationary"
        #     meas_models[f"AngleMM{obs+1}"] = (Angle_MM(f"Obstacle{obs+1}", fv_noise, sensor_angle), [f"PolarObstacle{obs+1}Pos"])
        #     if self.distance_sensor == "distsensor":
        #         meas_models[f"DistanceMM{obs+1}"] = (Distance_MM(f"Obstacle{obs+1}", moving_obs, fv_noise, sensor_angle), [f"PolarObstacle{obs+1}Pos"])
        #     elif self.distance_sensor == "distdotsensor":
        #         meas_models[f"DistanceMM{obs+1}"] = (DistanceDot_MM(f"Obstacle{obs+1}", moving_obs, fv_noise, sensor_angle), [f"PolarObstacle{obs+1}Pos"])
        #     if "Divergence" in self.smrs:
        #         meas_models[f"DivergenceSMR{obs+1}"] = (Divergence_SMR(f"Obstacle{obs+1}", moving_obs, fv_noise, sensor_angle), ["RobotVel", f"PolarObstacle{obs+1}Pos"])
        #     if "Triangulation" in self.smrs:
        #         meas_models[f"TriangulationSMR{obs+1}"] = (Triangulation_SMR(f"Obstacle{obs+1}", moving_obs, fv_noise, sensor_angle), ["RobotVel", f"PolarObstacle{obs+1}Pos"])
        return meas_models

    def define_active_interconnections(self):
        return {}

    def define_goal(self):
        return PolarGoToTargetGoal()

    def eval_interconnections(self, buffer_dict: Dict[str, Dict[str, torch.Tensor]]):
        return buffer_dict

    def compute_action_from_gradient(self, gradient):
        # TODO: improve timestep scaling of action generation
        decay = 0.9 ** (0.01 / 0.05)
        gradient_action = decay * self.last_action - 2e-1 * gradient
        return -gradient_action
    
    def print_estimators(self, buffer_dict=None):
        env_state = self.env.get_state()
        self.print_estimator('PolarTargetPos', buffer_dict=buffer_dict, print_cov=2)
        state = buffer_dict['PolarTargetPos']['mean'] if buffer_dict is not None else self.REs['PolarTargetPos'].mean
        real_state = torch.tensor([env_state['target_distance'], env_state['target_phi'], env_state['target_theta']])
        #real_state = torch.cat([real_state, torch.tensor([env_state['target_radius']])])
        #self.print_vector(state - real_state, "PolarTargetPos Err.")
        self.print_vector(real_state, "True PolarTargetPos: ")
        print("----------------------------------------------")
        # for i in range(1, self.env.num_obstacles+1):
        #     self.print_estimator(f"PolarObstacle{i}Pos", buffer_dict=buffer_dict, print_cov=2)
        #     state = buffer_dict[f"PolarObstacle{i}Pos"]['mean'] if buffer_dict is not None else self.REs[f"PolarObstacle{i}Pos"].mean
        #     real_state = torch.tensor([env_state[f'obstacle{i}_distance'], env_state[f'obstacle{i}_offset_angle']])
        #     if self.env_config["obstacles"][i-1][0] != "stationary":
        #         real_state = torch.cat([real_state, torch.tensor([env_state[f'obstacle{i}_distance_dot_global'], env_state[f'obstacle{i}_offset_angle_dot_global']])])
        #     real_state = torch.cat([real_state, torch.tensor([env_state[f'obstacle{i}_radius']])])
        #     self.print_vector(state - real_state, f"PolarObstacle{i}Pos Err.")
        #     #self.print_vector(real_state, f"True PolarObstacle{i}Pos: ")
        #     print("----------------------------------------------")
        self.print_estimator("RobotVel", buffer_dict=buffer_dict, print_cov=2)
        state = buffer_dict['RobotVel']['mean'] if buffer_dict is not None else self.REs['RobotVel'].mean
        real_state = torch.tensor([env_state['vel_x'], env_state['vel_y'], env_state['vel_z']])
        self.print_vector(state - real_state, "RobotVel Err.")
        self.print_vector(real_state, "True RobotVel: ")

    def get_control_input(self, action, buffer_dict, estimator_key) -> torch.Tensor:
        return torch.cat([torch.tensor([0.01]), action])

    def get_static_sensor_noise(self, key):
        # NOTE: you can use this to get real env sensor noise
        #(self.env.observation_noise[obs_key])
        if "phi" in key or "theta" in key:
            sensor_noise_mean   = 0.0
            sensor_noise_stddev = 1e-1
        elif "distance" in key:
            sensor_noise_mean   = 0.0
            sensor_noise_stddev = 1e-1
        else:
            sensor_noise_mean   = 0.0
            sensor_noise_stddev = 5e-1
        return torch.tensor(sensor_noise_mean), torch.eye(1)*sensor_noise_stddev

    def custom_reset(self):
        """
        sets desired distance and obstacles to avoid to the goal function upon environment reset
        """
        self.goal.desired_distance = 1.0
        self.goal.num_obstacles = 0