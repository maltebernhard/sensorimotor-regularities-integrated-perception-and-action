from typing import Dict
import torch

from components.aicon import DroneEnvAICON as AICON
from components.helpers import rotate_vector_2d
from models.smc_ais.estimators import Polar_Pos_Estimator, Polar_PosVel_Estimator, Robot_Vel_Estimator, Robot_Vel_Estimator_Acc_Action, Robot_VelWind_Estimator
from models.smc_ais.goals import PolarGoToTargetGoal
from models.smc_ais.smcs import Angle_MM, Distance_MM, DistanceVel_MM, DivergenceVel_SMC, Robot_Vel_MM, Triangulation_SMC, Divergence_SMC, TriangulationVel_SMC

# ========================================================================================================

class SMCAICON(AICON):
    def __init__(self, env_config, aicon_type):
        self.smcs = aicon_type["smcs"]
        self.distance_sensor = aicon_type["distance_sensor"]
        self.control = aicon_type["controller"]
        super().__init__(env_config)

    def define_estimators(self):
        if self.env_config["action_mode"] == 1:
            robot_vel_estimator = Robot_Vel_Estimator_Acc_Action(self.env.robot.max_vel, self.env.robot.max_vel_rot)
        elif self.env_config["action_mode"] == 3:
            robot_vel_estimator = Robot_Vel_Estimator() if self.env_config["wind"] == (0.0, 0.0) else Robot_VelWind_Estimator()
        else:
            raise ValueError("Invalid action mode")
        estimators = {
            "RobotVel":         robot_vel_estimator,
            "PolarTargetPos":   Polar_Pos_Estimator() if self.env_config["moving_target"][0] == "stationary" else Polar_PosVel_Estimator(),
        }
        return estimators

    def define_measurement_models(self):
        fv_noise = self.env_config["fv_noise"]
        sensor_angle = self.env_config["robot_sensor_angle"]
        meas_models = {}
        if self.distance_sensor == "distsensor":
            meas_models["DistanceMM"] = (Distance_MM(fv_noise=fv_noise, sensor_angle=sensor_angle), ["PolarTargetPos"]) if self.env_config["moving_target"][0] == "stationary" else (DistanceVel_MM(fv_noise=fv_noise, sensor_angle=sensor_angle), ["PolarTargetPos"])
        # TODO: take care: Vel meas model is gone, we only improve vel uncertainty through meas models now
        #meas_models["VelMM"]   = (Robot_Vel_MM(), ["RobotVel"])
        meas_models["AngleMM"] = (Angle_MM(fv_noise=fv_noise, sensor_angle=sensor_angle), ["PolarTargetPos"])
        if "Divergence" in self.smcs:
            smc = Divergence_SMC(fv_noise=fv_noise, sensor_angle=sensor_angle) if self.env_config["moving_target"][0] == "stationary" else DivergenceVel_SMC(fv_noise=fv_noise, sensor_angle=sensor_angle)
            #meas_models["DivergenceSMC"] = (smc, ["PolarTargetPos"])
            meas_models["DivergenceSMC"] = (smc, ["RobotVel", "PolarTargetPos"])
        if "Triangulation" in self.smcs:
            smc = Triangulation_SMC(fv_noise=fv_noise, sensor_angle=sensor_angle) if self.env_config["moving_target"][0] == "stationary" else TriangulationVel_SMC(fv_noise=fv_noise, sensor_angle=sensor_angle)
            #meas_models["TriangulationSMC"] = (smc, ["PolarTargetPos"])
            meas_models["TriangulationSMC"] = (smc, ["RobotVel", "PolarTargetPos"])
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
        if self.control == "aicon":
            return super().compute_action_gradients()
        else:
            # goal control
            task_grad = torch.zeros(3)
            task_vel_radial = 2e-1 * (self.REs["PolarTargetPos"].mean[0] - self.goals["PolarGoToTarget"].desired_distance)
            task_grad[:2] = - rotate_vector_2d(self.REs["PolarTargetPos"].mean[1], torch.tensor([task_vel_radial, 0.0])).squeeze()

            # smc control
            unc_grad = torch.zeros(3)
            unc_vel_tangential = 5e-1 * self.REs["PolarTargetPos"].cov[0][0] if "Triangulation" in self.smcs else 0.0
            unc_vel_radial     = 1e-1 * self.REs["PolarTargetPos"].cov[0][0] * task_vel_radial.sign() if "Divergence" in self.smcs else 0
            unc_grad[:2] = - rotate_vector_2d(self.REs["PolarTargetPos"].mean[1], torch.tensor([unc_vel_radial, unc_vel_tangential])).squeeze()
            unc_grad[2] = - 5e-3 * self.REs["PolarTargetPos"].cov[0][0] * self.REs["PolarTargetPos"].mean[1].sign() if len(self.env.fv_noise) > 0 else 0.0
            
            return {"PolarGoToTarget": {
                "distance":             task_grad,
                "distance_uncertainty": unc_grad,
                "total":                task_grad + unc_grad
            }}

    def compute_action_from_gradient(self, gradients):
        # TODO: improve timestep scaling of action generation
        if self.env_config["action_mode"] == 3:
            decay = 0.9 ** (self.env_config["timestep"] / 0.05)
            gradient_action = decay * self.last_action - torch.tensor([2e0, 2e0, 3e1]) * self.env_config["timestep"] * gradients["PolarGoToTarget"]
        elif self.env_config["action_mode"] == 1:
            #decay = 0.9 ** (self.env_config["timestep"] / 0.05)
            decay = 1.0
            gradient_action = decay * self.last_action - torch.tensor([2e3, 2e3, 0.0]) * self.env_config["timestep"] * gradients["PolarGoToTarget"]
        return gradient_action
    
    def print_estimators(self, buffer_dict=None):
        env_state = self.env.get_state()
        self.print_estimator("PolarTargetPos", buffer_dict=buffer_dict, print_cov=2)
        state = buffer_dict['PolarTargetPos']['mean'] if buffer_dict is not None else self.REs['PolarTargetPos'].mean
        real_state = torch.tensor([env_state['target_distance'], env_state['target_offset_angle']])
        if self.env_config["moving_target"][0] != "stationary":
            real_state = torch.cat([real_state, torch.tensor([env_state['target_distance_dot_global'], env_state['target_offset_angle_dot_global']])])
        real_state = torch.cat([real_state, torch.tensor([env_state['target_radius']])])
        self.print_vector(state - real_state, "PolarTargetPos Err.")
        #self.print_vector(real_state, "True PolarTargetPos: ")
        print("----------------------------------------------")
        self.print_estimator("RobotVel", buffer_dict=buffer_dict, print_cov=2)
        state = buffer_dict['RobotVel']['mean'] if buffer_dict is not None else self.REs['RobotVel'].mean
        wind_robot_frame = rotate_vector_2d(-self.env.robot.orientation, torch.tensor([self.env.wind[0], self.env.wind[1]]))
        if type(self.REs["RobotVel"]) in [Robot_Vel_Estimator, Robot_Vel_Estimator_Acc_Action]:
            real_state = torch.tensor([env_state['vel_frontal'], env_state['vel_lateral'], env_state['vel_rot']])
        else:
            real_state = torch.tensor([env_state['vel_frontal'], env_state['vel_lateral'], env_state['vel_rot'], wind_robot_frame[0], wind_robot_frame[1]])
        self.print_vector(state - real_state, "RobotVel Err.")
        self.print_vector(real_state, "True RobotVel: ")

    def get_control_input(self, action, buffer_dict, estimator_key) -> torch.Tensor:
        env_action = torch.empty_like(action)
        env_action[:2] = (action[:2] / action[:2].norm() if action[:2].norm() > 1.0 else action[:2]) * (self.env.robot.max_vel if self.env_config["action_mode"]==3 else self.env.robot.max_acc)
        env_action[2] = action[2] * (self.env.robot.max_vel_rot if self.env_config["action_mode"]==3 else self.env.robot.max_acc_rot)
        if estimator_key == "RobotVel":
            if self.prints > 0 and self.current_step % self.prints == 0:
                print("Action: ", end=""), self.print_vector(env_action)
            return torch.concat([torch.tensor([self.env_config["timestep"]]), env_action])
        elif estimator_key == "PolarTargetPos":
            if type(self.REs["RobotVel"]) == Robot_Vel_Estimator:
                return torch.concat([torch.tensor([self.env_config["timestep"]]), env_action])
            elif type(self.REs["RobotVel"]) in [Robot_VelWind_Estimator, Robot_Vel_Estimator_Acc_Action]:
                return torch.cat([torch.tensor([self.env_config["timestep"]]), buffer_dict["RobotVel"]["mean"][:3]])


    def get_static_sensor_noise(self, key):
        # NOTE: you can use this to get real env sensor noise
        #(self.env.observation_noise[obs_key])
        if "angle" in key or "rot" in key:
            sensor_noise_mean   = 0.0
            sensor_noise_stddev = 1e-1
        elif "distance" in key:
            sensor_noise_mean   = 0.0
            sensor_noise_stddev = 10e0
        else:
            sensor_noise_mean   = 0.0
            sensor_noise_stddev = 5e-1
        return torch.tensor(sensor_noise_mean), torch.eye(1)*sensor_noise_stddev
