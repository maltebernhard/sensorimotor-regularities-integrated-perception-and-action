from datetime import datetime
import os
import numpy as np
import torch
from torch.func import jacrev
from abc import ABC, abstractmethod
from typing import Dict, List

import matplotlib.pyplot as plt
import networkx as nx
import yaml

from components.estimator import Observation, RecursiveEstimator
from components.logger import AICONLogger
from components.measurement_model import MeasurementModel
from components.active_interconnection import ActiveInterconnection
from components.goal import Goal
from environment.base_env import BaseEnv
from environment.gaze_fix_env import GazeFixEnv

# ========================================================================================

class AICON(ABC):
    def __init__(self):
        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        self.dtype = torch.float64
        torch.set_default_device(self.device)
        torch.set_default_dtype(self.dtype)
        self.logger = AICONLogger()
        self.env: GazeFixEnv = self.define_env()
        self.REs: Dict[str, RecursiveEstimator] = self.define_estimators()
        self.MMs: Dict[str, MeasurementModel] = self.define_measurement_models()
        self.AIs: Dict[str, ActiveInterconnection] = self.define_active_interconnections()
        self.set_observations()
        self.connect_states()
        self.goals: Dict[str, Goal] = self.define_goals()
        self.last_action: torch.Tensor = torch.tensor([0.0, 0.0, 0.0])
        self.run_number = 0

    def run(self, timesteps, env_seed=0, initial_action=None, render=True, prints=0, step_by_step=True, record_path:str=None):
        """
        Runs AICON on the environment for a given number of timesteps
        Args:
            timesteps:    number of timesteps to run
            env_seed:     seed for the environment
            render:       render the environment
            prints:       print the states every "prints" timesteps
            step_by_step: wait for user input after each step
        """
        assert self.env is not None, "Environment not set"
        assert self.REs is not None, "Estimators not set"
        assert self.MMs is not None, "Measurement Models not set"
        assert self.AIs is not None, "Active Interconnections not set"
        assert self.goals is not None, "Goals not set"
        self.run_number += 1
        print(f"==================== AICON RUN NUMBER: {self.run_number} ======================")
        self.reset(seed=env_seed, video_path=record_path+f"/records/run{self.run_number}.mp4" if record_path is not None else None)
        if prints > 0:
            print(f"============================ Initial State ================================")
            self.print_estimators()
        if initial_action is not None:
            self.last_action = initial_action
        if render:
            self.render()
        for step in range(timesteps):
            action_gradients = self.compute_action_gradients()
            action = self.compute_action(action_gradients)
            if prints > 0 and step % prints == 0:
                print("Action: ", end=""), self.print_vector(action)
            if record_path is not None:
                self.logger.log(
                    step = step,
                    time = self.env.time,
                    estimators = self.get_buffer_dict(),
                    env_state = self.env.get_state(),
                    observation = {key: {"measurement": self.last_observation[key], "noise": (self.observation_noise[key])} for key in self.last_observation.keys()},
                    goal_loss = {key: goal.loss_function_from_buffer(self.get_buffer_dict()) for key, goal in self.goals.items()},
                )
            if step_by_step:
                input("Press Enter to continue...")
            self.step(action)
            if render:
                self.render()
            step += 1
            if prints > 0 and step % prints == 0:
                print(f"============================ Step {step} ================================")
                self.print_estimators()
        self.env.reset()

    def step(self, action):
        if action[:2].norm() > 1.0:
            action[:2] = action[:2] / action[:2].norm()
        if torch.abs(action[2]) > 1.0:
            action[2] = action[2]/torch.abs(action[2])
        self.last_action = action
        #print(f"-------- Action: ", end=""), self.print_vector(action, trail=" --------")
        self.env.step(np.array(action.cpu()))
        for obs in self.obs.values():
            obs.updated = False
        buffers = self.eval_step(action, new_step=True)
        for key, buffer_dict in buffers.items():
            self.REs[key].load_state_dict(buffer_dict) if key in self.REs.keys() else None

    def set_observations(self):
        self.obs: Dict[str, Observation] = {}
        required_observations = [key for key in [obs for mm in self.MMs.values() for obs in mm.required_states] + [obs for ai in self.AIs.values() for obs in ai.required_states] if key in self.env.observations.keys()]
        self.observation_noise = {key: (self.env.observation_noise[key] if key in self.env.observation_noise.keys() else 0.0) for key in required_observations}
        self.env.required_observations = required_observations
        for key in required_observations:
            self.obs[key] = Observation(key, 1, self.device)

    def connect_states(self):
        for ai in self.AIs.values():
            ai.set_connected_states([self.REs[estimator_id] for estimator_id in ai.required_states])
        for mm in self.MMs.values():
            mm.set_connected_states([self.obs[obs_id] for obs_id in mm.required_states])

    def update_observations(self):
        self.last_observation: dict = self.env.get_observation()
        for key, value in self.last_observation.items():
            if value is not None:
                self.obs[key].set_observation(
                    obs = torch.tensor([value]),
                    obs_cov = torch.tensor([[self.observation_noise[key] if key in self.observation_noise.keys() else 0.0]]),
                    time = self.env.time
                )

    def _eval_goal_with_aux(self, action: torch.Tensor, goal: Goal):
        buffer_dict = self.eval_step(action, new_step=False)
        loss = goal.loss_function_from_buffer(buffer_dict)
        return loss, loss
    
    def _eval_estimator_with_aux(self, action, estimator_id):
        buffer_dict = self.eval_step(action, new_step=False)
        state = buffer_dict[estimator_id]
        return state, state

    def compute_goal_action_gradient(self, goal):
        action = self.last_action
        jacobian, step_eval = jacrev(
            self._eval_goal_with_aux,
            argnums=0,
            has_aux=True)(action, goal)
        return jacobian
    
    def compute_estimator_action_gradient(self, estimator_id, action):
        jacobian, step_eval = jacrev(
            self._eval_estimator_with_aux,
            argnums=0,
            has_aux=True)(action, estimator_id)
        return jacobian, step_eval

    def compute_action_gradients(self):
        gradients: Dict[str, torch.Tensor] = {}
        for key, goal in self.goals.items():
            gradients[key] = self.compute_goal_action_gradient(goal)
        # print("Action gradients:")
        # for key, gradient in gradients.items():
        #     print(f"{key}: {[f'{x:.3f}' for x in gradient.tolist()]}")
        return gradients

    def get_steepest_gradient(self, gradients: Dict[str, torch.Tensor]):
        steepest_gradient = gradients.values()[0]
        for gradient in gradients.values():
            if gradient.norm() > steepest_gradient.norm():
                steepest_gradient = gradient
        return steepest_gradient

    def print_vector(self, vector: torch.Tensor, name = None, trail = "", use_scientific = False):
        if not use_scientific:
            use_scientific = any((abs(x.item()) < 1e-3 and not x.item() == 0.0) or abs(x.item()) > 1e3 for x in vector)
        if name is not None:
            print(f"{name}: [", end="")
        else:
            print("[", end="")
        for i, x in enumerate(vector):
            if i > 0:
                print(" ", end="")
            if 0 <= x.item():
                print(" ", end="")
            if use_scientific:
                print(f"{x.item():.3e}", end="")
            else:
                print(f"{x.item():.3f}", end="")
            if i < vector.shape[0] - 1:
                print(", ", end="")
        else:
            print("]" + trail)

    def print_matrix(self, matrix: torch.Tensor, name = None):
        use_scientific = any(abs(x.item()) < 1e-3 or abs(x.item()) > 1e3 for x in matrix.flatten())
        for i in range(matrix.shape[0]):
            if i == 0:
                if name is not None:
                    print(f"{name}: [", end="")
                else:
                    print("[", end="")
            else:
                if name is not None:
                    print(" " * (len(name) + 3), end="")
                else:
                    print(" ", end="")
            self.print_vector(matrix[i], trail="," if i < matrix.shape[0] - 1 else "]", use_scientific=use_scientific)

    def print_estimator(self, id, print_cov: bool = False, buffer_dict=None):
        if buffer_dict is None:
            mean = self.REs[id].state_mean if id in self.REs.keys() else self.obs[id].state_mean
        else:
            mean = buffer_dict[id]["state_mean"]
        self.print_vector(mean, id + " Mean" if print_cov else id)
        if print_cov:
            if buffer_dict is None:
                cov = self.REs[id].state_cov if id in self.REs.keys() else self.obs[id].state_cov
            else:
                cov = buffer_dict[id]["state_cov"]
            if type(print_cov) == int and print_cov == 2:
                self.print_vector(cov.diag(), id + " Cov")
            else:
                self.print_matrix(cov, f"{id} Cov")

    def reset(self, seed=None, video_path=None) -> None:
        for estimator in self.REs.values():
            estimator.reset()
        obs, _ = self.env.reset(seed, video_path)
        self.logger.run = self.run_number
        self.update_observations()
        self.custom_reset()

    def meas_updates(self, buffer_dict):
        for model_key, meas_model in self.MMs.items():
            meas_dict = self.get_meas_dict(self.MMs[model_key])
            if len(meas_dict["means"]) == len(meas_model.connected_states):
                self.REs[meas_model.estimator_id].call_update_with_meas_model(meas_model, buffer_dict, meas_dict)
            else:
                #print(f"Missing measurements for {model_key}")
                pass

    @abstractmethod
    def define_env(self) -> BaseEnv:
        """
        MUST be implemented by user. Returns the environment
        """
        raise NotImplementedError
    
    @abstractmethod
    def define_estimators(self) -> Dict[str, RecursiveEstimator]:
        """
        MUST be implemented by user. Returns the estimators
        """
        raise NotImplementedError
    
    @abstractmethod
    def define_measurement_models(self) -> Dict[str, MeasurementModel]:
        """
        MUST be implemented by user. Returns the measurement models
        """
        raise NotImplementedError
    
    @abstractmethod
    def define_active_interconnections(self) -> Dict[str, ActiveInterconnection]:
        """
        MUST be implemented by user. Returns the active interconnections
        """
        raise NotImplementedError
    
    @abstractmethod
    def define_goals(self) -> Dict[str, Goal]:
        """
        MUST be implemented by user. Returns the goals
        """
        raise NotImplementedError

    def eval_step(self, action, new_step = False) -> Dict[str, Dict[str, torch.Tensor]]:
        """
        Evaluates one step of the environment and returns the buffer_dict
        """
        if new_step:
            self.update_observations()
        buffer_dict = {key: estimator.get_buffer_dict() for key, estimator in list(self.REs.items())}

        return self.eval_update(action, new_step, buffer_dict)

    @abstractmethod
    def eval_update(self, action, new_step, buffer_dict) -> Dict[str, Dict[str, torch.Tensor]]:
        """
        MUST be implemented by user. Evaluates one step of the environment and returns the buffer_dict
        """
        raise NotImplementedError
    
    def render(self):
        """
        CAN be implemented by user. Renders the environment
        """
        return self.env.render()

    def custom_reset(self):
        """
        CAN be implemented by user. Custom reset function
        """
        pass

    def compute_action(self, gradients):
        """
        CAN be implemented by user. Computes the action based on the gradients
        """
        return self.last_action - 1.0 * self.get_steepest_gradient(gradients)

    def print_estimators(self):
        """
        CAN be implemented by user, only necessary if "prints" option is used in run()
        """
        for estimator in self.REs.values():
            self.print_estimator(estimator.id, print_cov=True)
    
    def get_meas_dict(self, meas_model: MeasurementModel):
        return {
            "means": {key: obs.state_mean for key, obs in meas_model.connected_states.items() if obs.updated},
            "covs": {key: obs.state_cov for key, obs in meas_model.connected_states.items() if obs.updated}
        }
    
    def get_buffer_dict(self):
        return {key: estimator.get_buffer_dict() for key, estimator in self.REs.items()}
    
class DroneEnvAICON(AICON):
    def __init__(self, vel_control=True, moving_target=False, sensor_angle_deg=360, num_obstacles=0, timestep=0.05, observation_noise={}):
        self.moving_target = moving_target
        self.vel_control = vel_control
        self.sensor_angle_deg = sensor_angle_deg
        self.num_obstacles = num_obstacles
        self.timestep = timestep
        self.config_observation_noise = observation_noise
        super().__init__()

    def define_env(self):
        config = 'environment/env_config.yaml'
        with open(config) as file:
            self.env_config = yaml.load(file, Loader=yaml.FullLoader)
            self.env_config["num_obstacles"] = self.num_obstacles
            self.env_config["action_mode"] = 3 if self.vel_control else 1
            self.env_config["moving_target"] = self.moving_target
            self.env_config["robot_sensor_angle"] = self.sensor_angle_deg / 180 * np.pi
            self.env_config["timestep"] = self.timestep
            self.env_config["observation_noise"] = self.config_observation_noise
        return GazeFixEnv(self.env_config)

    def convert_polar_to_cartesian_state(self, polar_mean, polar_cov):
        mean = torch.stack([
            polar_mean[0] * torch.cos(polar_mean[1]),
            polar_mean[0] * torch.sin(polar_mean[1])
        ])
        cov = torch.zeros((2, 2), device=self.device)
        cov[0, 0] = polar_cov[0, 0] * torch.cos(polar_mean[1])**2 + polar_cov[1, 1] * torch.sin(polar_mean[1])**2
        cov[1, 1] = polar_cov[0, 0] * torch.sin(polar_mean[1])**2 + polar_cov[1, 1] * torch.cos(polar_mean[1])**2
        cov[0, 1] = (polar_cov[0, 0] - polar_cov[1, 1]) * torch.cos(polar_mean[1]) * torch.sin(polar_mean[1])
        cov[1, 0] = cov[0, 1]
        return mean, cov
    
    def get_control_input(self, action):
        env_action = torch.empty_like(action)
        env_action[:2] = (action[:2] / action[:2].norm() if action[:2].norm() > 1.0 else action[:2]) * (self.env.robot.max_vel if self.vel_control else self.env.robot.max_acc)
        env_action[2] = action[2] * (self.env.robot.max_vel_rot if self.vel_control else self.env.robot.max_acc_rot)
        return torch.concat([torch.tensor([0.05], device=self.device), env_action])
    
    def render(self):
        target_mean, target_cov = self.convert_polar_to_cartesian_state(self.REs["PolarTargetPos"].state_mean, self.REs["PolarTargetPos"].state_cov)
        estimator_means: Dict[str, torch.Tensor] = {"PolarTargetPos": target_mean}
        estimator_covs: Dict[str, torch.Tensor] = {"PolarTargetPos": target_cov}
        return self.env.render(1.0, {key: np.array(mean.cpu()) for key, mean in estimator_means.items()}, {key: np.array(cov.cpu()) for key, cov in estimator_covs.items()})
    

    def visualize_graph(self, save_path=None, show:bool=False):
        G = nx.DiGraph()

        pos = {}
        ai_nodes=[]
        estimator_nodes=[]
        measurement_model_nodes=[]
        observation_nodes=[]

        # Add nodes and edges for Active Interconnections
        for ai_key, ai in self.AIs.items():
            ai_node = f"AI_{ai_key}"
            ai_nodes.append(ai_node)
            G.add_node(ai_node, shape='o', color='red')
            for estimator in ai.connected_states.values():
                estimator_node = f"RE_{estimator.id}"
                if estimator_node not in G:
                    G.add_node(estimator_node, shape='s', color='blue')
                    pos[estimator_node] = (len(estimator_nodes), 2)
                    estimator_nodes.append(estimator_node)
                G.add_edge(ai_node, estimator_node)
            pos[ai_node] = (sum([pos[f'RE_{est}'][0] for est in [state.id for state in ai.connected_states.values()]])/len(ai.connected_states), 3)

        # Add nodes and edges for Measurement Models
        for mm_key, mm in self.MMs.items():
            mm_node = f"MM_{mm_key}"
            G.add_node(mm_node, shape='o', color='green')
            measurement_model_nodes.append(mm_node)
            estimator_node = f"RE_{mm.estimator_id}"
            if estimator_node not in G:
                G.add_node(estimator_node, shape='s', color='blue')
                estimator_nodes.append(estimator_node)
            pos[mm_node] = (pos[estimator_node][0], 1)
            G.add_edge(mm_node, estimator_node)
            for observation in mm.connected_states.values():
                observation_node = f"OBS_{observation.id}"
                if observation_node not in G:
                    G.add_node(observation_node, shape='^', color='orange')
                    observation_nodes.append(observation_node)
                G.add_edge(observation_node, mm_node)
        
        # set x spacing for observation nodes
        min = 0
        max = 0
        for p in pos.values():
            if p[0] > max:
                max = p[0]
        for i,obs in enumerate(observation_nodes):
            pos[obs] = ((min+max-len(observation_nodes))/2+i, 0)

        shapes = nx.get_node_attributes(G, 'shape')
        colors = nx.get_node_attributes(G, 'color')

        for shape in set(shapes.values()):
            nx.draw_networkx_nodes(G, pos, nodelist=[sNode for sNode in shapes if shapes[sNode] == shape], node_shape=shape, node_color=[colors[sNode] for sNode in shapes if shapes[sNode] == shape], node_size=500)
        nx.draw_networkx_edges(G, pos)
        nx.draw_networkx_labels(G, pos, font_size=8)
        if save_path is not None:
            if not save_path.endswith('/'):
                    save_path += '/'
            plt.savefig(save_path + "aicon_graph.png")
        if show:
            plt.show()
            input("Press Enter to continue...")