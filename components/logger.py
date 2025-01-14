from typing import Dict, List, Tuple
import matplotlib.pyplot as plt
import numpy as np
import torch
import yaml
import pickle
import os

from components.helpers import rotate_vector_2d
from tqdm import tqdm

# ==================================================================================

class AICONLogger:
    def __init__(self):
        """
        Initializes the logger with default values for task variation, AICON type, and data storage.

        Attributes:
            task_variation (str): The type of task variation. Possible values include:
                - "SensorNoise"
                    - "Lvl0"
                    - "Lvl1"
                    - "Lvl2"
                - "TargetMovement"
                    - "None"
                    - "RunAway"
                    - "Sine"
                    - "Linear"
                - "ObservationLoss"
                    - "0Sec"
                    - "3Sec"
                    - "5Sec"
            aicon_type (str): The type of AICON. Possible values:
                - "Control"
                - "Goal"
                - "FovealVision"
                - "Interconnection"
                - "Divergence"
            data (Dict[int, dict]): A dictionary to store data, where the key is an integer and the value is another dictionary.
            run (int): The current run number, initialized to 0.
        """
        self.task_variation: Tuple[int,int,int] = None
        self.aicon_type: int = None
        self.config = (None, None)
        self.data: Dict[Tuple[int,Tuple[int,int,int]], Dict[int,dict]] = {
            "records": {},
            "configs": {
                "aicon_types":        {},
                "sensor_noises":      {},
                "target_movements":   {},
                "observation_losses": {},
            },
        }
        self.run = 0

    def assign_id(self, target_dict: Dict[str,str], target_value):
        if len(target_dict) > 0:
            if target_value not in target_dict.values():
                new_id = max([key_int for key_int in target_dict.keys()]) + 1
                target_dict[new_id] = target_value
                return new_id
            else:
                return [key_int for key_int, value in target_dict.items() if value == target_value][0]
        else:
            target_dict[0] = target_value
            return 1

    def set_config(self, aicon_type: str, sensor_noise: Dict[str,float], target_movement: str, observation_loss: Dict[str,float]):
        self.aicon_type = self.assign_id(self.data["configs"]["aicon_types"], aicon_type)
        self.sensor_noise = self.assign_id(self.data["configs"]["sensor_noises"], sensor_noise)
        self.target_movement = self.assign_id(self.data["configs"]["target_movements"], target_movement)
        self.observation_loss = self.assign_id(self.data["configs"]["observation_losses"], observation_loss)

        self.config = (self.aicon_type, (self.sensor_noise, self.target_movement, self.observation_loss))
        self.data["records"][self.config] = {}
        self.current_data = self.data["records"][self.config]

    # ======================================== logging ==========================================

    def log(self, step: int, time: float, estimators: Dict[str,Dict[str,torch.Tensor]], env_state: Dict[str,float], observation: Dict[str,Dict[str,float]], goal_loss: Dict[str,torch.Tensor]):
        # for new run, set up logging dict
        if self.run not in self.current_data:
            self.current_data[self.run] = {
                "step":             [],     # time step
                "time":             [],     # time in seconds
                "estimators": {estimator_key: {
                    "state_mean":   [],     # mean of state estimate
                    "state_cov":    [],     # covariance of state estimate
                    "uncertainty":  [],     # uncertainty of state estimate: sqrt(diag(covariance))
                    "motion_noise": np.array(estimators[estimator_key]["forward_noise"].tolist()),     # static estimator motion noise
                } for estimator_key in estimators.keys()},
                "observation": {obs_key: {
                    "measurement":  [],     # measurement
                    "noise":        [],     # noise
                } for obs_key in observation.keys()},
                "env_state": {
                    state_key:      []      # real state
                for state_key in estimators.keys()},
                "task_state": {
                    state_key:      []      # task state: real state with subtracted offsets (like desired target distance)
                for state_key in estimators.keys()},
                "estimation_error": {
                    state_key:      []      # estimation error: estimation - real state
                for state_key in estimators.keys()},
                "goal_loss": {
                    goal_key:       []      # goal loss function value
                for goal_key in goal_loss.keys()},
                "desired_distance": env_state["desired_target_distance"],
            }

        # extract real state from env_state, matching to estimator structure
        real_state = {}
        for key in estimators.keys():
            if key[:5] == "Polar" and key[-3:] == "Pos":
                obj = key[5:-9].lower() if "Global" in key else key[5:-3].lower()
                real_state[key] = np.array([
                    env_state[f"{obj}_distance"],
                    env_state[f"{obj}_offset_angle"],
                    env_state[f"{obj}_distance_dot"],
                    env_state[f"{obj}_offset_angle_dot"]
                ])
            elif key[:5] == "Polar" and key[-9:] == "PosRadius":
                obj = key[5:-9].lower()
                real_state[key] = np.array([
                    env_state[f"{obj}_distance"],
                    env_state[f"{obj}_offset_angle"],
                    env_state[f"{obj}_distance_dot"],
                    env_state[f"{obj}_offset_angle_dot"],
                    env_state[f"{obj}_radius"]
                ])
            elif key == "RobotVel":
                real_state[key] = np.array([
                    env_state["vel_frontal"],
                    env_state["vel_lateral"],
                    env_state["vel_rot"],
                ])
            elif key[-6:] == "Radius":
                obj = key[:-6].lower()
                real_state[key] = np.array([
                    env_state[f"{obj}_radius"]
                ])

        # log data
        self.current_data[self.run]["step"].append(step)
        self.current_data[self.run]["time"].append(time)
        for state_key in estimators.keys():
            # log estimator values
            estimator_mean = np.array(estimators[state_key]["state_mean"].tolist())
            estimator_cov = np.array(estimators[state_key]["state_cov"].tolist())
            self.current_data[self.run]["estimators"][state_key]["state_mean"].append(estimator_mean)
            self.current_data[self.run]["estimators"][state_key]["state_cov"].append(estimator_cov)
            # TODO: remove this hack once noise is always on
            estimator_cov[estimator_cov < 0] = 0
            self.current_data[self.run]["estimators"][state_key]["uncertainty"].append(np.sqrt(np.diag(estimator_cov)))
            # log env_state
            self.current_data[self.run]["env_state"][state_key].append(real_state[state_key])
            task_state = real_state[state_key]
            if state_key == "PolarTargetGlobalPos":
                # NOTE: this is only valid IF target moves and IF the estimator represents global target velocity
                # TODO: implement for decoupled PolarTargetDistance and PolarTargetAngle, IF I plan to use them
                rtf_vel = rotate_vector_2d(task_state[2], real_state["RobotVel"][:2])
                task_state[2] += rtf_vel[0]
                task_state[3] += real_state["RobotVel"][2]
            self.current_data[self.run]["estimation_error"][state_key].append(task_state - estimator_mean)
            if state_key in ["PolarTargetPos", "PolarTargetGlobalPos", "PolarTargetDistance"]:
                task_state[0] -= self.current_data[self.run]["desired_distance"]
            self.current_data[self.run]["task_state"][state_key].append(task_state)
        for obs_key in observation.keys():
            self.current_data[self.run]["observation"][obs_key]["measurement"].append(observation[obs_key]["measurement"])
            self.current_data[self.run]["observation"][obs_key]["noise"].append(observation[obs_key]["noise"])
        for goal_key in goal_loss.keys():
            self.current_data[self.run]["goal_loss"][goal_key].append(np.array(goal_loss[goal_key].tolist()))

    # ======================================= plotting ==========================================

    @staticmethod
    def convert_to_numpy(obj):
        if isinstance(obj, list):
            return np.array(obj)
        elif isinstance(obj, dict):
            return {k: AICONLogger.convert_to_numpy(v) for k, v in obj.items()}
        else:
            return obj

    @staticmethod
    def compute_mean_and_stddev(data) -> Tuple[np.ndarray, np.ndarray]:
        data_array = np.array(data)
        mean = np.mean(data_array, axis=0)
        stddev = np.std(data_array, axis=0)
        return mean, stddev

    @staticmethod
    def create_subplots(num_subplots: int):
        if num_subplots > 1:
            fig, axs = plt.subplots((num_subplots + 1) // 2, 2, figsize=(14, 6 * ((num_subplots + 1) // 2)))
            axs = axs.flatten()
        else:
            fig, axs = plt.subplots(1, 1, figsize=(7, 6))
            axs = [axs]
        return fig, axs

    def plot_mean_stddev(self, subplot: plt.Axes, error_means: np.ndarray, error_stddevs: np.ndarray, title:str=None, y_label:str=None, x_label:str=None, legend=False):
        subplot.plot(error_means, label='Mean')
        subplot.fill_between(
            range(len(error_means)), 
            error_means - error_stddevs,
            error_means + error_stddevs,
            alpha=0.2, label='Stddev'
        )
        if title is not None:
            subplot.set_title(title)
        if x_label is not None:
            subplot.set_xlabel(x_label)
        if y_label is not None:
            subplot.set_ylabel(y_label)
        subplot.grid()
        subplot.legend() if legend else None

    def plot_runs(self, subplot: plt.Axes, data:List[np.ndarray], state_index:int=None, title:str=None, y_label:str=None, x_label:str=None, legend=False):
        if state_index is None:
            for i, run in enumerate(data):
                subplot.plot(run[:], label=f'{i}')
        else:
            for i, run in enumerate(data):
                subplot.plot(run[:, state_index], label=f'{i+1}')
        if title is not None:
            subplot.set_title(title)
        if x_label is not None:
            subplot.set_xlabel(x_label)
        if y_label is not None:
            subplot.set_ylabel(y_label)
        subplot.grid()
        subplot.legend() if legend else None

    def save_fig(self, fig:plt.Figure, save_path:str=None, show:bool=False):
        if save_path is not None:
            os.makedirs(os.path.dirname(save_path), exist_ok=True)
            fig.savefig(save_path)
        if show:
            fig.show()
            input("Press Enter to continue...")
        plt.close('all')

    def plot_states(self, state_dict:Dict[str,Tuple[List[int],List[str]]], save_path:str=None, show:bool=False):
        figure = 0
        for state_id, config in state_dict.items():
            indices = np.array(config[0])
            labels = config[1]

            states = np.array([np.array(run["task_state"][state_id])[:,indices] for run in self.current_data.values()])
            ucttys = np.array([np.array(run["estimators"][state_id]["uncertainty"])[:,indices] for run in self.current_data.values()])
            errors = np.array([np.array(run["estimation_error"][state_id])[:,indices] for run in self.current_data.values()])

            # PLOT:
            # - state (goal error)
            # - estimation error: estimation - state
            # - uncertainty
            figure += 1
            fig_avg, axs_avg = plt.subplots(3, len(indices), figsize=(7 * len(indices), 18))
            fig_runs, axs_runs = plt.subplots(3, len(indices), figsize=(7 * len(indices), 18))
            
            state_means, state_stddevs = self.compute_mean_and_stddev(states)
            error_means, error_stddevs = self.compute_mean_and_stddev(np.array(errors))
            uctty_means, uctty_stddevs = self.compute_mean_and_stddev(np.array(ucttys))
            for i in range(len(indices)):
                self.plot_mean_stddev(axs_avg[0][i], state_means[:, i], state_stddevs[:, i], f"{state_id} {labels[i]}", "State" if i==0 else None, None, i==0)
                self.plot_mean_stddev(axs_avg[1][i], error_means[:, i], error_stddevs[:, i], None, "Estimation Error" if i==0 else None, None)
                self.plot_mean_stddev(axs_avg[2][i], uctty_means[:, i], uctty_stddevs[:, i], None, "Estimation Uncertainty" if i==0 else None, "Time Step")    
                self.plot_runs(axs_runs[0][i], np.array(states), i, f"{state_id} {labels[i]}", "State" if i==0 else None, None, i==0)
                self.plot_runs(axs_runs[1][i], np.array(errors), i, None, "Estimation Error" if i==0 else None, None)
                self.plot_runs(axs_runs[2][i], np.array(ucttys), i, None, "Estimation Uncertainty" if i==0 else None, "Time Step")
                if state_id in ["PolarTargetPos", "PolarTargetGlobalPos"]:
                    if indices[i] in [0,2]:
                        axs_avg[2][i].set_ylim(0, 10)
                        axs_runs[2][i].set_ylim(0, 10)
                    elif indices[i] in [1,3]:
                        axs_avg[2][i].set_ylim(0, 0.5)
                        axs_runs[2][i].set_ylim(0, 0.5)
            axs_runs[0][0].legend()

            # save / show
            avg_path = os.path.join(save_path, f"records/{state_id}_avg.png") if save_path is not None else None
            runs_path = os.path.join(save_path, f"records/{state_id}_runs.png") if save_path is not None else None
            self.save_fig(fig_avg, avg_path, show)
            self.save_fig(fig_runs, runs_path, show)

    def plot_goal_losses(self, save_path:str=None, show:bool=False):
        # Plot goal losses
        goal_losses = {goal_key: np.array([run["goal_loss"][goal_key] for run in self.current_data.values()]) for goal_key in self.current_data[1]["goal_loss"].keys()}
        fig_goal, axs_goal = plt.subplots(2, len(goal_losses), figsize=(12, 10*len(goal_losses)))
        if len(goal_losses) == 1:
            axs_goal = [[axs_goal[0]], [axs_goal[1]]]
        for i, (goal_key, losses) in enumerate(goal_losses.items()):
            goal_loss_means, goal_loss_stddevs = self.compute_mean_and_stddev(losses)
            self.plot_mean_stddev(axs_goal[0][i], goal_loss_means, goal_loss_stddevs, f"Goal Loss for {goal_key}", "Loss Mean and Stddev", None)
            self.plot_runs(axs_goal[1][i], losses, None, None, "Loss", "Time Step")
            axs_goal[0][i].set_ylim(0, 500)
            axs_goal[1][i].set_ylim(0, 500)
        # save / show
        loss_path = os.path.join(save_path, "records/goal_losses.png") if save_path is not None else None
        self.save_fig(fig_goal, loss_path, show)
        

    # ================================ saving & loading ================================

    def save(self, record_dir):
        self.data = self.convert_to_numpy(self.data)
        with open(os.path.join(record_dir, "records/data.pkl"), 'wb') as f:
            pickle.dump(self.data, f)

    def load(self, folder):
        with open(os.path.join(folder, 'records/data.pkl'), 'rb') as f:
            self.data = pickle.load(f)
    