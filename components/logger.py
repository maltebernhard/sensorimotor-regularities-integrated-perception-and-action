import time
from typing import Dict, List, Tuple
import matplotlib.pyplot as plt
import numpy as np
import torch
import pickle
import os

from components.helpers import rotate_vector_2d

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
        self.task_variation: Tuple[int,int,int,int] = None
        self.aicon_type: int = None
        self.config = (None, None)
        self.data: Dict[Tuple[int,Tuple[int,int,int,int]], Dict[int,dict]] = {
            "records": {},
            "configs": {
                "aicon_types":          {},
                "sensor_noises":        {},
                "target_movements":     {},
                "observation_losses":   {},
                "fv_noises": {},
            },
        }
        self.run = 0

    def get_config_id(self, config_id: str, target_value):
        config_values = self.data["configs"][config_id]
        if len(config_values) > 0:
            if target_value not in config_values.values():
                new_id = max([int_key for int_key in config_values.keys()]) + 1
                config_values[new_id] = target_value
                #print(f"Couldn't find existing config for {config_id}")
                return new_id
            else:
                #print(f"Found existing config for {config_id}")
                return [key_int for key_int, value in config_values.items() if value == target_value][0]
        else:
            config_values[1] = target_value
            #print(f"Couldn't find existing config for {config_id}")
            return 1

    def set_config(self, smcs: list[str], control: bool, distance_sensor: bool, sensor_noise: Dict[str,float], moving_target: str, observation_loss: Dict[str,float], fv_noise: Dict[str,float]):
        # aicon_type = {
        #     "smcs":            smcs,
        #     "control":         control,
        #     "distance_sensor": distance_sensor,
        # }
        aicon_type = {
            "SMCs":            smcs,
            "Control":         control,
            "DistanceSensor": distance_sensor,
        }
        self.aicon_type = self.get_config_id("aicon_types", aicon_type)
        self.sensor_noise = self.get_config_id("sensor_noises", sensor_noise)
        self.target_movement = self.get_config_id("target_movements", moving_target)
        self.observation_loss = self.get_config_id("observation_losses", observation_loss)
        self.fv_noise = self.get_config_id("fv_noises", fv_noise)

        self.config = (self.aicon_type, (self.sensor_noise, self.target_movement, self.observation_loss, self.fv_noise))
        if self.config not in self.data["records"]:
            self.data["records"][self.config] = {}
        self.current_data = self.data["records"][self.config]

    # ======================================== logging ==========================================

    def log(self, step: int, time: float, estimators: Dict[str,Dict[str,torch.Tensor]], env_state: Dict[str,float], observation: Dict[str,Dict[str,float]], goal_loss: Dict[str,Dict[str,torch.Tensor]], action: torch.Tensor, gradients: Dict[str,Dict[str,torch.Tensor]]):
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
                "goal_loss": {goal_key: {
                        subgoal_key: []     # goal loss function value
                    for subgoal_key in goal_loss[goal_key].keys()}
                for goal_key in goal_loss.keys()},
                "gradient": {goal_key: {
                        subgoal_key: []     # gradient of goal loss function
                    for subgoal_key in gradients[goal_key].keys()}
                for goal_key in gradients.keys()},
                "action":           [],     # action
                "desired_distance": env_state["desired_target_distance"],
                # TODO: log run seed and config Analysis to skip runs which are already logged
            }

        # extract real state from env_state, matching to estimator structure
        real_state = {}
        for key in estimators.keys():
            if key[:5] == "Polar" and key[-3:] == "Pos":
                obj = key[5:-9].lower() if "Global" in key else key[5:-3].lower()
                real_state[key] = np.array([
                    env_state[f"{obj}_distance"],
                    env_state[f"{obj}_offset_angle"],
                    env_state[f"{obj}_visual_angle"],
                    #env_state[f"{obj}_offset_angle_dot"]
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
            # log uncertainty
            # TODO: remove this hack once noise is always on
            estimator_cov[estimator_cov < 0] = 0
            self.current_data[self.run]["estimators"][state_key]["uncertainty"].append(np.sqrt(np.diag(estimator_cov)))
            # log env_state
            self.current_data[self.run]["env_state"][state_key].append(real_state[state_key])
            task_state = real_state[state_key]
            # if state_key == "PolarTargetGlobalPos":
            #     # NOTE: this is only valid IF target moves and IF the estimator represents global target velocity
            #     # TODO: implement for decoupled PolarTargetDistance and PolarTargetAngle, IF I plan to use them
            #     rtf_vel = rotate_vector_2d(-task_state[2], real_state["RobotVel"][:2])
            #     task_state[2] += rtf_vel[0]
            #     task_state[3] += real_state["RobotVel"][2]
            self.current_data[self.run]["estimation_error"][state_key].append(task_state - estimator_mean)
            if state_key in ["PolarTargetPos", "PolarTargetGlobalPos", "PolarTargetDistance"]:
                task_state[0] -= self.current_data[self.run]["desired_distance"]
            self.current_data[self.run]["task_state"][state_key].append(task_state)
        for obs_key in observation.keys():
            self.current_data[self.run]["observation"][obs_key]["measurement"].append(observation[obs_key]["measurement"])
            self.current_data[self.run]["observation"][obs_key]["noise"].append(observation[obs_key]["noise"])
        for goal_key in goal_loss.keys():
            for subgoal_key in goal_loss[goal_key].keys():
                self.current_data[self.run]["goal_loss"][goal_key][subgoal_key].append(np.array(goal_loss[goal_key][subgoal_key].item()))
        for goal_key in gradients.keys():
            for subgoal_key in gradients[goal_key].keys():
                self.current_data[self.run]["gradient"][goal_key][subgoal_key].append(np.array(gradients[goal_key][subgoal_key].tolist()))
        self.current_data[self.run]["action"].append(np.array(action.tolist()))

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

    def plot_mean_stddev(self, subplot: plt.Axes, error_means: np.ndarray, error_stddevs: np.ndarray, title:str=None, data_label:str=None, y_label:str=None, x_label:str=None, legend=False):
        subplot.plot(error_means, label=data_label)
        subplot.fill_between(
            range(len(error_means)), 
            error_means - error_stddevs,
            error_means + error_stddevs,
            alpha=0.2,
            #label='Stddev'
        )
        if title is not None:
            subplot.set_title(title, fontsize=16)
        if x_label is not None:
            subplot.set_xlabel(x_label)
        if y_label is not None:
            subplot.set_ylabel(y_label, fontsize=16)
        subplot.grid(True)
        subplot.legend() if legend else None

    def plot_runs(self, subplot: plt.Axes, data:List[np.ndarray], labels:List[int], state_index:int=None, title:str=None, y_label:str=None, x_label:str=None, legend=False):
        if state_index is None:
            for i, run in enumerate(data):
                subplot.plot(run[:], label=f'{labels[i]}')
        else:
            for i, run in enumerate(data):
                subplot.plot(run[:, state_index], label=f'{labels[i]}')
        if title is not None:
            subplot.set_title(title)
        if x_label is not None:
            subplot.set_xlabel(x_label)
        if y_label is not None:
            subplot.set_ylabel(y_label)
        subplot.grid(True)
        subplot.legend() if legend else None

    def save_fig(self, fig:plt.Figure, save_path:str=None, show:bool=False):
        if save_path is not None:
            os.makedirs(os.path.dirname(save_path), exist_ok=True)
            fig.tight_layout()
            time.sleep(0.3)
            print(f"Saving to {save_path}...")
            fig.savefig(save_path)
            plt.close(fig)
        if show:
            fig.show()
            input("Press Enter to continue...")
        plt.close('all')

    def plot_states(self, plotting_config:Dict[str,Dict[str,Tuple[List[int],List[str],List[Tuple[float,float]]]]], save_path:str=None, show:bool=False):
        for state_id, config in plotting_config["states"].items():
            indices = np.array(config["indices"])
            labels = config["labels"]
            ybounds = config["ybounds"]
            if len(indices) == 1:
                fig, axs = plt.subplots(len(indices), 3, figsize=(21, 6))
                axs = [[ax] for ax in axs]
            else:
                fig, axs = plt.subplots(3, len(indices), figsize=(7 * len(indices), 18))

            for label, config in plotting_config["axes"].items():
                self.set_config(**config)

                states = np.array([np.array(run["task_state"][state_id])[:,indices] for run in self.current_data.values()])
                ucttys = np.array([np.array(run["estimators"][state_id]["uncertainty"])[:,indices] for run in self.current_data.values()])
                errors = np.array([np.array(run["estimation_error"][state_id])[:,indices] for run in self.current_data.values()])

                state_means, state_stddevs = self.compute_mean_and_stddev(states)
                error_means, error_stddevs = self.compute_mean_and_stddev(errors)
                uctty_means, uctty_stddevs = self.compute_mean_and_stddev(ucttys)
                for i in range(len(indices)):
                    self.plot_mean_stddev(axs[0][i], state_means[:, i], state_stddevs[:, i], None, label, f"{labels[i]} State" if i==0 else None, "Time Step", True)
                    self.plot_mean_stddev(axs[1][i], error_means[:, i], error_stddevs[:, i], None, label, f"{labels[i]} Estimation Error" if i==0 else None, "Time Step", True)
                    self.plot_mean_stddev(axs[2][i], uctty_means[:, i], uctty_stddevs[:, i], None, label, f"{labels[i]} Estimation Uncertainty" if i==0 else None, "Time Step", True)    
                    axs[0][i].set_ylim(ybounds[0][i])
                    axs[1][i].set_ylim(ybounds[1][i])
                    axs[2][i].set_ylim(ybounds[2][i])
            # save / show
            path = os.path.join(save_path, f"records/state_{plotting_config['name']}.png") if save_path is not None else None
            self.save_fig(fig, path, show)

    def plot_state_runs(self, plotting_config:Dict[str,Dict[str,Tuple[List[int],List[str],List[Tuple[float,float]]]]], axs_id: str, runs: list[int], save_path:str=None, show:bool=False):
        for state_id, config in plotting_config["states"].items():
            indices = np.array(config["indices"])
            labels = config["labels"]
            ybounds = config["ybounds"]
            fig, axs = plt.subplots(3, len(indices), figsize=(7 * len(indices), 18))
            if len(indices) == 1:
                axs = [[ax] for ax in axs]
            config = plotting_config["axes"][axs_id]
            self.set_config(**config)
            if runs is None:
                runs = list(self.current_data.keys())
            states = np.array([np.array(self.current_data[run]["task_state"][state_id])[:,indices] for run in runs])
            ucttys = np.array([np.array(self.current_data[run]["estimators"][state_id]["uncertainty"])[:,indices] for run in runs])
            errors = np.array([np.array(self.current_data[run]["estimation_error"][state_id])[:,indices] for run in runs])
            for i in range(len(indices)):
                self.plot_runs(axs[0][i], states, runs, i, f"{state_id} {labels[i]}", "State" if i==0 else None, None, True)
                self.plot_runs(axs[1][i], errors, runs, i, None, "Estimation Error" if i==0 else None, None, True)
                self.plot_runs(axs[2][i], ucttys, runs, i, None, "Estimation Uncertainty" if i==0 else None, "Time Step", True)
                axs[0][i].set_ylim(ybounds[0][i])
                axs[1][i].set_ylim(ybounds[1][i])
                axs[2][i].set_ylim(ybounds[2][i])
            # save / show
            path = os.path.join(save_path, f"records/state_runs_{axs_id}.png") if save_path is not None else None
            self.save_fig(fig, path, show)

    def plot_goal_losses(self, plotting_config:Dict[str,Dict[str,Tuple[List[int],List[str],List[Tuple[float,float]]]]], plot_subgoals:bool, save_path:str=None, show:bool=False):
        num_goals = len(plotting_config["goals"])
        # Plot goal losses
        fig_goal, axs_goal = plt.subplots(1, num_goals, figsize=(12, 10*num_goals))
        if num_goals == 1:
            axs_goal = [axs_goal]
        for i, (goal_key, config) in enumerate(plotting_config["goals"].items()):
            ybounds = config["ybounds"]
            for label, config in plotting_config["axes"].items():
                self.set_config(**config)
                total_loss = np.array([[0.0]*len(run["step"]) for run in self.current_data.values()])
                for subgoal_key in self.current_data[1]["goal_loss"][goal_key].keys():
                    goal_losses = np.array([run["goal_loss"][goal_key][subgoal_key] for run in self.current_data.values()])
                    total_loss += goal_losses
                    if plot_subgoals:
                        goal_loss_means, goal_loss_stddevs = self.compute_mean_and_stddev(goal_losses)
                        self.plot_mean_stddev(axs_goal[i], goal_loss_means, goal_loss_stddevs, f"{goal_key} loss", f"{label} {subgoal_key} loss", "Loss Mean and Stddev", "Timestep", True)
                total_loss_means, total_loss_stddevs = self.compute_mean_and_stddev(total_loss)
                self.plot_mean_stddev(axs_goal[i], total_loss_means, total_loss_stddevs, f"{goal_key} loss", f"{label} total loss", "Loss Mean and Stddev", "Timestep", True)
                axs_goal[i].set_ylim(ybounds[0], ybounds[1])
        # save / show
        loss_path = os.path.join(save_path, f"records/loss_{plotting_config['name']}.png") if save_path is not None else None
        self.save_fig(fig_goal, loss_path, show)

    # ================================ saving & loading ================================

    def save(self, record_dir):
        self.data["records"] = self.convert_to_numpy(self.data["records"])
        with open(os.path.join(record_dir, "records/data.pkl"), 'wb') as f:
            pickle.dump(self.data, f)

    def load(self, folder):
        with open(os.path.join(folder, 'records/data.pkl'), 'rb') as f:
            self.data = pickle.load(f)
    