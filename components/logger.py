import pickle
import time
from typing import Dict, List, Tuple
from matplotlib import pyplot as plt
import numpy as np
import torch
import os

import wandb
os.environ["WANDB_SILENT"] = "true"

# ==================================================================================

class VariationLogger:
    def __init__(self, variation_config:dict, logging_config:dict):
        self.data = {}
        self.run = len(self.data.keys())
        self.variation_config = variation_config
        self.wandb_project: str = logging_config['wandb_project']
        self.wandb_group: str = logging_config['wandb_group']
        self.wandb_run = None

    def log(self, step: int, time: float, estimators: Dict[str,Dict[str,torch.Tensor]], env_state: Dict[str,float], observation: Dict[str,Dict[str,float]], goal_loss: Dict[str,Dict[str,torch.Tensor]], action: torch.Tensor, gradients: Dict[str,Dict[str,torch.Tensor]]):
        # for new run, set up logging dict
        if self.run not in self.data:
            self.create_run_dict(estimators, env_state, observation, goal_loss, action, gradients)
            if self.wandb_group is not None:
                self.create_wandb_run()
        # log data
        real_state = self.create_real_state(estimators, env_state)
        if self.wandb_run is not None:
            self.log_wandb(estimators, real_state)
        self.log_local(step, time, estimators, real_state, observation, goal_loss, action, gradients)

    def create_run_dict(self, estimators: Dict[str,Dict[str,torch.Tensor]], env_state: Dict[str,float], observation: Dict[str,Dict[str,float]], goal_loss: Dict[str,Dict[str,torch.Tensor]], action: torch.Tensor, gradients: Dict[str,Dict[str,torch.Tensor]]):
        self.data[self.run] = {
                "step":             np.array([]),     # time step
                "time":             np.array([]),     # time in seconds
                "estimators": {estimator_key: {
                    'mean':         np.empty((0,) + estimators[estimator_key]["mean"].shape),     # mean of state estimate
                    'cov':          np.empty((0,) + estimators[estimator_key]["cov"].shape),      # covariance of state estimate
                    "uncertainty":  np.empty((0,) + (estimators[estimator_key]["mean"].shape[0],)), # uncertainty of state estimate: sqrt(diag(covariance))
                    "motion_noise": np.array(estimators[estimator_key]["forward_noise"].tolist()),     # static estimator motion noise
                } for estimator_key in estimators.keys()},
                "observation": {obs_key: {
                    "measurement":  np.array([]),     # measurement
                    "noise":        np.array([]),           # noise
                } for obs_key in observation.keys()},
                "env_state": {
                    state_key:      np.empty((0,) + estimators[state_key]["mean"].shape)      # real state
                for state_key in estimators.keys()},
                "task_state": {
                    state_key:      np.empty((0,) + estimators[state_key]["mean"].shape)      # task state: real state with subtracted offsets (like desired target distance)
                for state_key in estimators.keys()},
                "estimation_error": {
                    state_key:      np.empty((0,) + estimators[state_key]["mean"].shape)      # estimation error: estimation - real state
                for state_key in estimators.keys()},
                "goal_loss": {goal_key: {
                    subgoal_key: np.array([])     # goal loss function value
                    for subgoal_key in goal_loss[goal_key].keys()}
                for goal_key in goal_loss.keys()},
                "gradient": {goal_key: {
                    subgoal_key: np.empty((0,) + gradients[goal_key][subgoal_key].shape)     # gradient of goal loss function
                    for subgoal_key in gradients[goal_key].keys()}
                for goal_key in gradients.keys()},
                "action":           np.empty((0,) + action.shape),     # action
                "desired_distance": env_state["desired_target_distance"],
                # TODO: log run seed and config Analysis to skip runs which are already logged
            }

    def create_wandb_run(self):
        if self.wandb_run is not None:
            self.wandb_run.finish()
        self.wandb_run = wandb.init(
            project=self.wandb_project,
            name=f'{self.variation_config}_{self.run}',
            group=self.wandb_group,
            config = {
                "id": self.variation_config,
                "smcs": self.variation_config["smcs"],
                "control": self.variation_config["control"],
                "distance_sensor": self.variation_config["distance_sensor"],
                "sensor_noise": self.variation_config["sensor_noise"],
                "moving_target": self.variation_config["moving_target"],
                "observation_loss": self.variation_config["observation_loss"],
                "fv_noise": self.variation_config["fv_noise"],
            },
            save_code=False,
        )

    def create_real_state(self, estimators: Dict[str,Dict[str,torch.Tensor]], env_state: Dict[str,float]):
        # extract real state from env_state, matching to estimator structure
        real_state = {}
        for key in estimators.keys():
            if key[:5] == "Polar" and key[-3:] == "Pos":
                obj = key[5:-3].lower()
                real_state[key] = np.array([
                    env_state[f"{obj}_distance"],
                    env_state[f"{obj}_offset_angle"],
                ])
                if obj=='target' and self.variation_config["moving_target"] != "stationary":
                    real_state[key] = np.append(real_state[key], env_state[f"{obj}_distance_dot_global"])
                    real_state[key] = np.append(real_state[key], env_state[f"{obj}_offset_angle_dot_global"])
                real_state[key] = np.append(real_state[key], env_state[f"{obj}_radius"])
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
        return real_state

    def log_wandb(self, estimators: Dict[str,Dict[str,torch.Tensor]], real_state: Dict[str,Dict[str,torch.Tensor]]):
        if self.variation_config["moving_target"] != "stationary":
            index_keys = {
                "PolarTargetPos": ["distance", "angle", "distance_dot", "angle_dot", "radius"],
                "RobotVel": ["frontal", "lateral", "rot"],
            }
        else:
            index_keys = {
                "PolarTargetPos": ["distance", "angle", "radius"],
                "RobotVel": ["frontal", "lateral", "rot"],
            }
        self.wandb_run.log({
            #"step": step,
            #"time": time,
            "estimators": {estimator_key: {
                'mean': {index_keys[estimator_key][i]: val for i, val in enumerate(np.array(estimator['mean'].tolist())[:len(index_keys[estimator_key])])},
                #'cov': estimator['cov'],
                'uncertainty': {index_keys[estimator_key][i]: val for i, val in enumerate(np.sqrt(np.diag(np.array(estimator['cov'].tolist())))[:len(index_keys[estimator_key])])},
                'estimation_error': {index_keys[estimator_key][i]: val for i, val in enumerate(real_state[estimator_key] - np.array(estimator['mean'].tolist())[:len(real_state[estimator_key])])},
                'task_state': {index_keys[estimator_key][i]: val for i, val in enumerate((real_state[estimator_key] - self.data[self.run]["desired_distance"] * np.array([1.0]+[0.0]*(len(real_state[estimator_key])-1))) if estimator_key == "PolarTargetPos" else real_state[estimator_key])},
            } for estimator_key, estimator in estimators.items() if estimator_key != "RobotVel"},
            #"observation": observation,
            #"goal_loss": goal_loss,
            #"gradients": gradients,
            #"action": action,
        })

    def log_local(self, step: int, time: float, estimators: Dict[str,Dict[str,torch.Tensor]], real_state: Dict[str,Dict[str,torch.Tensor]], observation: Dict[str,Dict[str,float]], goal_loss: Dict[str,Dict[str,torch.Tensor]], action: torch.Tensor, gradients: Dict[str,Dict[str,torch.Tensor]]):
        self.data[self.run]["step"] = np.append(self.data[self.run]["step"], step)
        self.data[self.run]["time"] = np.append(self.data[self.run]["time"], time)
        for state_key in estimators.keys():
            # log estimator values
            estimator_mean = np.array(estimators[state_key]['mean'].tolist())
            estimator_cov = np.array(estimators[state_key]['cov'].tolist())
            self.data[self.run]["estimators"][state_key]['mean'] = np.append(self.data[self.run]["estimators"][state_key]['mean'], [estimator_mean], axis=0)
            self.data[self.run]["estimators"][state_key]['cov'] = np.append(self.data[self.run]["estimators"][state_key]['cov'], [estimator_cov], axis=0)
            # log uncertainty
            estimator_cov[estimator_cov < 0] = 0
            self.data[self.run]["estimators"][state_key]["uncertainty"] = np.append(self.data[self.run]["estimators"][state_key]["uncertainty"], [np.sqrt(np.diag(estimator_cov))], axis=0)
            # log env_state
            self.data[self.run]["env_state"][state_key] = np.append(self.data[self.run]["env_state"][state_key], [real_state[state_key]], axis=0)
            task_state = real_state[state_key]
            self.data[self.run]["estimation_error"][state_key] = np.append(self.data[self.run]["estimation_error"][state_key], [task_state - estimator_mean[:len(task_state)]], axis=0)
            if state_key == "PolarTargetPos":
                task_state[0] -= self.data[self.run]["desired_distance"]
            self.data[self.run]["task_state"][state_key] = np.append(self.data[self.run]["task_state"][state_key], [task_state], axis=0)
        for obs_key in observation.keys():
            self.data[self.run]["observation"][obs_key]["measurement"] = np.append(self.data[self.run]["observation"][obs_key]["measurement"], [observation[obs_key]["measurement"]], axis=0)
            self.data[self.run]["observation"][obs_key]["noise"] = np.append(self.data[self.run]["observation"][obs_key]["noise"], [observation[obs_key]["noise"]], axis=0)
        for goal_key in goal_loss.keys():
            for subgoal_key in goal_loss[goal_key].keys():
                self.data[self.run]["goal_loss"][goal_key][subgoal_key] = np.append(self.data[self.run]["goal_loss"][goal_key][subgoal_key], goal_loss[goal_key][subgoal_key].item())
        for goal_key in gradients.keys():
            for subgoal_key in gradients[goal_key].keys():
                self.data[self.run]["gradient"][goal_key][subgoal_key] = np.append(self.data[self.run]["gradient"][goal_key][subgoal_key], [gradients[goal_key][subgoal_key].tolist()], axis=0)
        self.data[self.run]["action"] = np.append(self.data[self.run]["action"], [action.tolist()], axis=0)

    def end_wandb_run(self):
        if self.wandb_run is not None:
            self.wandb_run.finish()
            self.wandb_run = None

# ==================================================================================

class AICONLogger:
    def __init__(self, variations: List[dict]):
        self.current_variation_id = 0
        self.variations: Dict[int,Dict[str,dict]] = {}
        for variation in variations:
            self.set_variation(variation)

    def get_variation_config(self, id=None):
        if id is None:
            id = self.current_variation_id
        return self.variations[id]["config"]

    def set_variation(self, variation):
        self.current_variation_id = None
        for variation_id, saved_variation in self.variations.items():
            if all(saved_variation["config"].get(key) == val for key, val in variation.items()):
                self.current_variation_id = variation_id
        if self.current_variation_id is None:
            if len(self.variations) == 0:
                self.current_variation_id = 1
            else:
                self.current_variation_id = max(self.variations.keys()) + 1
            self.variations[self.current_variation_id] = {
                "config": variation,
                "data": {},
            }
        return self.current_variation_id

    def add_variations(self, variations):
        for variation in variations:
            self.set_variation(variation)

    # =============================================== helpers ======================================================

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
    
    # ======================================= plotting ==========================================

    @staticmethod
    def plot_mean_stddev(subplot: plt.Axes, means: np.ndarray, stddevs: np.ndarray, title:str, label:str, y_label:str, y_bounds:Tuple[float,float]=None, style_dict:Dict[str,str]=None):
        plot_kwargs = style_dict if style_dict is not None else {'label': label}
        subplot.plot(
            means,
            **plot_kwargs
        )
        stddev_kwargs = {"color": plot_kwargs["color"]} if "color" in plot_kwargs.keys() else {}
        subplot.fill_between(
            range(len(means)), 
            means - stddevs,
            means + stddevs,
            alpha=0.2,
            **stddev_kwargs
        )
        subplot.set_title(title, fontsize=16)
        subplot.set_xlabel("Time Step", fontsize=16)
        subplot.set_ylabel(y_label, fontsize=16)
        subplot.grid(True)
        subplot.legend()
        subplot.set_xlim(0, len(means))
        if y_bounds is not None:
            subplot.set_ylim(y_bounds)

    @staticmethod
    def plot_runs(subplot: plt.Axes, data:List[np.ndarray], labels:List[int], state_index:int=None, title:str=None, y_label:str=None, x_label:str=None, legend=False):
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

    @staticmethod
    def save_fig(fig:plt.Figure, save_path:str=None, show:bool=False):
        if save_path is not None:
            os.makedirs(os.path.dirname(save_path), exist_ok=True)
            fig.tight_layout()
            time.sleep(0.3)
            print(f"Saving to {save_path}...")
            fig.savefig(save_path)
        if show:
            fig.show()
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

            for i in range(len(indices)):
                axs[0][i].axhline(y=0, color='black', linestyle='--', linewidth=1)
                axs[1][i].axhline(y=0, color='black', linestyle='--', linewidth=1)
                axs[2][i].axhline(y=0, color='black', linestyle='--', linewidth=1)

            for label, variation in plotting_config["axes"].items():
                self.set_variation(variation)
                states = np.array([run_data["task_state"][state_id][:,indices] for run_key, run_data in self.variations[self.current_variation_id]['data'].items() if run_key not in plotting_config["exclude_runs"]])
                ucttys = np.array([run_data["estimators"][state_id]["uncertainty"][:,indices] for run_key, run_data in self.variations[self.current_variation_id]['data'].items() if run_key not in plotting_config["exclude_runs"]])
                errors = np.array([run_data["estimation_error"][state_id][:,indices] for run_key, run_data in self.variations[self.current_variation_id]['data'].items() if run_key not in plotting_config["exclude_runs"]])

                state_means, state_stddevs = self.compute_mean_and_stddev(states)
                error_means, error_stddevs = self.compute_mean_and_stddev(errors)
                uctty_means, uctty_stddevs = self.compute_mean_and_stddev(ucttys)
                for i in range(len(indices)):
                    self.plot_mean_stddev(axs[0][i], state_means[:, i], state_stddevs[:, i], "Task State", label, "Offset from Desired Distance", y_bounds=ybounds[0][i], style_dict=plotting_config["style"][label] if "style" in plotting_config.keys() else None)
                    self.plot_mean_stddev(axs[1][i], error_means[:, i], error_stddevs[:, i], "Estimation Error", label, "Estimation Error", y_bounds=ybounds[1][i], style_dict=plotting_config["style"][label] if "style" in plotting_config.keys() else None)
                    self.plot_mean_stddev(axs[2][i], uctty_means[:, i], uctty_stddevs[:, i], "Estimation Uncertainty", label, "Estimation Uncertainty (stddev)", y_bounds=ybounds[2][i], style_dict=plotting_config["style"][label] if "style" in plotting_config.keys() else None)    
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
            variation = plotting_config["axes"][axs_id]
            self.set_variation(variation)
            if runs is None:
                runs = list(self.current_data.keys())
            states = np.array([self.variations[self.current_variation_id]['data'][run]["task_state"][state_id][:,indices] for run in runs])
            ucttys = np.array([self.variations[self.current_variation_id]['data'][run]["estimators"][state_id]["uncertainty"][:,indices] for run in runs])
            errors = np.array([self.variations[self.current_variation_id]['data'][run]["estimation_error"][state_id][:,indices] for run in runs])
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
            for label, variation in plotting_config["axes"].items():
                self.set_variation(variation)
                total_loss = np.array([[0.0]*len(run_data["step"]) for run_key, run_data in self.variations[self.current_variation_id]['data'].items() if run_key not in plotting_config["exclude_runs"]])
                for subgoal_key in self.variations[self.current_variation_id]['data'][1]["goal_loss"][goal_key].keys():
                    goal_losses = np.array([run_data["goal_loss"][goal_key][subgoal_key] for run_key, run_data in self.variations[self.current_variation_id]['data'].items() if run_key not in plotting_config["exclude_runs"]])
                    total_loss += goal_losses
                    if plot_subgoals:
                        goal_loss_means, goal_loss_stddevs = self.compute_mean_and_stddev(goal_losses)
                        self.plot_mean_stddev(axs_goal[i], goal_loss_means, goal_loss_stddevs, f"{goal_key} goal loss", f"{label} {subgoal_key} loss", "Loss Mean and Stddev", ybounds, style_dict=plotting_config["style"][label] if "style" in plotting_config.keys() else None)
                total_loss_means, total_loss_stddevs = self.compute_mean_and_stddev(total_loss)
                self.plot_mean_stddev(axs_goal[i], total_loss_means, total_loss_stddevs, f"{goal_key} goal loss", f"{label} total loss", "Loss Mean and Stddev", ybounds, style_dict=plotting_config["style"][label] if "style" in plotting_config.keys() else None)
        # save / show
        loss_path = os.path.join(save_path, f"records/loss_{plotting_config['name']}.png") if save_path is not None else None
        self.save_fig(fig_goal, loss_path, show)

    def save(self, save_path:str):
        yaml_dict = {
            variation_id: {
                "variation": variation["config"],
                "data":      variation["data"],
            } for variation_id, variation in self.variations.items()
        }
        with open(os.path.join(save_path, "records/data.pkl"), 'wb') as f:
            pickle.dump(yaml_dict, f)

    