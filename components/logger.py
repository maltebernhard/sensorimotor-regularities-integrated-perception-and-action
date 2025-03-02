import pickle
import time
from typing import Dict, List, Tuple
from matplotlib import pyplot as plt
import numpy as np
import torch
import os

import wandb

# Set up matplotlib to use LaTeX fonts
plt.rcParams.update(
    {
        "text.usetex": True,
        "font.family": "serif",
        "font.serif": ["Computer Modern Roman"],
        "figure.figsize": (7, 5),
        "savefig.dpi": 300,
        "axes.titlesize": 13,
        "axes.labelsize": 13,
        "font.size": 12,
        "legend.fontsize": 11,
        "xtick.labelsize": 11,
        "ytick.labelsize": 11,
        "savefig.format": "pdf",
        "savefig.bbox": "tight",
    }
)

from components.helpers import rotate_vector_2d
os.environ["WANDB_SILENT"] = "true"

# ==================================================================================

class VariationLogger:
    def __init__(self, variation_config:dict, variation_id:int, wandb_config:dict):
        self.data = {}
        self.run_seed = 0
        self.variation_config = variation_config
        self.variation_id = variation_id
        self.wandb_project: str = wandb_config['wandb_project'] if wandb_config is not None else None
        self.wandb_group: str = wandb_config['wandb_group'] if wandb_config is not None else None
        self.wandb_run = None

    def log(self, step: int, time: float, estimators: Dict[str,Dict[str,torch.Tensor]], env_state: Dict[str,float], observation: Dict[str,Dict[str,float]], goal_loss: Dict[str,Dict[str,torch.Tensor]], action: torch.Tensor, gradients: Dict[str,Dict[str,torch.Tensor]]):
        # for new run, set up logging dict
        if self.run_seed not in self.data:
            self.create_run_dict(estimators, env_state, observation, goal_loss, action, gradients)
            if self.wandb_group is not None:
                self.create_wandb_run()
        # log data
        real_state = self.create_real_state(estimators, env_state)
        step_log = self.create_step_log(step, time, estimators, real_state, observation, goal_loss, action, gradients)

        if self.wandb_run is not None:
            self.log_wandb(step_log)
        self.log_local(step_log)

    def create_run_dict(self, estimators: Dict[str,Dict[str,torch.Tensor]], env_state: Dict[str,float], observation: Dict[str,Dict[str,float]], goal_loss: Dict[str,Dict[str,torch.Tensor]], action: torch.Tensor, gradient: Dict[str,torch.Tensor]):
        # TODO: skip runs which are already logged
        self.data[self.run_seed] = {
                "step":                 np.array([]),     # time step
                "time":                 np.array([]),     # time in seconds
                "estimators": {estimator_key: {
                    'mean':             np.empty((0,) + estimators[estimator_key]["mean"].shape),     # mean of state estimate
                    'cov':              np.empty((0,) + estimators[estimator_key]["cov"].shape),      # covariance of state estimate
                    'uncertainty':      np.empty((0,) + (estimators[estimator_key]["mean"].shape[0],)), # uncertainty of state estimate: sqrt(diag(covariance))
                    'estimation_error': np.empty((0,) + estimators[estimator_key]["mean"].shape),     # estimation error: estimation - real state
                    'env_state':        np.empty((0,) + estimators[estimator_key]["mean"].shape),     # task state: real state with subtracted offsets (like desired target distance)
                } for estimator_key in estimators.keys()},
                "observation": {obs_key: {
                    "measurement":      np.array([]),     # measurement
                    "noise_mean":       np.array([]),     # mean of noise
                    "noise_stddev":     np.array([]),     # stddev of noise
                } for obs_key in observation.keys()},
                "goal_loss": {
                    subgoal_key:        np.array([])     # goal loss function value
                for subgoal_key in goal_loss.keys()},
                "gradient": {
                    subgoal_key:        np.empty((0,) + gradient[subgoal_key].shape)     # gradient of goal loss function
                for subgoal_key in gradient.keys()},
                "rtf_gradient": {
                    subgoal_key:        np.empty((0,) + gradient[subgoal_key].shape)     # gradient of goal loss function
                for subgoal_key in gradient.keys()},
                "action":               np.empty((0,) + action.shape),     # action
                "rtf_action":           np.empty((0,) + action.shape),     # action rotated to target frame
                "desired_distance":     env_state["desired_target_distance"],
                "collision":            np.array([]),     # collision flag
            }

    def create_wandb_run(self):
        if self.wandb_run is not None:
            self.wandb_run.finish()
        config = self.variation_config.copy()
        config.update({"id": self.variation_id})
        self.wandb_run = wandb.init(
            project=self.wandb_project,
            name=f'{self.variation_id}_{self.run_seed}',
            group=self.wandb_group,
            config = config,
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
                if (obj=='target' and self.variation_config["target_config"][0] != "stationary") or ('obstacle' in obj and self.variation_config["obstacles"][int(obj[-1])-1][0] != "stationary"):
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
        real_state["collision"] = max([val for key, val in env_state.items() if "collision" in key])
        return real_state

    def create_step_log(self, step: int, time: float, estimators: Dict[str,Dict[str,torch.Tensor]], real_state: Dict[str,Dict[str,torch.Tensor]], observation: Dict[str,Dict[str,float]], goal_loss: Dict[str,Dict[str,torch.Tensor]], action: torch.Tensor, gradients: Dict[str,Dict[str,torch.Tensor]]):
        estimators = {estimator_key: {attribute_key: estimators[estimator_key][attribute_key].cpu().numpy() for attribute_key in estimators[estimator_key].keys()} for estimator_key in estimators.keys()}
        # HACK: remove negative covariances
        # TODO: check why they even exist
        for estimator in estimators.values():
            estimator['cov'][estimator['cov'] < 0] = 0
        return {
            "step":                     step,
            "time":                     time,
            "estimators": {
                estimator_key: {
                    'mean':             estimator['mean'],
                    'cov':              estimator['cov'],
                    'uncertainty':      np.sqrt(np.diag(estimator['cov'])),
                    'estimation_error': real_state[estimator_key] - estimator['mean'],
                    'env_state':        real_state[estimator_key],
                } for estimator_key, estimator in estimators.items() if estimator_key != "RobotVel"
            },
            "collision":                real_state["collision"],
            "observation": {
                obs_key: {
                    "measurement":      observation[obs_key]["measurement"],
                    "noise_mean":       observation[obs_key]["noise"][0],
                    "noise_stddev":     observation[obs_key]["noise"][1],
                } for obs_key in observation.keys()
            },
            "goal_loss":                {subgoal_key: goal_loss[subgoal_key].cpu().numpy() for subgoal_key in goal_loss.keys()},
            "gradient":                 {subgoal_key: gradients[subgoal_key].cpu().numpy() for subgoal_key in gradients.keys()},
            "rtf_gradient":             {subgoal_key: np.append(rotate_vector_2d(-estimators["PolarTargetPos"]['mean'][1],gradients[subgoal_key].cpu().numpy()[:2]),gradients[subgoal_key].cpu().numpy()[2]) for subgoal_key in gradients.keys()},
            "action":                   action.cpu().numpy(),
            "rtf_action":               np.append(rotate_vector_2d(-estimators["PolarTargetPos"]['mean'][1],action.cpu().numpy()[:2]),action.cpu().numpy()[2]),
        }

    def log_wandb(self, step_log: dict):
        if self.variation_config["target_config"][0] != "stationary":
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
            #"step": step_log["step"],
            #"time": step_log["time"],
            "estimators": {estimator_key: {
                attribute_key: {index_keys[estimator_key][i]: val for i, val in enumerate(estimator[attribute_key])} for attribute_key in estimator.keys()
            } for estimator_key, estimator in step_log["estimators"].items() if estimator_key != "RobotVel"},
            "collision": step_log["collision"],
            #"observation": step_log["observation"],
            "goal_loss": step_log["goal_loss"],
            #"gradient": step_log["gradient"],
            #"action": step_log["action"],
        })

    def log_local(self, step_log: dict):
        for key in ["step", "time", "collision"]:
            if key != "estimators":
                self.data[self.run_seed][key] = np.append(self.data[self.run_seed][key], step_log[key])
        for estimator_key in step_log["estimators"].keys():
            for attribute_key in step_log["estimators"][estimator_key].keys():
                self.data[self.run_seed]["estimators"][estimator_key][attribute_key] = np.append(self.data[self.run_seed]["estimators"][estimator_key][attribute_key], [step_log["estimators"][estimator_key][attribute_key]], axis=0)
        for obs_key in step_log["observation"].keys():
            for sub_key in step_log["observation"][obs_key].keys():
                self.data[self.run_seed]["observation"][obs_key][sub_key] = np.append(self.data[self.run_seed]["observation"][obs_key][sub_key], [step_log["observation"][obs_key][sub_key]], axis=0)
        for subgoal_key in step_log["goal_loss"].keys():
            self.data[self.run_seed]["goal_loss"][subgoal_key] = np.append(self.data[self.run_seed]["goal_loss"][subgoal_key], [step_log["goal_loss"][subgoal_key]], axis=0)
        for subgoal_key in step_log["gradient"].keys():
            self.data[self.run_seed]["gradient"][subgoal_key] = np.append(self.data[self.run_seed]["gradient"][subgoal_key], [step_log["gradient"][subgoal_key]], axis=0)
            self.data[self.run_seed]["rtf_gradient"][subgoal_key] = np.append(self.data[self.run_seed]["rtf_gradient"][subgoal_key], [step_log["rtf_gradient"][subgoal_key]], axis=0)
        self.data[self.run_seed]["action"] = np.append(self.data[self.run_seed]["action"], [step_log["action"]], axis=0)
        self.data[self.run_seed]["rtf_action"] = np.append(self.data[self.run_seed]["rtf_action"], [step_log["rtf_action"]], axis=0)

    def end_wandb_run(self):
        if self.wandb_run is not None:
            self.wandb_run.finish()
            del self.wandb_run
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
    
    def add_data(self, data:dict):
        for variation in data.values():
            config = variation["config"]
            self.set_variation(config)
            self.variations[self.current_variation_id]["data"].update(variation["data"])

    # =============================================== helpers ======================================================

    @staticmethod
    def compute_mean_and_stddev(data: List[np.ndarray]) -> Tuple[np.ndarray, np.ndarray]:
        max_length = max(len(d) for d in data)
        means = []
        stddevs = []
        for t in range(max_length):
            valid_data = [d[t] for d in data if len(d) > t]
            if valid_data:
                mean = np.mean(valid_data, axis=0)
                stddev = np.std(valid_data, axis=0)
            else:
                mean = np.nan
                stddev = np.nan
            means.append(mean)
            stddevs.append(stddev)
        return np.array(means), np.array(stddevs)

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
    def plot_mean_stddev(subplot: plt.Axes, means: np.ndarray, stddevs: np.ndarray, collisions:List[Tuple[float,float]], label:str, plotting_dict:Dict[str,str]):
        style_dict = {key: plotting_dict['style'][label][key] for key in ['label', 'color', 'linestyle', 'linewidth']}
        
        subplot.plot(
            means,
            **style_dict
        )
        stddev_kwargs = {"color": style_dict["color"]}
        subplot.fill_between(
            range(len(means)), 
            means - stddevs,
            means + stddevs,
            alpha=0.2,
            **stddev_kwargs
        )
        for (x, y) in collisions:
            subplot.plot(x, y, 'x', markersize=10, markeredgewidth=2, **stddev_kwargs)

    @staticmethod
    def plot_runs(subplot: plt.Axes, data:List[np.ndarray], labels:List[int], title:str, y_label:str, collisions, state_index:int=None):
        color_cycle = ['b', 'g', 'r', 'c', 'm', 'y', 'k', 'orange', 'purple', 'brown']
        if state_index is None:
            for i, run in enumerate(data):
                c = {'color': color_cycle[i%len(color_cycle)]}
                subplot.plot(run[:], label=f'{labels[i]}', **c)
                if collisions[i][-1]:
                    subplot.plot(len(collisions[i]), run[-1], 'x', markersize=10, markeredgewidth=2, **c)
        else:
            for i, run in enumerate(data):
                c = {'color': color_cycle[i%len(color_cycle)]}
                subplot.plot(run[:, state_index], label=f'{labels[i]}', **c)
                if collisions[i][-1]:
                    subplot.plot(len(collisions[i]), run[-1], 'x', markersize=10, markeredgewidth=2, **c)
        subplot.set_title("\\textbf{" + title + "}", fontsize=20)
        subplot.set_xlabel("Time Step", fontsize=20)
        subplot.set_ylabel(y_label, fontsize=20)
        max_steps = max(len(run) for run in data)
        subplot.set_xlim(0, max_steps)
        subplot.grid(True)
        subplot.legend(fontsize=16)

    @staticmethod
    def compute_mean_and_stddev(data: List[np.ndarray], col) -> Tuple[np.ndarray, np.ndarray, List[Tuple[int, float]]]:
        max_length = max(len(d) for d in data)
        means = []
        stddevs = []
        collisions = []
        for t in range(max_length):
            valid_data = [d[t] for d in data if len(d) > t]
            if valid_data:
                mean = np.mean(valid_data, axis=0)
                stddev = np.std(valid_data, axis=0)
            else:
                mean = np.nan
                stddev = np.nan
            means.append(mean)
            stddevs.append(stddev)
        for i,c in enumerate(col):
            if c[-1]:
                collisions.append((len(c) - 1, data[i][-1]))
        return np.array(means), np.array(stddevs), collisions
    
    def plot_state_boxplots(self, plotting_config:Dict[str,Dict[str,Tuple[List[int],List[str],List[Tuple[float,float]]]]], save_path:str=None, show:bool=False):
        xbounds = plotting_config["xbounds"]
        for state_id, config in plotting_config["states"].items():
            indices = np.array(config["indices"])
            if len(indices) == 1:
                fig, axs = plt.subplots(len(indices), 3, figsize=(15, 5))
                fig_abs, axs_abs = plt.subplots(len(indices), 3, figsize=(15, 5))
                axs = [[ax] for ax in axs]
                axs_abs = [[ax] for ax in axs_abs]
            else:
                fig, axs = plt.subplots(3, len(indices), figsize=(6 * len(indices), 21))
                fig_abs, axs_abs = plt.subplots(3, len(indices), figsize=(6 * len(indices), 21))
            len_data = max(len(run_data["step"]) for variation in self.variations.values() for run_data in variation['data'].values())

            states = {}
            abs_states = {}
            errors = {}
            abs_errors = {}
            ucttys = {}
            
            for label, variation in plotting_config["axes"].items():
                self.set_variation(variation)

                var_states = [run_data["estimators"][state_id]["env_state"][xbounds[0]:min(xbounds[1]+1,len_data),indices] for run_key, run_data in self.variations[self.current_variation_id]['data'].items() if run_key not in plotting_config["exclude_runs"]]
                desired_distance = [run_data["desired_distance"]*np.ones_like(run_data["estimators"][state_id]["env_state"][xbounds[0]:min(xbounds[1]+1,len_data),indices]) for run_key, run_data in self.variations[self.current_variation_id]['data'].items() if run_key not in plotting_config["exclude_runs"]]

                var_ucttys = [run_data["estimators"][state_id]["uncertainty"][xbounds[0]:min(xbounds[1]+1,len_data),indices] for run_key, run_data in self.variations[self.current_variation_id]['data'].items() if run_key not in plotting_config["exclude_runs"]]
                var_errors = [run_data["estimators"][state_id]["estimation_error"][xbounds[0]:min(xbounds[1]+1,len_data),indices] for run_key, run_data in self.variations[self.current_variation_id]['data'].items() if run_key not in plotting_config["exclude_runs"]]
                var_collisions = [run_data["collision"] for run_key, run_data in self.variations[self.current_variation_id]['data'].items() if run_key not in plotting_config["exclude_runs"]]

                states[label] = np.array([np.mean(run - dd) for run, dd, col in zip(var_states, desired_distance, var_collisions) if len(col) >= xbounds[1]])
                abs_states[label] = np.array([np.mean(np.abs(run - dd)) for run, dd, col in zip(var_states, desired_distance, var_collisions) if len(col) >= xbounds[1]])
                errors[label] = np.array([np.mean(run) for run, col in zip(var_errors, var_collisions) if len(col) >= xbounds[1]])
                abs_errors[label] = np.array([np.mean(np.abs(run)) for run, col in zip(var_errors, var_collisions) if len(col) >= xbounds[1]])
                ucttys[label] = np.array([np.mean(run) for run, col in zip(var_ucttys, var_collisions) if len(col) >= xbounds[1]])

            # NOTE: used only to accumulate different target motions into one plot
            if 'extended' in save_path:
                alt_abs_states = {
                    "stationary_aicon": abs_states["stationary_aicon"],
                    "stationary_trc": abs_states["stationary_trc"],
                    "motion_aicon": np.concatenate([abs_states[label] for label in abs_states if "aicon" in label and "stationary" not in label]),
                    "motion_trc": np.concatenate([abs_states[label] for label in abs_states if "trc" in label and "stationary" not in label]),
                }
                alt_abs_errors = {
                    "stationary_aicon": abs_errors["stationary_aicon"],
                    "stationary_trc": abs_errors["stationary_trc"],
                    "motion_aicon": np.concatenate([abs_errors[label] for label in abs_errors if "aicon" in label and "stationary" not in label]),
                    "motion_trc": np.concatenate([abs_errors[label] for label in abs_errors if "trc" in label and "stationary" not in label]),
                }
                alt_ucttys = {
                    "stationary_aicon": ucttys["stationary_aicon"],
                    "stationary_trc": ucttys["stationary_trc"],
                    "motion_aicon": np.concatenate([ucttys[label] for label in ucttys if "aicon" in label and "stationary" not in label]),
                    "motion_trc": np.concatenate([ucttys[label] for label in ucttys if "trc" in label and "stationary" not in label]),
                }
                alt_fig, alt_axs = plt.subplots(1, 3, figsize=(21, 6))
                for i, data in enumerate([alt_abs_states, alt_abs_errors, alt_ucttys]):
                    alt_ax = alt_axs[i]
                    label_index = -1
                    for alt_label in ["stationary_aicon", "motion_aicon", "stationary_trc", "motion_trc"]:
                        label = ("sine_aicon" if "aicon" in alt_label else "sine_trc") if "motion" in alt_label else alt_label
                        label_key = 'boxlabel' if 'boxlabel' in plotting_config['style'][label] else 'label'
                        plot_label: str = plotting_config['style'][label][label_key]
                        if "sine" in plot_label:
                            plot_label = plot_label.replace("sine", "moves")
                        if "stat." in plot_label:
                            plot_label = plot_label.replace("stat.", "stationary")
                        # use the label index as the x-position
                        label_index += 1
                        alt_ax.boxplot(
                            data[alt_label],
                            positions=[label_index + 1],
                            labels=[plot_label],
                            patch_artist=True,
                            boxprops=dict(
                                facecolor=plotting_config['style'][label]['boxcolor'],
                            ),
                            medianprops=dict(
                                linewidth=2.0,
                                color=plotting_config['style'][label]['color'],
                            ),
                            showfliers=False,
                            widths=0.5  # Adjust the width of the box
                        )
                    alt_ax.tick_params(axis='x', labelsize=14)
                    max_val = max(data[alt_label].max() for alt_label in alt_abs_states.keys())
                    alt_ax.set_ylim(0.0, max_val * 1.1)
                    #alt_ax.axhline(y=0, color='black', linestyle='solid', linewidth=1)
                    alt_ax.set_title("\\textbf{" + f"{['Absolute Task (Distance) Error', 'Absolute Estimation Error', 'Estimation Uncertainty'][i]}" + "}", fontsize=16)
                    alt_ax.set_ylabel(['Distance Offset from $d^\\ast$', 'Distance Estimation Error', 'Distance Estimate Standard Deviation'][i], fontsize=16)
                    alt_ax.grid(True)#, axis='y')
                alt_fig.tight_layout()
                path = os.path.join(save_path, f"records/alt_abs_box_{plotting_config['name']}") if save_path is not None else None
                self.save_fig(alt_fig, path, show)

            # plot absolute_bars
            for i in range(len(indices)):
                for j, data in enumerate([abs_states, abs_errors, ucttys]):
                    for label in plotting_config["axes"].keys():
                        label_key = 'boxlabel' if 'boxlabel' in plotting_config['style'][label] else 'label'
                        plot_label = plotting_config['style'][label][label_key]
                        # use the label index as the x-position
                        label_index = list(plotting_config["axes"].keys()).index(label)
                        axs_abs[j][i].boxplot(
                            data[label],
                            positions=[label_index + 1],
                            labels=[plot_label],
                            patch_artist=True,
                            boxprops=dict(
                                facecolor=plotting_config['style'][label]['boxcolor'],
                            ),
                            medianprops=dict(
                                linewidth=2.0,
                                color=plotting_config['style'][label]['color'],
                            ),
                            showfliers=False,
                            widths=0.5  # Adjust the width of the box
                        )
                    axs_abs[j][i].tick_params(axis='x', labelsize=14)
                    max_val = max(data[label].max() for label in plotting_config["axes"].keys())
                    axs_abs[j][i].set_ylim(0.0, max_val * 1.1)
                    axs_abs[j][i].axhline(y=0, color='black', linestyle='solid', linewidth=1)
                    axs_abs[j][i].set_title("\\textbf{" + f"{['Task Error', 'Estimation Error', 'Estimation Uncertainty'][j]}" + "}", fontsize=16)
                    axs_abs[j][i].set_ylabel(['Distance Offset from $d^\\ast$', 'Distance Estimation Error', 'Distance Estimate Standard Deviation'][j], fontsize=16)
                    axs_abs[j][i].grid(True)#, axis='y')

            # plot avg_bars
            for i in range(len(indices)):
                for j, data in enumerate([states, errors, ucttys]):
                    for label in plotting_config["axes"].keys():
                        label_key = 'boxlabel' if 'boxlabel' in plotting_config['style'][label] else 'label'
                        plot_label = plotting_config['style'][label][label_key]
                        label_index = list(plotting_config["axes"].keys()).index(label)
                        axs[j][i].boxplot(
                            data[label],
                            positions=[label_index+1],
                            labels=[plot_label],
                            patch_artist=True,
                            boxprops=dict(
                                facecolor=plotting_config['style'][label]['boxcolor'],
                            ),
                            medianprops=dict(
                                linewidth=2.0,
                                color=plotting_config['style'][label]['color'],

                            ),
                            showfliers=False,
                            widths=0.5  # Adjust the width of the box
                        )
                    axs_abs[j][i].tick_params(axis='x', labelsize=14)
                    max_val = max(data[label].max() for label in plotting_config["axes"].keys())
                    min_val = min(data[label].min() for label in plotting_config["axes"].keys())
                    axs[j][i].set_ylim(min(0.0, min_val * 1.1), max(0.0, max_val * 1.1))
                    axs[j][i].axhline(y=0, color='black', linestyle='solid', linewidth=1)
                    axs[j][i].set_title("\\textbf{" + f"{['Task Error', 'Estimation Error', 'Estimation Uncertainty'][j]}" + "}", fontsize=16)
                    axs[j][i].set_ylabel(['Distance Offset from $d^\\ast$', 'Distance Estimation Error', 'Distance Estimate Standard Deviation'][j], fontsize=16)
                    axs[j][i].grid(True)#, axis='y')
            # save / show
            path = os.path.join(save_path, f"records/box_{plotting_config['name']}") if save_path is not None else None
            self.save_fig(fig, path, show)
            path = os.path.join(save_path, f"records/abs_box_{plotting_config['name']}") if save_path is not None else None
            self.save_fig(fig_abs, path, show)

    def plot_collisions(self, plotting_config:Dict[str,Dict[str,Tuple[List[int],List[str],List[Tuple[float,float]]]]], save_path:str=None, show:bool=False):
        xbounds = plotting_config["xbounds"]
        
        ratio_collisions = {}
        total_runs = len(self.variations[self.current_variation_id]['data']) - len(plotting_config["exclude_runs"])
        for label, variation in plotting_config["axes"].items():
            self.set_variation(variation)
            var_collisions = [run_data["collision"] for run_key, run_data in self.variations[self.current_variation_id]['data'].items() if run_key not in plotting_config["exclude_runs"]]
            ratio_collisions[label] = sum(collisions_run[:min(len(collisions_run),xbounds[1])].sum() for collisions_run in var_collisions) / total_runs
        if not all(col == 0 for col in ratio_collisions.values()):
            fig, ax = plt.subplots(1, 1, figsize=(7, 6))
            ax.grid(axis='x')
            for label, ratio in ratio_collisions.items():
                ax.barh(
                    plotting_config['style'][label]['label'],
                    ratio,
                    color=plotting_config['style'][label]['color']
                )
            ax.tick_params(axis='y', labelsize=16)
            ax.set_title("\\textbf{Collisions }", fontsize=20)
            ax.set_xlabel("Avg. Collisions per Run", fontsize=16)

            ax.set_xlim(0, 1)
            path = os.path.join(save_path, f"records/collisions_{plotting_config['name']}") if save_path is not None else None
            self.save_fig(fig, path, show)

    def plot_states(self, plotting_config:Dict[str,Dict[str,Tuple[List[int],List[str],List[Tuple[float,float]]]]], save_path:str=None, show:bool=False):
        xbounds = plotting_config["xbounds"]
        for state_id, config in plotting_config["states"].items():
            indices = np.array(config["indices"])
            ybounds = config["ybounds"]
            len_data = max(len(run_data["step"]) for variation in self.variations.values() for run_data in variation['data'].values())
            absybounds = config.get("absybounds", None)
            if len(indices) == 1:
                fig, axs = plt.subplots(len(indices), 3, figsize=(21, 6))
                abs_fig, abs_axs = plt.subplots(len(indices), 3, figsize=(21, 6))
                axs = [[ax] for ax in axs]
                abs_axs = [[ax] for ax in abs_axs]
            else:
                fig, axs = plt.subplots(3, len(indices), figsize=(7 * len(indices), 18))
                abs_fig, abs_axs = plt.subplots(3, len(indices), figsize=(7 * len(indices), 18))

            sensor_failures = []
            if all("Divergence" in var["smcs"] for var in plotting_config["axes"].values()) and all(any("visual_angle_dot" in loss_key for loss_key in var["observation_loss"].keys()) for var in plotting_config["axes"].values()):
                sensor_failures.append("divergence")
            elif all("Triangulation" in var["smcs"] for var in plotting_config["axes"].values()) and all(any("offset_angle_dot" in loss_key for loss_key in var["observation_loss"].keys()) for var in plotting_config["axes"].values()):
                sensor_failures.append("triangulation")
            elif all(ax['distance_sensor']=="distsensor" for ax in plotting_config["axes"].values()) and all(any("distance" in loss_key for loss_key in var["observation_loss"].keys()) for var in plotting_config["axes"].values()):
                sensor_failures.append("distance")
            elif all(ax['distance_sensor']=="distsensor" for ax in plotting_config["axes"].values()) and any(any("distance" in loss_key for loss_key in var["observation_loss"].keys()) for var in plotting_config["axes"].values()):
                print("WARN: not all variations have distance sensor failures")
            if len(set([var["desired_distance"] for var in plotting_config["axes"].values()])) > 1:
                print("WARN: not all variations have the same desired distance!")

            for i in range(len(indices)):
                # HACK: plot desired distance
                # TODO: think about how to handle this if there are different desired distances
                axs[0][i].axhline(y=list(plotting_config['axes'].values())[0]['desired_distance'], color='black', linestyle='dotted', linewidth=2, label='desired distance to target: $d^\\ast=10$')
                
                axs[0][i].axhline(y=0, color='black', linestyle='solid', linewidth=2)
                axs[1][i].axhline(y=0, color='black', linestyle='solid', linewidth=2)
                axs[2][i].axhline(y=0, color='black', linestyle='solid', linewidth=2)
                if len(sensor_failures) > 0:
                    axs[0][i].axvspan(100, 200, color='grey', alpha=0.2, label=f'sensor failure for {sensor_failures}')
                    axs[1][i].axvspan(100, 200, color='grey', alpha=0.2, label=f'sensor failure for {sensor_failures}')
                    axs[2][i].axvspan(100, 200, color='grey', alpha=0.2, label=f'sensor failure for {sensor_failures}')

            for label, variation in plotting_config["axes"].items():
                self.set_variation(variation)
                # NOTE: calculate mean_stddev with collisions and plot collisions
                states = [run_data["estimators"][state_id]["env_state"][:,indices] for run_key, run_data in self.variations[self.current_variation_id]['data'].items() if run_key not in plotting_config["exclude_runs"]]
                # ucttys = [run_data["estimators"][state_id]["uncertainty"][:,indices] for run_key, run_data in self.variations[self.current_variation_id]['data'].items() if run_key not in plotting_config["exclude_runs"]]
                # errors = [run_data["estimators"][state_id]["estimation_error"][:,indices] for run_key, run_data in self.variations[self.current_variation_id]['data'].items() if run_key not in plotting_config["exclude_runs"]]
                #collisions = [run_data["collision"] for run_key, run_data in self.variations[self.current_variation_id]['data'].items() if run_key not in plotting_config["exclude_runs"]]

                # NOTE: calculate mean_stddev without collisions
                states = [run_data["estimators"][state_id]["env_state"][xbounds[0]:min(xbounds[1]+1,len_data),indices] for run_key, run_data in self.variations[self.current_variation_id]['data'].items() if run_key not in plotting_config["exclude_runs"] and not len(run_data["collision"])<xbounds[1]]
                ucttys = [run_data["estimators"][state_id]["uncertainty"][xbounds[0]:min(xbounds[1]+1,len_data),indices] for run_key, run_data in self.variations[self.current_variation_id]['data'].items() if run_key not in plotting_config["exclude_runs"] and not len(run_data["collision"])<xbounds[1]]
                errors = [run_data["estimators"][state_id]["estimation_error"][xbounds[0]:min(xbounds[1]+1,len_data),indices] for run_key, run_data in self.variations[self.current_variation_id]['data'].items() if run_key not in plotting_config["exclude_runs"] and not len(run_data["collision"])<xbounds[1]]
                collisions = [run_data["collision"][xbounds[0]:min(xbounds[1]+1,len_data)] for run_key, run_data in self.variations[self.current_variation_id]['data'].items() if run_key not in plotting_config["exclude_runs"] and not len(run_data["collision"])<xbounds[1]]

                # calculate absolute arrors
                desired_distance = [desired_distance*np.ones_like(np.array(states[i])) for i,desired_distance in enumerate([run_data["desired_distance"] for run_key, run_data in self.variations[self.current_variation_id]['data'].items() if run_key not in plotting_config["exclude_runs"] and not len(run_data["collision"])<xbounds[1]])]
                abs_task_errors = [np.abs(states[i]-desired_distance[i]) for i in range(len(states))]
                abs_estimation_errors = [np.abs(errors[i]) for i in range(len(errors))]

                state_means, state_stddevs, state_collisions = self.compute_mean_and_stddev(states, collisions)
                # HACK: overwrite distances to target at collision step, cause they slightly vary due to how much "inside" the agent is in the target at the step
                for i, col in enumerate(state_collisions):
                    state_collisions[i] = (col[0], 0.0)
                error_means, error_stddevs, error_collisions = self.compute_mean_and_stddev(errors, collisions)
                uctty_means, uctty_stddevs, uctty_collisions = self.compute_mean_and_stddev(ucttys, collisions)
                abs_task_means, abs_task_stddevs, abs_task_collisions = self.compute_mean_and_stddev(abs_task_errors, collisions)
                abs_error_means, abs_error_stddevs, abs_error_collisions = self.compute_mean_and_stddev(abs_estimation_errors, collisions)

                for i in range(len(indices)):
                    self.plot_mean_stddev(axs[0][i], state_means[xbounds[0]:min(xbounds[1]+1,len(state_means)), i], state_stddevs[xbounds[0]:min(xbounds[1]+1,len(state_means)), i], state_collisions, label, plotting_config)
                    self.plot_mean_stddev(axs[1][i], error_means[xbounds[0]:min(xbounds[1]+1,len(state_means)), i], error_stddevs[xbounds[0]:min(xbounds[1]+1,len(state_means)), i], error_collisions, label, plotting_config)
                    self.plot_mean_stddev(axs[2][i], uctty_means[xbounds[0]:min(xbounds[1]+1,len(state_means)), i], uctty_stddevs[xbounds[0]:min(xbounds[1]+1,len(state_means)), i], uctty_collisions, label, plotting_config)    
                    self.plot_mean_stddev(abs_axs[0][i], abs_task_means[xbounds[0]:min(xbounds[1]+1,len(state_means)), i], abs_task_stddevs[xbounds[0]:min(xbounds[1]+1,len(state_means)), i], abs_task_collisions, label, plotting_config)
                    self.plot_mean_stddev(abs_axs[1][i], abs_error_means[xbounds[0]:min(xbounds[1]+1,len(state_means)), i], abs_error_stddevs[xbounds[0]:min(xbounds[1]+1,len(state_means)), i], abs_error_collisions, label, plotting_config)
                    self.plot_mean_stddev(abs_axs[2][i], uctty_means[xbounds[0]:min(xbounds[1]+1,len(state_means)), i], uctty_stddevs[xbounds[0]:min(xbounds[1]+1,len(state_means)), i], uctty_collisions, label, plotting_config)

            titles   = ["\\textbf{Task Error}", "\\textbf{Estimation Error}", "\\textbf{Estimation Uncertainty}"]
            y_labels = ["Distance to Target", "Distance Estimation Error", "Distance Estimation Uncertainty (stddev)"]
            max_steps = max(len(run_data["step"]) for variation in self.variations.values() for run_data in variation['data'].values())
            for j in range(3):
                axs[j][i].set_title(titles[j], fontsize=20)
                axs[j][i].set_ylabel(y_labels[j], fontsize=20)
                axs[j][i].set_xlabel("Time Step", fontsize=20)
                axs[j][i].grid(True)
                axs[j][i].legend(fontsize=16)
                axs[j][i].set_xlim(xbounds[0], min(xbounds[1], max_steps))
                axs[j][i].set_ylim(ybounds[j][i])

                abs_axs[j][i].set_title(titles[j], fontsize=20)
                abs_axs[j][i].set_ylabel(y_labels[j], fontsize=20)
                abs_axs[j][i].set_xlabel("Time Step", fontsize=20)
                abs_axs[j][i].grid(True)
                abs_axs[j][i].legend(fontsize=16)
                abs_axs[j][i].set_xlim(xbounds[0], min(xbounds[1], max_steps))
                if absybounds is not None:
                    abs_axs[j][i].set_ylim(absybounds[j][i])
                else:
                    if j == 0:
                        abs_axs[j][i].set_ylim(0, ybounds[j][i][1] - self.variations[self.current_variation_id]['data'][1]["desired_distance"])
                    else:
                        abs_axs[j][i].set_ylim(0, ybounds[j][i][1]-ybounds[j][i][0])
            
            # save / show
            path = os.path.join(save_path, f"records/state_{plotting_config['name']}") if save_path is not None else None
            self.save_fig(fig, path, show)
            path = os.path.join(save_path, f"records/abs_state_{plotting_config['name']}") if save_path is not None else None
            self.save_fig(abs_fig, path, show)

    def plot_state_runs(self, plotting_config:Dict[str,Dict[str,Tuple[List[int],List[str],List[Tuple[float,float]]]]], axs_id: str, runs: list[int], save_path:str=None, show:bool=False):
        xbounds = plotting_config["xbounds"]
        for state_id, config in plotting_config["states"].items():
            indices = np.array(config["indices"])
            labels = config["labels"]
            ybounds = config["ybounds"]
            len_data = max(len(run_data["step"]) for variation in self.variations.values() for run_data in variation['data'].values())
            if len(indices) == 1:
                fig, axs = plt.subplots(len(indices), 3, figsize=(21, 6))
                axs = [[ax] for ax in axs]
            else:
                fig, axs = plt.subplots(3, len(indices), figsize=(7 * len(indices), 18))
            variation = plotting_config["axes"][axs_id]
            self.set_variation(variation)
            if runs is None:
                runs = list(self.variations[self.current_variation_id]['data'].keys())
            states = [self.variations[self.current_variation_id]['data'][run]["estimators"][state_id]["env_state"][:,indices] for run in runs]
            ucttys = [self.variations[self.current_variation_id]['data'][run]["estimators"][state_id]["uncertainty"][:,indices] for run in runs]
            errors = [self.variations[self.current_variation_id]['data'][run]["estimators"][state_id]["estimation_error"][:,indices] for run in runs]
            collisions = [self.variations[self.current_variation_id]['data'][run]["collision"] for run in runs]
            for i in range(len(indices)):
                titles   = ["Task State", "Estimation Error", "Estimation Uncertainty"]
                y_labels = ["Distance to Target", "Distance Estimation Error", "Distance Estimation Uncertainty (stddev)"]
                self.plot_runs(axs[0][i], states, runs, titles[0], y_labels[0], collisions, i)
                self.plot_runs(axs[1][i], errors, runs, titles[1], y_labels[1], collisions, i)
                self.plot_runs(axs[2][i], ucttys, runs, titles[2], y_labels[2], collisions, i)
                axs[0][i].set_ylim(ybounds[0][i])
                axs[0][i].set_xlim(xbounds[0], min(xbounds[1], len_data))
                axs[1][i].set_ylim(ybounds[1][i])
                axs[1][i].set_xlim(xbounds[0], min(xbounds[1], len_data))
                axs[2][i].set_ylim(ybounds[2][i])
                axs[2][i].set_xlim(xbounds[0], min(xbounds[1], len_data))

            # save / show
            path = os.path.join(save_path, f"records/runs/{axs_id}") if save_path is not None else None
            self.save_fig(fig, path, show)

    def plot_goal_losses(self, plotting_config:Dict[str,Dict[str,Tuple[List[int],List[str],List[Tuple[float,float]]]]], plot_subgoals:bool, save_path:str=None, show:bool=False):
        for label, style in list(plotting_config["style"].items()):
            plotting_config["style"][f"{label} total loss"] = style
            for subgoal_key in self.variations[self.current_variation_id]['data'][1]["goal_loss"]:
                plotting_config["style"][f"{label} {subgoal_key} loss"] = style
        # Plot goal losses
        fig_goal, ax = plt.subplots(1, 1, figsize=(7, 6))
        
        xbounds = plotting_config["xbounds"]
        len_data = max(len(run_data["step"]) for variation in self.variations.values() for run_data in variation['data'].values())
        
        for label, variation in plotting_config["axes"].items():
            self.set_variation(variation)
            total_loss = [[0.0]*len(run_data["step"]) for run_key, run_data in self.variations[self.current_variation_id]['data'].items() if run_key not in plotting_config["exclude_runs"]]
            collisions = [run_data["collision"] for run_key, run_data in self.variations[self.current_variation_id]['data'].items() if run_key not in plotting_config["exclude_runs"]]
            for subgoal_key in self.variations[self.current_variation_id]['data'][1]["goal_loss"].keys():
                goal_losses = [run_data["goal_loss"][subgoal_key] for run_key, run_data in self.variations[self.current_variation_id]['data'].items() if run_key not in plotting_config["exclude_runs"]]
                for j in range(len(total_loss)):
                    total_loss[j] += goal_losses[j]
                if plot_subgoals:
                    goal_loss_means, goal_loss_stddevs, goal_loss_collisions = self.compute_mean_and_stddev(goal_losses, collisions)
                    self.plot_mean_stddev(ax, goal_loss_means, goal_loss_stddevs, goal_loss_collisions, f"{label} {subgoal_key} loss", "Loss Mean and Stddev", plotting_config)
            total_loss_means, total_loss_stddevs, total_loss_collisions = self.compute_mean_and_stddev(total_loss, collisions)
            self.plot_mean_stddev(ax, total_loss_means, total_loss_stddevs, total_loss_collisions, f"{label} total loss", plotting_config)
        ax.set_title("\\textbf{Goal Loss}", fontsize=20)
        ax.set_ylabel("Loss Value", fontsize=20)
        ax.set_xlabel("Time Step", fontsize=20)
        ax.grid(True)
        ax.legend(fontsize=16)
        ax.set_xlim(xbounds[0], min(xbounds[1], len_data))
        # save / show
        loss_path = os.path.join(save_path, f"records/loss/{plotting_config['name']}") if save_path is not None else None
        self.save_fig(fig_goal, loss_path, show)

    def plot_losses_and_gradients(self, plotting_config:Dict[str,Dict[str,Tuple[List[int],List[str],List[Tuple[float,float]]]]], save_path:str=None, show:bool=False):
        subgoal_labels = ["Total", "Task", "Uncertainty"]
        
        for i, (ax_label, style) in enumerate(list(plotting_config["style"].items())):
            plotting_config["style"][f"{ax_label} frontal"] = {
                'label':     'axial motion',
                'linestyle': 'solid',
                'color':     'c',
                'linewidth': 2
            }
            plotting_config["style"][f"{ax_label} lateral"] = {
                'label':     'orthogonal motion',
                'linestyle': 'solid',
                'color':     'm',
                'linewidth': 2
            }
        
        xbounds = plotting_config["xbounds"]
        len_data = max(len(run_data["step"]) for variation in self.variations.values() for run_data in variation['data'].values())

        y_lim_start = min(int(len_data/2), 25)
        y_lim_end = min(len_data, xbounds[1])

        # TODO: use this to only plot variations from style config
        # for ax_label, variation in [(lab, var) for lab, var in plotting_config["axes"].items() if lab in plotting_config["style"]]:
        for ax_label, variation in plotting_config["axes"].items():
            fig_goal, axs = plt.subplots(3, 3, figsize=(21, 18))
            self.set_variation(variation)

            # HACK: exclude all runs except first, to get gradients that make sense instead of mean/stddev
            plotting_config['exclude_runs'] = [run for run in list(self.variations[self.current_variation_id]['data'].keys())[1:]]

            collisions = [run_data["collision"] for run_key, run_data in self.variations[self.current_variation_id]['data'].items() if run_key not in plotting_config["exclude_runs"]]
            
            total_loss = [[0.0]*len(run_data["step"]) for run_key, run_data in self.variations[self.current_variation_id]['data'].items() if run_key not in plotting_config["exclude_runs"]]
            task_loss = [run_data["goal_loss"]["target_distance"] for run_key, run_data in self.variations[self.current_variation_id]['data'].items() if run_key not in plotting_config["exclude_runs"]]
            uctty_loss = [run_data["goal_loss"]["target_distance_uncertainty"] for run_key, run_data in self.variations[self.current_variation_id]['data'].items() if run_key not in plotting_config["exclude_runs"]]
            for j in range(len(total_loss)):
                total_loss[j] = task_loss[j] + uctty_loss[j]

            action = np.array([run_data["rtf_action"] for run_key, run_data in self.variations[self.current_variation_id]['data'].items() if run_key not in plotting_config["exclude_runs"]])

            task_grad =  np.array([run_data["rtf_gradient"]["target_distance"] for run_key, run_data in self.variations[self.current_variation_id]['data'].items() if run_key not in plotting_config["exclude_runs"]])
            uctty_grad = np.array([run_data["rtf_gradient"]["target_distance_uncertainty"] for run_key, run_data in self.variations[self.current_variation_id]['data'].items() if run_key not in plotting_config["exclude_runs"]])
            total_grad = np.array([run_data["rtf_gradient"]["total"] for run_key, run_data in self.variations[self.current_variation_id]['data'].items() if run_key not in plotting_config["exclude_runs"]])
            
            # # TODO: utilize to get correlations of gradients and context variables
            # uctty = np.array([run_data["estimators"]["PolarTargetPos"]["uncertainty"] for run_key, run_data in self.variations[self.current_variation_id]['data'].items() if run_key not in plotting_config["exclude_runs"]])
            # if uctty.shape[2] == 5:
            #     datapoints = np.array([uctty_grad[0, :, 0], uctty[0, :, 2]])
            #     correlation = np.corrcoef(datapoints)
            #     print(f"Correlation between frontal gradient and frontal target vel uncertainty:")
            #     print(correlation[0,1])
            #     datapoints = np.array([uctty_grad[0, :, 1], uctty[0, :, 3]])
            #     correlation = np.corrcoef(datapoints)
            #     print(f"Correlation between lateral gradient and lateral target vel uncertainty:")
            #     print(correlation[0,1])

            for j, loss in enumerate([total_loss, task_loss, uctty_loss]):
                loss_means, loss_stddevs, loss_collisions = self.compute_mean_and_stddev(loss, collisions)
                self.plot_mean_stddev(axs[0][j], loss_means, loss_stddevs, loss_collisions, ax_label, plotting_config)
                min_y = np.min(loss_means[y_lim_start:y_lim_end])
                max_y = np.max(loss_means[y_lim_start:y_lim_end])
                axs[0][j].set_ylim(min_y + 0.1*(min_y-max_y), max_y + 0.1*(max_y-min_y))

            action_means_frontal, action_stddevs_frontal, action_collisions_frontal = self.compute_mean_and_stddev(action[:,:,0], collisions)
            action_means_lateral, action_stddevs_lateral, action_collisions_lateral = self.compute_mean_and_stddev(action[:,:,1], collisions)
            action_means_frontal = np.clip(action_means_frontal, -1, 1)
            action_means_lateral = np.clip(action_means_lateral, -1, 1)
            for j in [0,1,2]:
                self.plot_mean_stddev(axs[1][j], action_means_frontal, action_stddevs_frontal, action_collisions_frontal, f"{ax_label} frontal", plotting_config)
                self.plot_mean_stddev(axs[1][j], action_means_lateral, action_stddevs_lateral, action_collisions_lateral, f"{ax_label} lateral", plotting_config)
                min_y = np.min([np.min(data[y_lim_start:y_lim_end]) for data in [action_means_frontal, action_means_lateral]])
                max_y = np.max([np.max(data[y_lim_start:y_lim_end]) for data in [action_means_frontal, action_means_lateral]])
                axs[1][j].set_ylim(min_y + 0.1*(min_y-max_y), max_y + 0.1*(max_y-min_y))

            for j, grad in enumerate([total_grad, task_grad, uctty_grad]):
                grad_means_frontal, grad_stddevs_frontal, grad_collisions_frontal = self.compute_mean_and_stddev(grad[:,:,0], collisions)
                grad_means_lateral, grad_stddevs_lateral, grad_collisions_lateral = self.compute_mean_and_stddev(grad[:,:,1], collisions)
                self.plot_mean_stddev(axs[2][j], grad_means_frontal, grad_stddevs_frontal, grad_collisions_frontal, f"{ax_label} frontal", plotting_config)
                self.plot_mean_stddev(axs[2][j], grad_means_lateral, grad_stddevs_lateral, grad_collisions_lateral, f"{ax_label} lateral", plotting_config)
                min_y = np.min([np.min(data[y_lim_start:y_lim_end]) for data in [grad_means_frontal, grad_means_lateral]])
                max_y = np.max([np.max(data[y_lim_start:y_lim_end]) for data in [grad_means_frontal, grad_means_lateral]])
                axs[2][j].set_ylim(min_y + 0.1*(min_y-max_y), max_y + 0.1*(max_y-min_y))
                if j == 2:
                    extrafig, ax = plt.subplots(1, 1, figsize=(7, 6))
                    ax.axhline(y=0, color='black', linestyle='solid', linewidth=1)
                    self.plot_mean_stddev(ax, grad_means_frontal, grad_stddevs_frontal, grad_collisions_frontal, f"{ax_label} frontal", plotting_config)
                    self.plot_mean_stddev(ax, grad_means_lateral, grad_stddevs_lateral, grad_collisions_lateral, f"{ax_label} lateral", plotting_config)
                    ax.set_title("\\textbf{Uncertainty Gradient}", fontsize=20)
                    ax.set_ylabel("Gradient Magnitude", fontsize=20)
                    ax.set_xlabel("Time Step", fontsize=20)
                    ax.grid(True)
                    ax.legend(fontsize=16)
                    ax.set_xlim(0, min(xbounds[1], len_data))
                    ax.set_ylim(min_y + 0.1*(min_y-max_y), max_y + 0.1*(max_y-min_y))
                    extrafig.tight_layout()

            for j, subgoal_label in enumerate(subgoal_labels):
                axs[0][j].set_title("\\textbf{"+f"{subgoal_label} Loss" + "}", fontsize=20)
                axs[0][j].set_ylabel("Loss Value", fontsize=20)
                axs[1][j].set_title("\\textbf{Action }", fontsize=20)
                axs[1][j].set_ylabel("Action Magnitude (Normalized)", fontsize=20)
                axs[2][j].set_title("\\textbf{" + f"{subgoal_label} Gradient" + "}", fontsize=20)
                axs[2][j].set_ylabel("Gradient Magnitude", fontsize=20)
                for k in [0,1,2]:
                    axs[k][j].set_xlabel("Time Step", fontsize=20)
                    axs[k][j].grid(True)
                    axs[k][j].legend(fontsize=16)
                    axs[k][j].set_xlim(0, min(xbounds[1], len_data))

            # save / show
            loss_path = os.path.join(save_path, f"records/loss/goal_gradients_{ax_label}") if save_path is not None else None
            self.save_fig(fig_goal, loss_path, show)
            self.save_fig(extrafig, os.path.join(save_path, f"records/loss/gradient_{ax_label}"), show)

    # ================================== saving ==========================================

    def save(self, save_path:str):
        with open(os.path.join(save_path, "records/data.pkl"), 'wb') as f:
            pickle.dump(self.variations, f)

    @staticmethod
    def save_fig(fig:plt.Figure, save_path:str=None, show:bool=False):
        if save_path is not None:
            os.makedirs(os.path.dirname(save_path), exist_ok=True)
            fig.tight_layout()
            time.sleep(0.3)
            print(f"Saving to {save_path}...")
            fig.savefig(save_path+'.pdf', format='pdf')
        if show:
            fig.show()
        plt.close('all')