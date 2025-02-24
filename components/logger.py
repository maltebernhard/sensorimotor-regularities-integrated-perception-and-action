import pickle
import time
from typing import Dict, List, Tuple
from matplotlib import pyplot as plt
import numpy as np
import torch
import os

import wandb

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

    def create_run_dict(self, estimators: Dict[str,Dict[str,torch.Tensor]], env_state: Dict[str,float], observation: Dict[str,Dict[str,float]], goal_loss: Dict[str,Dict[str,torch.Tensor]], action: torch.Tensor, gradients: Dict[str,Dict[str,torch.Tensor]]):
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
                "goal_loss": {goal_key: {
                    subgoal_key:        np.array([])     # goal loss function value
                    for subgoal_key in goal_loss[goal_key].keys()}
                for goal_key in goal_loss.keys()},
                "gradient": {goal_key: {
                    subgoal_key:        np.empty((0,) + gradients[goal_key][subgoal_key].shape)     # gradient of goal loss function
                    for subgoal_key in gradients[goal_key].keys()}
                for goal_key in gradients.keys()},
                "rtf_gradient": {goal_key: {
                    subgoal_key:        np.empty((0,) + gradients[goal_key][subgoal_key].shape)     # gradient of goal loss function
                    for subgoal_key in gradients[goal_key].keys()}
                for goal_key in gradients.keys()},
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
                if (obj=='target' and self.variation_config["moving_target"][0] != "stationary") or ('obstacle' in obj and self.variation_config["moving_obstacles"][0] != "stationary"):
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
            "goal_loss":                {goal_key: {subgoal_key: goal_loss[goal_key][subgoal_key].cpu().numpy() for subgoal_key in goal_loss[goal_key].keys()} for goal_key in goal_loss.keys()},
            "gradient":                 {goal_key: {subgoal_key: gradients[goal_key][subgoal_key].cpu().numpy() for subgoal_key in gradients[goal_key].keys()} for goal_key in gradients.keys()},
            "rtf_gradient":             {goal_key: {subgoal_key: np.append(rotate_vector_2d(-estimators["PolarTargetPos"]['mean'][1],gradients[goal_key][subgoal_key].cpu().numpy()[:2]),gradients[goal_key][subgoal_key].cpu().numpy()[2]) for subgoal_key in gradients[goal_key].keys()} for goal_key in gradients.keys()},
            "action":                   action.cpu().numpy(),
            "rtf_action":               np.append(rotate_vector_2d(-estimators["PolarTargetPos"]['mean'][1],action.cpu().numpy()[:2]),action.cpu().numpy()[2]),
        }

    def log_wandb(self, step_log: dict):
        if self.variation_config["moving_target"][0] != "stationary":
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
        for goal_key in step_log["goal_loss"].keys():
            for subgoal_key in step_log["goal_loss"][goal_key].keys():
                self.data[self.run_seed]["goal_loss"][goal_key][subgoal_key] = np.append(self.data[self.run_seed]["goal_loss"][goal_key][subgoal_key], [step_log["goal_loss"][goal_key][subgoal_key]], axis=0)
        for goal_key in step_log["gradient"].keys():
            for subgoal_key in step_log["gradient"][goal_key].keys():
                self.data[self.run_seed]["gradient"][goal_key][subgoal_key] = np.append(self.data[self.run_seed]["gradient"][goal_key][subgoal_key], [step_log["gradient"][goal_key][subgoal_key]], axis=0)
                self.data[self.run_seed]["rtf_gradient"][goal_key][subgoal_key] = np.append(self.data[self.run_seed]["rtf_gradient"][goal_key][subgoal_key], [step_log["rtf_gradient"][goal_key][subgoal_key]], axis=0)
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
        style_dict = plotting_dict['style'][label]
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
        color_cycle = ['b', 'g', 'r', 'c', 'm', 'y', 'k', 'orange', 'purple', 'brown', 'pink']
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
        subplot.set_title(title)
        subplot.set_xlabel("Time Step", fontsize=16)
        subplot.set_ylabel(y_label)
        max_steps = max(len(run) for run in data)
        subplot.set_xlim(0, max_steps)
        subplot.grid(True)
        subplot.legend()

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
    
    def plot_state_bars(self, plotting_config:Dict[str,Dict[str,Tuple[List[int],List[str],List[Tuple[float,float]]]]], save_path:str=None, show:bool=False):
        for state_id, config in plotting_config["states"].items():
            indices = np.array(config["indices"])
            if len(indices) == 1:
                fig, axs = plt.subplots(len(indices), 3, figsize=(21, 6))
                axs = [[ax] for ax in axs]
            else:
                fig, axs = plt.subplots(3, len(indices), figsize=(7 * len(indices), 18))
            
            # TODO: plot bars of accumulated [task error, est. error, est. uncertainty] with stddevs and outliers

            # TODO: make wind cause an acceleration in env, not a velocity change

    def plot_states(self, plotting_config:Dict[str,Dict[str,Tuple[List[int],List[str],List[Tuple[float,float]]]]], save_path:str=None, show:bool=False):
        if not "style" in plotting_config.keys():
            plotting_config=plotting_config.copy()
            plotting_config["style"] = {}
            colors = ['b', 'g', 'r', 'c', 'm', 'y', 'k', 'orange', 'purple', 'brown', 'pink']
            for i, ax_label in enumerate(plotting_config["axes"].keys()):
                plotting_config["style"][ax_label] = {'color': colors[i]}
        for state_id, config in plotting_config["states"].items():
            indices = np.array(config["indices"])
            ybounds = config["ybounds"]
            if len(indices) == 1:
                fig, axs = plt.subplots(len(indices), 3, figsize=(21, 6))
                axs = [[ax] for ax in axs]
            else:
                fig, axs = plt.subplots(3, len(indices), figsize=(7 * len(indices), 18))

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
                axs[0][i].axhline(y=list(plotting_config['axes'].values())[0]['desired_distance'], color='black', linestyle='--', linewidth=1, label='desired distance to target')
                
                axs[0][i].axhline(y=0, color='black', linestyle='solid', linewidth=1)
                axs[1][i].axhline(y=0, color='black', linestyle='solid', linewidth=1)
                axs[2][i].axhline(y=0, color='black', linestyle='solid', linewidth=1)
                if len(sensor_failures) > 0:
                    axs[0][i].axvspan(100, 200, color='grey', alpha=0.2, label=f'sensor failure for {sensor_failures}')
                    axs[1][i].axvspan(100, 200, color='grey', alpha=0.2, label=f'sensor failure for {sensor_failures}')
                    axs[2][i].axvspan(100, 200, color='grey', alpha=0.2, label=f'sensor failure for {sensor_failures}')

            for label, variation in plotting_config["axes"].items():
                self.set_variation(variation)
                states = [run_data["estimators"][state_id]["env_state"][:,indices] for run_key, run_data in self.variations[self.current_variation_id]['data'].items() if run_key not in plotting_config["exclude_runs"]]
                ucttys = [run_data["estimators"][state_id]["uncertainty"][:,indices] for run_key, run_data in self.variations[self.current_variation_id]['data'].items() if run_key not in plotting_config["exclude_runs"]]
                errors = [run_data["estimators"][state_id]["estimation_error"][:,indices] for run_key, run_data in self.variations[self.current_variation_id]['data'].items() if run_key not in plotting_config["exclude_runs"]]
                collisions = [run_data["collision"] for run_key, run_data in self.variations[self.current_variation_id]['data'].items() if run_key not in plotting_config["exclude_runs"]]

                state_means, state_stddevs, state_collisions = self.compute_mean_and_stddev(states, collisions)
                # HACK: overwrite distances to target at collision step, cause they slightly vary due to how much "inside" the agent is in the target at the step
                for i, col in enumerate(state_collisions):
                    state_collisions[i] = (col[0], 0.0)
                error_means, error_stddevs, error_collisions = self.compute_mean_and_stddev(errors, collisions)
                uctty_means, uctty_stddevs, uctty_collisions = self.compute_mean_and_stddev(ucttys, collisions)
                for i in range(len(indices)):
                    self.plot_mean_stddev(axs[0][i], state_means[:, i], state_stddevs[:, i], state_collisions, label, plotting_config)
                    self.plot_mean_stddev(axs[1][i], error_means[:, i], error_stddevs[:, i], error_collisions, label, plotting_config)
                    self.plot_mean_stddev(axs[2][i], uctty_means[:, i], uctty_stddevs[:, i], uctty_collisions, label, plotting_config)    
            
            titles   = ["Task State", "Estimation Error", "Estimation Uncertainty"]
            y_labels = ["Distance to Target", "Distance Estimation Error", "Distance Estimation Uncertainty (stddev)"]
            max_steps = max(len(run_data["step"]) for variation in self.variations.values() for run_data in variation['data'].values())
            for j in range(3):
                axs[j][i].set_title(titles[j], fontsize=16)
                axs[j][i].set_ylabel(y_labels[j], fontsize=16)
                axs[j][i].set_xlabel("Time Step", fontsize=16)
                axs[j][i].grid(True)
                axs[j][i].legend()
                axs[j][i].set_xlim(0, max_steps)
                if ybounds is not None:
                    axs[j][i].set_ylim(ybounds[j][i])
            
            # save / show
            path = os.path.join(save_path, f"records/state_{plotting_config['name']}.png") if save_path is not None else None
            self.save_fig(fig, path, show)

    def plot_state_runs(self, plotting_config:Dict[str,Dict[str,Tuple[List[int],List[str],List[Tuple[float,float]]]]], axs_id: str, runs: list[int], save_path:str=None, show:bool=False):
        for state_id, config in plotting_config["states"].items():
            indices = np.array(config["indices"])
            labels = config["labels"]
            ybounds = config["ybounds"]
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
                axs[1][i].set_ylim(ybounds[1][i])
                axs[2][i].set_ylim(ybounds[2][i])

            # save / show
            path = os.path.join(save_path, f"records/runs/{axs_id}.png") if save_path is not None else None
            self.save_fig(fig, path, show)

    def plot_goal_losses(self, plotting_config:Dict[str,Dict[str,Tuple[List[int],List[str],List[Tuple[float,float]]]]], plot_subgoals:bool, save_path:str=None, show:bool=False):
        if not "style" in plotting_config.keys():
            plotting_config["style"] = {}
            colors = ['b', 'g', 'r', 'c', 'm', 'y', 'k', 'orange', 'purple', 'brown', 'pink']
            for i, ax_label in enumerate(plotting_config["axes"].keys()):
                for label in [f"{ax_label} total loss"] + [f"{ax_label} {subgoal_key} loss" for subgoal_key in self.variations[self.current_variation_id]['data'][1]["goal_loss"]]:
                    plotting_config["style"][label] = {'color': colors[i]}
        else:
            for label, style in list(plotting_config["style"].items()):
                plotting_config["style"][f"{label} total loss"] = style
                for subgoal_key in self.variations[self.current_variation_id]['data'][1]["goal_loss"]:
                    plotting_config["style"][f"{label} {subgoal_key} loss"] = style
        num_goals = len(plotting_config["goals"])
        # Plot goal losses
        fig_goal, axs = plt.subplots(1, num_goals, figsize=(12, 10*num_goals))
        if num_goals == 1:
            axs = [axs]
        for i, (goal_key, config) in enumerate(plotting_config["goals"].items()):
            ybounds = config.get("ybounds")
            for label, variation in plotting_config["axes"].items():
                self.set_variation(variation)
                total_loss = [[0.0]*len(run_data["step"]) for run_key, run_data in self.variations[self.current_variation_id]['data'].items() if run_key not in plotting_config["exclude_runs"]]
                collisions = [run_data["collision"] for run_key, run_data in self.variations[self.current_variation_id]['data'].items() if run_key not in plotting_config["exclude_runs"]]
                for subgoal_key in self.variations[self.current_variation_id]['data'][1]["goal_loss"][goal_key].keys():
                    goal_losses = [run_data["goal_loss"][goal_key][subgoal_key] for run_key, run_data in self.variations[self.current_variation_id]['data'].items() if run_key not in plotting_config["exclude_runs"]]
                    for j in range(len(total_loss)):
                        total_loss[j] += goal_losses[j]
                    if plot_subgoals:
                        goal_loss_means, goal_loss_stddevs, goal_loss_collisions = self.compute_mean_and_stddev(goal_losses, collisions)
                        self.plot_mean_stddev(axs[i], goal_loss_means, goal_loss_stddevs, goal_loss_collisions, f"{label} {subgoal_key} loss", "Loss Mean and Stddev", plotting_config)
                total_loss_means, total_loss_stddevs, total_loss_collisions = self.compute_mean_and_stddev(total_loss, collisions)
                self.plot_mean_stddev(axs[i], total_loss_means, total_loss_stddevs, total_loss_collisions, f"{label} total loss", plotting_config)
            axs[i].set_title(f"{goal_key} Goal Loss", fontsize=16)
            axs[i].set_ylabel("Loss Value", fontsize=16)
            axs[i].set_xlabel("Time Step", fontsize=16)
            axs[i].grid(True)
            axs[i].legend()
            axs[i].set_xlim(0, max(len(run_data["step"]) for variation in self.variations.values() for run_data in variation['data'].values()))
            if ybounds is not None:
                axs[i].set_ylim(ybounds[i])
        # save / show
        loss_path = os.path.join(save_path, f"records/loss/{plotting_config['name']}.png") if save_path is not None else None
        self.save_fig(fig_goal, loss_path, show)

    def plot_losses_and_gradients(self, plotting_config:Dict[str,Dict[str,Tuple[List[int],List[str],List[Tuple[float,float]]]]], save_path:str=None, show:bool=False):
        subgoal_labels = ["total", "target_distance", "target_distance_uncertainty"]
        
        colors = ['b', 'g', 'r', 'c', 'm', 'y', 'k', 'orange', 'purple', 'brown', 'pink']
        if not "style" in plotting_config.keys():
            plotting_config["style"] = {}
            for i, ax_label in enumerate(plotting_config["axes"].keys()):
                for subgoal_label in subgoal_labels:
                    plotting_config["style"][f"{ax_label} {subgoal_label} loss"] = {'color': colors[i]}
                    plotting_config["style"][f"{ax_label} {subgoal_label} loss"] = {'color': colors[i]}
        else:
            for i, (ax_label, style) in enumerate(list(plotting_config["style"].items())):
                plotting_config["style"][f"{ax_label} frontal"] = {
                    'label':     'frontal',
                    'linestyle': 'dashed',
                    'color':     'c',
                    'linewidth': 1
                }
                plotting_config["style"][f"{ax_label} lateral"] = {
                    'label':     'lateral',
                    'linestyle': 'solid',
                    'color':     'm',
                    'linewidth': 1
                }

        for i, (goal_key, config) in enumerate(plotting_config["goals"].items()):
            ybounds = config.get("ybounds",None)
            for ax_label, variation in plotting_config["axes"].items():
                fig_goal, axs = plt.subplots(3, 3, figsize=(14, 12))
                self.set_variation(variation)

                # HACK: exclude all runs except first, to get gradients that make sense instead of mean/stddev
                plotting_config['exclude_runs'] = [run for run in list(self.variations[self.current_variation_id]['data'].keys())[1:]]
                collisions = [run_data["collision"] for run_key, run_data in self.variations[self.current_variation_id]['data'].items() if run_key not in plotting_config["exclude_runs"]]
                
                total_loss = [[0.0]*len(run_data["step"]) for run_key, run_data in self.variations[self.current_variation_id]['data'].items() if run_key not in plotting_config["exclude_runs"]]
                task_loss = [run_data["goal_loss"][goal_key]["target_distance"] for run_key, run_data in self.variations[self.current_variation_id]['data'].items() if run_key not in plotting_config["exclude_runs"]]
                uctty_loss = [run_data["goal_loss"][goal_key]["target_distance_uncertainty"] for run_key, run_data in self.variations[self.current_variation_id]['data'].items() if run_key not in plotting_config["exclude_runs"]]
                for j in range(len(total_loss)):
                    total_loss[j] = task_loss[j] + uctty_loss[j]

                action = np.array([run_data["rtf_action"] for run_key, run_data in self.variations[self.current_variation_id]['data'].items() if run_key not in plotting_config["exclude_runs"]])

                task_grad =  np.array([run_data["rtf_gradient"][goal_key]["target_distance"] for run_key, run_data in self.variations[self.current_variation_id]['data'].items() if run_key not in plotting_config["exclude_runs"]])
                uctty_grad = np.array([run_data["rtf_gradient"][goal_key]["target_distance_uncertainty"] for run_key, run_data in self.variations[self.current_variation_id]['data'].items() if run_key not in plotting_config["exclude_runs"]])
                total_grad = np.array([run_data["rtf_gradient"][goal_key]["total"] for run_key, run_data in self.variations[self.current_variation_id]['data'].items() if run_key not in plotting_config["exclude_runs"]])
                
                for j, loss in enumerate([total_loss, task_loss, uctty_loss]):
                    loss_means, loss_stddevs, loss_collisions = self.compute_mean_and_stddev(loss, collisions)
                    self.plot_mean_stddev(axs[0][j], loss_means, loss_stddevs, loss_collisions, ax_label, plotting_config)
                    axs[0][j].set_ylim(np.min(loss_means[100:]), np.max(loss_means[100:]))

                action_means_frontal, action_stddevs_frontal, action_collisions_frontal = self.compute_mean_and_stddev(action[:,:,0], collisions)
                action_means_lateral, action_stddevs_lateral, action_collisions_lateral = self.compute_mean_and_stddev(action[:,:,1], collisions)
                for j in [0,1,2]:
                    self.plot_mean_stddev(axs[1][j], action_means_frontal, action_stddevs_frontal, action_collisions_frontal, f"{ax_label} frontal", plotting_config)
                    self.plot_mean_stddev(axs[1][j], action_means_lateral, action_stddevs_lateral, action_collisions_lateral, f"{ax_label} lateral", plotting_config)
                    axs[1][j].set_ylim(np.min([np.min(data[100:]) for data in [action_means_frontal, action_means_lateral]]), np.max([np.max(data[100:]) for data in [action_means_frontal, action_means_lateral]]))

                for j, grad in enumerate([total_grad, task_grad, uctty_grad]):
                    grad_means_frontal, grad_stddevs_frontal, grad_collisions_frontal = self.compute_mean_and_stddev(grad[:,:,0], collisions)
                    grad_means_lateral, grad_stddevs_lateral, grad_collisions_lateral = self.compute_mean_and_stddev(grad[:,:,1], collisions)
                    self.plot_mean_stddev(axs[2][j], grad_means_frontal, grad_stddevs_frontal, grad_collisions_frontal, f"{ax_label} frontal", plotting_config)
                    self.plot_mean_stddev(axs[2][j], grad_means_lateral, grad_stddevs_lateral, grad_collisions_lateral, f"{ax_label} lateral", plotting_config)
                    axs[2][j].set_ylim(np.min([np.min(data[100:]) for data in [grad_means_frontal, grad_means_lateral]]), np.max([np.max(data[100:]) for data in [grad_means_frontal, grad_means_lateral]]))

                titles = ["loss", "action", "gradient"]
                y_labels = ["Loss Value", "Action Magnitude", "Gradient Magnitude"]
                for j, subgoal_label in enumerate(subgoal_labels):
                    
                    axs[1][j].set_title(f"{subgoal_label} action", fontsize=16)
                    axs[1][j].set_ylabel("Action Magnitude", fontsize=16)
                    axs[2][j].set_title(f"{subgoal_label} gradient", fontsize=16)
                    axs[2][j].set_ylabel("Gradient Magnitude", fontsize=16)
                    for k in [0,1,2]:
                        axs[k][j].set_title(f"{subgoal_label} {titles[k]}", fontsize=16)
                        axs[k][j].set_ylabel(y_labels[k], fontsize=16)
                        axs[k][j].set_xlabel("Time Step", fontsize=16)
                        axs[k][j].grid(True)
                        axs[k][j].legend()
                        axs[k][j].set_xlim(0, max(len(run_data["step"]) for variation in self.variations.values() for run_data in variation['data'].values()))
                    
                        if ybounds is not None:
                            axs[k][j].set_ylim(ybounds[k][j])
                # save / show
                loss_path = os.path.join(save_path, f"records/loss/{goal_key}_{ax_label}.png") if save_path is not None else None
                self.save_fig(fig_goal, loss_path, show)

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
            fig.savefig(save_path)
        if show:
            fig.show()
        plt.close('all')