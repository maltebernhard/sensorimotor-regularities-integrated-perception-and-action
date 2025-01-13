from typing import Dict, List, Tuple
import matplotlib.pyplot as plt
import numpy as np
import torch
import yaml
import os

from components.helpers import rotate_vector_2d

# ==================================================================================

class AICONLogger:
    def __init__(self):
        self.data: Dict[int,dict] = {}
        self.run = 0

    # ======================================== logging ==========================================

    def log(self, step: int, time: float, estimators: Dict[str,Dict[str,torch.Tensor]], env_state: Dict[str,float], observation: Dict[str,Dict[str,float]]):
        real_state = {}
        for key in estimators.keys():
            if key[:5] == "Polar" and key[-3:] == "Pos":
                obj = key[5:-3].lower()
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

        if self.run not in self.data:
            self.data[self.run] = {
                "data": [],
                "desired_distance": env_state["desired_target_distance"],
                # more run information, like num obstacles?
            }

        self.data[self.run]["data"].append({
            "step": step,
            "time": time,
            "estimators": {
                estimator_key: {
                    "state_mean":   np.array(estimators[estimator_key]["state_mean"].tolist()),
                    "state_cov":    np.array(estimators[estimator_key]["state_cov"].tolist()),
                    "motion_noise": np.array(estimators[estimator_key]["forward_noise"].tolist()),
                } for estimator_key in estimators.keys()
            },
            "observation": {
                key: {
                    "measurement":  val["measurement"],
                    "noise":        val["noise"],
                } for key, val in observation.items()
            },
            "env_state": real_state
        })

    # ======================================= plotting ==========================================

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

    def plot_mean_stddev(self, subplot: plt.Axes, label: str, plot_type: str, error_means: np.ndarray, error_stddevs: np.ndarray):
        subplot.plot(error_means, label=f'{label} Mean')
        subplot.fill_between(
            range(len(error_means)), 
            error_means - error_stddevs,
            error_means + error_stddevs,
            alpha=0.2, label=f'{label} Stddev'
        )
        subplot.set_title(f'{plot_type} for {label}')
        subplot.set_xlabel('Time Step')
        subplot.set_ylabel(plot_type)
        subplot.grid()
        subplot.legend()

    def plot_runs(self, subplot: plt.Axes, state_index:int, label:str, plot_type:str, data:List[np.ndarray]):
        for run_state in data:
            subplot.plot(run_state[:, state_index])
        subplot.set_title(f'{plot_type} for {label}')
        subplot.set_xlabel('Time Step')
        subplot.set_ylabel(plot_type)
        subplot.grid()
        subplot.legend()

    def plot_states(self, state_dict:Dict[str,Tuple[List[int],List[str]]], avg=True, save_path:str=None):
        """
        Example state_dict:
        {
            "PolarTargetPos": ([0,1],["Distance","Angle"])
            "RobotVel":       ([0,1,2],["Frontal","Lateral","Rot"])
        }
        """
        for state_id, config in state_dict.items():
            states = [[] for _ in range(len(self.data))]
            indices = config[0]
            labels = config[1]

            for i, run in enumerate(self.data.values()):
                for entry in run["data"]:
                    state = entry["env_state"][state_id][indices]
                    # subtract desired distance from target distance
                    if state_id in ["PolarTargetPos", "PolarTargetDistance"] and 0 in indices:
                        state[indices.index(0)] -= run["desired_distance"]
                    # make sure angles are positive
                    # for idx in [i for i, label in enumerate(labels) if "Angle" in label]:
                    #     state[idx] = abs(state[idx])
                    states[i].append(state)

            fig, axs = self.create_subplots(len(indices))

            if avg:
                error_means, error_stddevs = self.compute_mean_and_stddev(np.array(states))
                for i in range(len(indices)):
                    self.plot_mean_stddev(axs[i], f"{state_id} {labels[i]}", "Goal Error", error_means[:, i], error_stddevs[:, i])
            else:
                for i in range(len(indices)):
                    self.plot_runs(axs[i], i, f"{state_id} {labels[i]}", "Goal Error", np.array(states))

            if save_path is not None:
                if not save_path.endswith('/'):
                    save_path += '/'
                os.makedirs(save_path, exist_ok=True)
                fig.savefig(save_path + f"goal_error_{state_id}.png")
            else:
                plt.show()

    def plot(self, state_dict:Dict[str,Tuple[List[int],List[str]]], save_path:str=None):
        for state_id, config in state_dict.items():
            # initialize lists for plotting
            states = [[] for _ in range(len(self.data))]
            #estimates = [[] for _ in range(len(self.data))]
            uncertainties = [[] for _ in range(len(self.data))]
            errors = [[] for _ in range(len(self.data))]
            indices = config[0]
            labels = config[1]
            # fill lists for plotting
            for i, run in enumerate(self.data.values()):
                for entry in run["data"]:
                    estimate = entry["estimators"][state_id]["state_mean"][indices]
                    #estimates.append(estimate)
                    uncertainties[i].append(np.sqrt(np.diag(entry["estimators"][state_id]["state_cov"])[indices]))
                    state = entry["env_state"][state_id][indices]
                    if state_id == "PolarTargetPos":
                        # NOTE: this is only valid if the estimator represents global target velocity
                        if 2 in indices:
                            rtf_vel = rotate_vector_2d(entry["env_state"][state_id][2], entry["env_state"]["RobotVel"][:2])
                            state[indices.index(2)] += rtf_vel[0]
                        if 3 in indices:
                            state[indices.index(3)] += entry["env_state"]["RobotVel"][2]
                    # TODO: the same might get relevant for decoupled PolarTargetDistance and PolarTargetAngle, but not for now
                    errors[i].append(estimate-state)
                    if state_id in ["PolarTargetPos", "PolarTargetDistance"]:
                        if 0 in indices:
                            state[indices.index(0)] -= run["desired_distance"]
                    states[i].append(state)

            # PLOT:
            # - state (goal error)
            # - estimation error: estimation - state
            # - uncertainty
            fig_avg, axs_avg = plt.subplots(3, len(indices), figsize=(7 * len(indices), 18))
            fig_runs, axs_runs = plt.subplots(3, len(indices), figsize=(7 * len(indices), 18))
            state_means, state_stddevs = self.compute_mean_and_stddev(np.array(states))
            est_error_means, est_error_stddevs = self.compute_mean_and_stddev(np.array(errors))
            uncertainty_means, uncertainty_stddevs = self.compute_mean_and_stddev(np.array(uncertainties))
            for i in range(len(indices)):
                self.plot_mean_stddev(axs_avg[0][i], f"{state_id} {labels[i]}", "State", state_means[:, i], state_stddevs[:, i])
                self.plot_mean_stddev(axs_avg[1][i], f"{state_id} {labels[i]}", "Estimation Error", est_error_means[:, i], est_error_stddevs[:, i])
                self.plot_mean_stddev(axs_avg[2][i], f"{state_id} {labels[i]}", "Estimation Uncertainty", uncertainty_means[:, i], uncertainty_stddevs[:, i])
                self.plot_runs(axs_runs[0][i], i, f"{state_id} {labels[i]}", "State", np.array(states))
                self.plot_runs(axs_runs[1][i], i, f"{state_id} {labels[i]}", "Estimation Error", np.array(errors))
                self.plot_runs(axs_runs[2][i], i, f"{state_id} {labels[i]}", "Estimation Uncertainty", np.array(uncertainties))


            # save that sheeeeeet
            if save_path is not None:
                if not save_path.endswith('/'):
                    save_path += '/'
                os.makedirs(save_path, exist_ok=True)
                fig_avg.savefig(save_path + f"avg_{state_id}.png")
                fig_runs.savefig(save_path + f"runs_{state_id}.png")
            else:
                fig_avg.show()
                fig_runs.show()
            

    def plot_estimation_error(self, estimator_id:str, value_indices:Dict[int,str]=None, save_path:str=None):
        errors = [[] for _ in range(len(self.data))]
        error_norms = [[] for _ in range(len(self.data))]

        for i, run in enumerate(self.data.values()):
            for entry in run["data"]:
                error = entry["estimators"][estimator_id]["state_mean"] - entry["env_state"][estimator_id]
                errors[i].append(error)
                error_norms[i].append(np.linalg.norm(error).item())

        errors = np.array(errors)
        error_norms = np.array(error_norms)
        error_means, error_stddevs = self.compute_mean_and_stddev(errors)
        norm_means, norm_stddevs = self.compute_mean_and_stddev(error_norms)

        plt.figure(figsize=(14, 6))

        if error_means.shape[1] > 1 and (value_indices is None or len(value_indices) > 1):
            plt.subplot(1, 2, 1)
            indices = value_indices.values() if value_indices else range(error_means.shape[1])
            labels = value_indices.keys() if value_indices else [f"State {i}" for i in range(error_means.shape[1])]
            for index, label in zip(indices, labels):
                plt.plot(error_means[:, index], label=f"{label} Mean")
                plt.fill_between(
                    range(len(error_means)), 
                    error_means[:, index] - error_stddevs[:, index], 
                    error_means[:, index] + error_stddevs[:, index], 
                    alpha=0.2, label=f"{label} Stddev"
                )
            plt.title(f'Estimation Error for {estimator_id}')
            plt.xlabel('Time step')
            plt.ylabel('Error')
            plt.grid()
            plt.legend()

            plt.subplot(1, 2, 2)
            plt.plot(norm_means, label='Norm Mean')
            plt.fill_between(
                range(len(norm_means)), 
                norm_means - norm_stddevs, 
                norm_means + norm_stddevs, 
                alpha=0.2, label='Norm Stddev'
            )
            plt.title(f'Norm of Estimation Error for {estimator_id}')
            plt.xlabel('Time step')
            plt.ylabel('Norm of Error')
            plt.grid()
            plt.legend()

        else:
            if value_indices is None:
                plt.plot(error_means, label='Error Mean')
                plt.fill_between(
                    range(len(error_means)), 
                    error_means - error_stddevs, 
                    error_means + error_stddevs, 
                    alpha=0.2, label='Error Stddevce'
                )
            else:
                index = value_indices.keys()[0]
                plt.plot(error_means[index], label=f'{value_indices.values()[0]} Error Mean')
                plt.fill_between(
                    range(len(error_means)), 
                    error_means[index] - error_stddevs[index],
                    error_means[index] + error_stddevs[index], 
                    alpha=0.2, label=f'{value_indices.values()[0]} Error Stddevce'
                )
            plt.title(f'Estimation Error for {estimator_id}')
            plt.xlabel('Time step')
            plt.ylabel('Error')
            plt.grid()
            plt.legend()

        if save_path is not None:
            if not save_path.endswith('/'):
                save_path += '/'
            os.makedirs(save_path, exist_ok=True)
            plt.savefig(save_path+f"/estimation_error_{estimator_id}.png")
        else:
            plt.show()

    # ================================ saving & loading ================================

    def load(self, file):
        def convert_to_numpy(obj):
            if isinstance(obj, list):
                return np.array(obj)
            elif isinstance(obj, dict):
                return {k: convert_to_numpy(v) for k, v in obj.items()}
            else:
                return obj

        with open(file, 'r') as f:
            loaded_data = yaml.safe_load(f)

        self.data = {int(k): {
            "desired_distance": v["desired_distance"],
            "data": [convert_to_numpy(item) for item in v["data"]]
        } for k, v in loaded_data.items()}
        print(f"Loaded {len(self.data)} runs")

    def save(self, record_dir):
        def convert_to_serializable(obj):
            # if isinstance(obj, torch.Tensor):
            #     return obj.tolist()
            if isinstance(obj, dict):
                return {k: convert_to_serializable(v) for k, v in obj.items()}
            elif isinstance(obj, list):
                return [convert_to_serializable(i) for i in obj]
            elif isinstance(obj, np.ndarray):
                return obj.tolist()
            elif isinstance(obj, np.generic):
                return obj.item()
            else:
                return obj

        serializable_data = convert_to_serializable(self.data)
        os.makedirs(record_dir, exist_ok=True)
        with open(os.path.join(record_dir, "data.yaml"), 'w') as f:
            f.write(yaml.dump(serializable_data))


    # TODO: tkinter plotting UI
    