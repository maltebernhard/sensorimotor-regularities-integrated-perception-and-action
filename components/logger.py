from typing import Dict, List, Tuple
import matplotlib.pyplot as plt
import numpy as np
import torch
import yaml
import os

# ==================================================================================

class AICONLogger:
    def __init__(self):
        self.data: Dict[int,List[dict]] = {}
        self.run = 0

    def log(self, step: int, time: float, estimators: Dict[str,Dict[str,torch.Tensor]], reality: Dict[str,torch.Tensor], observation: Dict[str,Dict[str,torch.Tensor]]):
        real_state = {}
        for key in estimators.keys():
            if key[:5] == "Polar" and key[-3:] == "Pos":
                obj = key[5:-3].lower()
                real_state[key] = torch.tensor([
                    reality[f"{obj}_distance"],
                    reality[f"{obj}_offset_angle"],
                    reality[f"{obj}_distance_dot"],
                    reality[f"{obj}_offset_angle_dot"]
                ], device=estimators[key]["state_mean"].device)
            elif key == "RobotVel":
                real_state[key] = torch.tensor([
                    reality["vel_frontal"],
                    reality["vel_lateral"],
                    reality["vel_rot"],
                ], device=estimators[key]["state_mean"].device)
            elif key[-6:] == "Radius":
                obj = key[:-6].lower()
                real_state[key] = torch.tensor(reality[f"{obj}_radius"], device=estimators[key]["state_mean"].device)

        if self.run not in self.data:
            self.data[self.run] = {
                "data": [],
                "desired_distance": reality["desired_target_distance"],
                # more run information, like num obstacles?
            }

        self.data[self.run]["data"].append({
            "step": step,
            "time": time,
            "estimators": estimators,
            "observation": observation,
            "reality": real_state,
        })

    # ======================================= generated code ==========================================

    # TODO: tkinter plotting UI

    @staticmethod
    def compute_mean_and_variance(data) -> Tuple[np.ndarray, np.ndarray]:
        data_array = np.array(data)
        mean = np.mean(data_array, axis=0)
        variance = np.var(data_array, axis=0)
        return mean, variance

    def plot_mean_variance(self, subplot, label:str, plot_type:str, error_means:np.ndarray, error_variances:np.ndarray):
        subplot.plot(error_means, label=f'{label} Mean')
        subplot.fill_between(
            range(len(error_means)), 
            error_means - error_variances,
            error_means + error_variances,
            alpha=0.2, label=f'{label} Variance'
        )
        subplot.set_title(f'{plot_type} for {label}')
        subplot.set_xlabel('Time Step')
        subplot.set_ylabel(plot_type)
        subplot.grid()
        subplot.legend()


    def plot_state(self, reality_id:str, offset:list=None, indices:List[int]=None, save_path:str=None):
        errors = [[] for _ in range(len(self.data))]

        if reality_id == "PolarTargetPos":
            indices = np.array([0, 1])
        else:
            indices = np.array([i for i in range(len(self.data[1]["data"][0]["reality"][reality_id]))])

        for i, run in enumerate(self.data.values()):
            if offset is None:
                if reality_id == "PolarTargetPos":
                    offset = torch.tensor([run["desired_distance"], 0.0], device=run["data"][0]["estimators"]["PolarTargetPos"]["state_mean"].device)
                else:
                    offset = torch.zeros(len(indices))
            for j, entry in enumerate(run["data"]):
                assert reality_id in entry["reality"], f"ID {reality_id} not found in run {i}"
                error = entry["reality"][reality_id][indices] - offset
                errors[i].append(error.cpu().numpy())
        
        errors = np.array(errors)
        error_means, error_variances = self.compute_mean_and_variance(errors)
        
        if reality_id == "PolarTargetPos":
            fig, axs = plt.subplots(1, 2, figsize=(14, 6))
            self.plot_mean_variance(axs[0], "Distance", "Goal Error", error_means[:,0], error_variances[:,0])
            self.plot_mean_variance(axs[1], "Angle", "Goal Error", error_means[:,1], error_variances[:,1])
        else:
            fig, axs = plt.subplots(1, 1, figsize=(14, 6))
            for i in range(len(indices)):
                self.plot_mean_variance(axs, f"{reality_id} {indices[i]}", "Goal Error", error_means[:,i], error_variances[:,i])

        if save_path is not None:
            os.makedirs(save_path, exist_ok=True)
            fig.savefig(save_path+f"/goal_error_{reality_id}.png")
        else:
            fig.show()

    def plot_estimation_error(self, estimator_id:str, value_keys:Dict[int,str]=None, save_path:str=None):
        errors = [[] for _ in range(len(self.data))]
        error_norms = [[] for _ in range(len(self.data))]

        for i, run in enumerate(self.data.values()):
            for j, entry in enumerate(run["data"]):
                assert estimator_id in entry["estimators"] and estimator_id in entry["reality"], f"Estimator ID {estimator_id} not found in run {i}"
                error = entry["estimators"][estimator_id]["state_mean"] - entry["reality"][estimator_id]
                errors[i].append(error.cpu().numpy())
                error_norms[i].append(torch.norm(error).item())

        errors = np.array(errors)
        error_norms = np.array(error_norms)
        error_means, error_variances = self.compute_mean_and_variance(errors)
        norm_means, norm_variances = self.compute_mean_and_variance(error_norms)

        plt.figure(figsize=(14, 6))

        if error_means.shape[1] > 1 and (value_keys is None or len(value_keys) > 1):
            plt.subplot(1, 2, 1)
            indices = value_keys.keys() if value_keys else range(error_means.shape[1])
            labels = value_keys.values() if value_keys else [f"State {i}" for i in range(error_means.shape[1])]
            for index, label in zip(indices, labels):
                plt.plot(error_means[:, index], label=f"{label} Mean")
                plt.fill_between(
                    range(len(error_means)), 
                    error_means[:, index] - error_variances[:, index], 
                    error_means[:, index] + error_variances[:, index], 
                    alpha=0.2, label=f"{label} Variance"
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
                norm_means - norm_variances, 
                norm_means + norm_variances, 
                alpha=0.2, label='Norm Variance'
            )
            plt.title(f'Norm of Estimation Error for {estimator_id}')
            plt.xlabel('Time step')
            plt.ylabel('Norm of Error')
            plt.grid()
            plt.legend()

        else:
            if value_keys is None:
                plt.plot(error_means, label='Error Mean')
                plt.fill_between(
                    range(len(error_means)), 
                    error_means - error_variances, 
                    error_means + error_variances, 
                    alpha=0.2, label='Error Variance'
                )
            else:
                index = value_keys.keys()[0]
                plt.plot(error_means[index], label=f'{value_keys.values()[0]} Error Mean')
                plt.fill_between(
                    range(len(error_means)), 
                    error_means[index] - error_variances[index],
                    error_means[index] + error_variances[index], 
                    alpha=0.2, label=f'{value_keys.values()[0]} Error Variance'
                )
            plt.title(f'Estimation Error for {estimator_id}')
            plt.xlabel('Time step')
            plt.ylabel('Error')
            plt.grid()
            plt.legend()

        if save_path is not None:
            os.makedirs(save_path, exist_ok=True)
            plt.savefig(save_path+f"/estimation_error_{estimator_id}.png")
        else:
            plt.show()

    def load(self, file):
        def convert_to_tensors(obj):
            if isinstance(obj, list):
                return torch.tensor(obj)
            elif isinstance(obj, dict):
                return {k: convert_to_tensors(v) for k, v in obj.items()}
            else:
                return obj

        with open(file, 'r') as f:
            loaded_data = yaml.safe_load(f)

        # self.data = {int(k): [convert_to_tensors(item) for item in v] for k, v in loaded_data.items()}
        self.data = {int(k): {
            "desired_distance": v["desired_distance"],
            "data": [convert_to_tensors(item) for item in v["data"]]
        } for k, v in loaded_data.items()}
        print(f"Loaded {len(self.data)} runs")
    
    # ======================================= generated code ==========================================

    def save(self, record_dir):
        def convert_tensors(obj):
            if isinstance(obj, torch.Tensor):
                return obj.tolist()
            elif isinstance(obj, dict):
                return {k: convert_tensors(v) for k, v in obj.items()}
            elif isinstance(obj, list):
                return [convert_tensors(i) for i in obj]
            elif isinstance(obj, np.ndarray):
                return obj.tolist()
            elif isinstance(obj, np.generic):
                return obj.item()
            else:
                return obj

        serializable_data = convert_tensors(self.data)
        os.makedirs(record_dir, exist_ok=True)
        with open(os.path.join(record_dir, "data.yaml"), 'w') as f:
            f.write(yaml.dump(serializable_data))

    