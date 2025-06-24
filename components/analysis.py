from datetime import datetime
import os
import pickle
import time
from typing import Dict, List, Tuple, Type
import numpy as np
import torch
import multiprocessing as mp
from tqdm import tqdm
import yaml
import matplotlib.pyplot as plt
import networkx as nx
import socket

from components.aicon import AICON, Logger
from configs.configs import ExperimentConfig as config
from drone_env_model.aicon import SMRAICON

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

# ========================================================================================================

class Runner:
    def __init__(self, run_config: dict, base_env_config: dict, variation: dict, variation_id=None, wandb_config:dict=None):
        self.env_config = self.generate_env_config(base_env_config, variation)
        self.aicon_type = {
            "smrs":            variation["smrs"],
            "controller":      variation["controller"],
            "distance_sensor": variation["distance_sensor"],
        }
        # NOTE: adapt this to allow various AICON configurations
        self.aicon = SMRAICON(self.env_config, self.aicon_type)

        self.num_steps = run_config['num_steps']
        self.base_seed = run_config['seed']
        self.seed = self.base_seed

        self.initial_action = torch.tensor([0.0,0.0,0.0]) if variation["control"] == "vel" else torch.tensor([0.1,0.0,0.0])

        self.render = run_config['render']
        self.video_record_path = None
        self.prints = run_config['prints']
        self.step_by_step = run_config['step_by_step']
        
        self.logger: Logger = Logger(variation, variation_id, wandb_config) if variation_id is not None else None
        self.num_run = self.logger.run_seed if self.logger is not None else 0

    def generate_env_config(self, base_env_config, variation):
        with open("components/env_config.yaml") as file:
            env_config = yaml.load(file, Loader=yaml.FullLoader)
            env_config["robot_sensor_angle"] = base_env_config["sensor_angle_deg"] / 180 * np.pi
            env_config["timestep"]           = base_env_config["timestep"]
            env_config["observation_noise"]  = variation["sensor_noise"]
            env_config["target_config"]      = variation["target_config"]
            env_config["obstacles"]          = variation["obstacles"]
            env_config["observation_loss"]   = variation["observation_loss"]
            env_config["fv_noise"]           = variation["fv_noise"]
            env_config["target_distance"]    = variation["desired_distance"]
            env_config["start_distance"]     = variation["start_distance"]
            env_config["action_mode"]        = 3 if variation["control"] == "vel" else 1
            env_config["wind"]               = variation["wind"]
        return env_config

    def run(self):
        self.num_run += 1
        self.seed = self.base_seed + self.num_run - 1
        if self.logger is not None:
            self.logger.run_seed = self.seed
        self.aicon.run(
            timesteps      = self.num_steps,
            env_seed       = self.seed,
            initial_action = self.initial_action,
            render         = self.render,
            prints         = self.prints,
            step_by_step   = self.step_by_step,
            logger         = self.logger,
            video_path     = self.video_record_path,
        )

# =====================================================================================

class Analysis:
    def __init__(self, experiment_config: dict):
        self.base_env_config: dict = experiment_config['base_env_config']

        self.experiment_config = experiment_config
        self.variations = experiment_config["variations"]
        self.variant_config = experiment_config["variation_config"]

        self.base_run_config: dict = self.create_base_run_config()
        
        self.logger = Plotter(self.variations)
        self.record_videos = experiment_config["record_videos"]
        
        self.record_dir = self.create_record_dir()
        self.timestamp = datetime.now().strftime('%Y-%m-%d-%H-%M')

        self.wandb_config = None
        if self.experiment_config['wandb'] and self.check_internet():
            if self.experiment_config.get("wandb_group", None) is None:
                self.experiment_config['wandb_group'] = f"{datetime.now().strftime('%Y_%m_%d_%H_%M')}"
                self.wandb_config = {
                    "wandb_project": f"aicon-{self.experiment_config['name']}",
                    "wandb_group": self.experiment_config['wandb_group'],
                }

    def create_base_run_config(self):
        return {
            "seed": 1,
            "render": False,
            "prints": 0,
            "step_by_step": False,
        }
    
    def get_num_runs(self):
        while "num_runs" not in self.variant_config.keys():
            try:
                self.variant_config["num_runs"] = int(input("Number of runs: "))
            except:
                print("Must be an integer.")
        return self.variant_config["num_runs"]
    
    def get_num_steps(self):
        while "num_steps" not in self.variant_config.keys():
            try:
                self.variant_config["num_steps"] = int(input("Number of steps: "))
            except:
                print("Must be an integer")
        return self.variant_config["num_steps"]

    # -------------------------------------------------------------------------

    def run_variation(self, base_run_config, base_env_config, variation_id, variation, wandb_config, num_runs, queue:mp.Queue, counter):
        # TODO: think about parallelizing even single seeded runs
        runner = Runner(
            run_config=base_run_config,
            base_env_config=base_env_config,
            variation=variation,
            variation_id=variation_id,
            wandb_config=wandb_config,
        )
        for run in range(num_runs):
            runner.run()
            with counter.get_lock():
                counter.value += 1
            if wandb_config is not None:
                runner.logger.end_wandb_run()
            queue.put((variation_id, run+1, runner.logger.data[run+1].copy()))
        del runner.aicon.env
        del runner.aicon
        del runner.logger
        del runner

    def add_variations(self, variations: List[dict]):
        self.variations += [variation for variation in variations if not any(all(variation[sub_key]==var[sub_key] for sub_key in var) for var in self.variations)]
        self.logger.add_variations(variations)
        self.experiment_config["variations"] = self.variations

    def run_analysis(self, variations: List[dict] = None):
        num_runs = self.get_num_runs()
        self.base_run_config["num_steps"] = self.get_num_steps()
        if variations is None:
            variations = self.variations
        total_configs = len(variations)
        total_runs = total_configs * num_runs
        mp.set_start_method('spawn', force=True)
        with tqdm(total=total_runs, desc="Running Analysis", position=0, leave=True, smoothing=0.0) as pbar:
            counter = mp.Value('i', 0)
            data_queue = mp.Queue()
            processes = []
            active_processes = []
            for variation in variations:
                var_id = self.logger.set_variation(variation)
                p = mp.Process(target=self.run_variation, args=(self.base_run_config, self.base_env_config, var_id, variation, self.wandb_config, num_runs, data_queue, counter))
                processes.append(p)
            run_save_counter = 0
            while len(processes) > 0 or len(active_processes) > 0:
                for p in active_processes:
                    if not p.is_alive():
                        p.join()
                        active_processes.remove(p)
                if len(active_processes) < mp.cpu_count() and len(processes) > 0:
                    active_processes.append(processes.pop(0))
                    active_processes[-1].start()
                pbar.update(counter.value - pbar.n)
                if not data_queue.empty():
                    var_id, num_run, data_dict = data_queue.get(timeout=5)
                    self.logger.variations[var_id]['data'][num_run] = data_dict
                    run_save_counter += 1
                time.sleep(.1)
            pbar.update(total_runs - pbar.n)
        while not data_queue.empty():
            var_id, num_run, data_dict = data_queue.get(timeout=5)
            self.logger.variations[var_id]['data'][num_run] = data_dict
            run_save_counter += 1
        for p in processes:
            p.join(timeout=10)  # Add a timeout for joining processes
            if p.is_alive():
                print(f"WARN: Process {p.pid} did not terminate within the timeout period.")
                p.terminate()
                p.join()
        print(f"Saved {run_save_counter} runs.")
        self.save()
    
    def save(self):
        os.makedirs(os.path.join(self.record_dir, 'configs'), exist_ok=True)
        os.makedirs(os.path.join(self.record_dir, 'records'), exist_ok=True)
        with open(os.path.join(self.record_dir, 'notes.txt'), 'w') as file:
            file.write(f"timestamp: {self.timestamp}\n")
        #self.visualize_graph(aicon=runner.aicon, save=True, show=False)
        with open(os.path.join(self.record_dir, 'configs/experiment_config.yaml'), 'w') as file:
            yaml.dump(self.experiment_config, file)
        self.logger.save(self.record_dir)
        print(f"Saved recorded runs to {self.record_dir}")

    @staticmethod
    def load(folder: str):
        with open(os.path.join(folder, 'configs/experiment_config.yaml'), 'r') as file:
            experiment_config = yaml.load(file, Loader=yaml.FullLoader)
        analysis = Analysis(experiment_config)
        #analysis.logger.load(folder)
        with open(os.path.join(folder, 'records/data.pkl'), 'rb') as f:
            yaml_dict: dict = pickle.load(f)
            for variation_id, data in yaml_dict.items():
                analysis.logger.variations[variation_id] = data
        analysis.record_dir = folder
        return analysis

    def create_record_dir(self):
        records_path = "./records"
        if not os.path.exists(records_path):
            os.makedirs(records_path)
        highest_num = 0
        for d in os.listdir(records_path):
            full_path = os.path.join(records_path, d)
            if os.path.isdir(full_path):
                parts = d.split("_")
                if "_".join(parts[:-1]) == self.experiment_config["name"]:
                    highest_num = max(highest_num, int(parts[-1]))
        record_dir = f"{records_path}/{self.experiment_config['name']}_{highest_num+1:02d}"
        return record_dir

    def run_demo(self, variation: dict, run_seed=None, step_by_step:bool=False, record_video=False):
        print("================ DEMO Variation ================")
        for key, value in variation.items():
            if type(value) == dict:
                print(f"{key}:")
                for k, v in value.items():
                    print(f"  {k}: {v}")
            else:
                print(f"{key}: {value}")
        print("===============================================")
        run_config = self.base_run_config.copy()
        run_config['num_steps'] = self.get_num_steps()
        run_config['render'] = True
        run_config['prints'] = 1
        run_config['step_by_step'] = step_by_step
        runner = Runner(
            variation = variation,
            run_config = run_config,
            base_env_config = self.base_env_config,
        )
        runner.base_seed = run_seed if run_seed is not None else 1
        if record_video:
            self.logger.set_variation(variation)
            config_id = self.logger.current_variation_id
            video_path = self.record_dir + f"/records/variation{config_id}_seed{runner.base_seed+runner.num_run-1}.mp4"
            runner.video_record_path = video_path
        runner.run()

    def plot_time(self, plotting_config: Dict[str,Tuple[List[int],List[str]]], save: bool=True, show: bool=False):
        """
        Plots the logged states according to the state dictionary.
        Args:
            plotting_config (Dict[str, Tuple[List[int], List[str]]], optional): 
                A dictionary where keys are estimator names and values are tuples containing 
                a list of state indices and a list of corresponding labels to plot.
        """
        self.logger.plot_states(plotting_config, abs=False, save_path=self.record_dir if save else None, show=show)
        self.logger.plot_states(plotting_config, abs=True, save_path=self.record_dir if save else None, show=show)
        self.logger.plot_accumulated_motion_states(plotting_config, save_path=self.record_dir if save else None, show=show)

    def plot_boxplots(self, plotting_config, save: bool=True, show:bool=False):
        self.logger.plot_state_boxplots(plotting_config, abs=False, save_path=self.record_dir if save else None, show=show)
        self.logger.plot_state_boxplots(plotting_config, abs=True, save_path=self.record_dir if save else None, show=show)
        self.logger.plot_accumulated_motion_boxplots(plotting_config, abs=False, save_path=self.record_dir if save else None, show=show)
        self.logger.plot_accumulated_motion_boxplots(plotting_config, abs=True, save_path=self.record_dir if save else None, show=show)

    def plot_state_runs(self, plotting_config: Dict[str,Tuple[List[int],List[str]]], config_id: str, runs: list[int]=None, save: bool=True, show: bool=False):
        self.logger.plot_state_runs(plotting_config, config_id, runs, save_path=self.record_dir if save else None, show=show)

    def plot_goal_losses(self, plotting_config:dict, plot_subgoals:bool=False, save:bool=True, show:bool=False):
        self.logger.plot_goal_losses(plotting_config, plot_subgoals, save_path=self.record_dir if save else None, show=show)

    def plot_loss_and_gradient(self, plotting_config:dict, save:bool=True, show:bool=False):
        self.logger.plot_losses_and_gradients(plotting_config, save_path=self.record_dir if save else None, show=show)

    def plot_collisions(self, plotting_config:dict, save:bool=True, show:bool=False):
        self.logger.plot_collisions(plotting_config, save_path=self.record_dir if save else None, show=show)

    @staticmethod
    def get_key_from_value(d: dict, value):
        for k, v in d.items():
            if v == value:
                return k
        print(f"Value {value} not found in dictionary {d}")
        return None

    @staticmethod
    def count_variations(variations, invariant_key: str):
        seen_variations = []
        for variation in variations:
            if variation[invariant_key] not in seen_variations:
                seen_variations.append(variation[invariant_key])
        return len(seen_variations)
    
    @staticmethod
    def check_internet(host="8.8.8.8", port=53, timeout=3):
        try:
            socket.setdefaulttimeout(timeout)
            socket.socket(socket.AF_INET, socket.SOCK_STREAM).connect((host, port))
            return True
        except socket.error as ex:
            print(ex)
            return False

    def plot_states_and_losses(self, time=True, boxplots=True, runs=True, losses=True, gradients=True, collisions=True):
        plotting_config: dict = self.variant_config["plotting_config"]
        plot_variation_values = {key: [variation[key] for variation in self.variations] for key in config.keys}
        #print(plot_variation_values)
        axes = {
            "_".join([self.get_key_from_value(vars(vars(config)[key]), variation[key]) for key in config.keys if (key not in plot_variation_values.keys() or len(plot_variation_values[key])>1) and self.count_variations(self.variations, key)>1]): {
                subkey: variation[subkey] for subkey in variation.keys()
            } for variation in self.variations if all([variation[key] in plot_variation_values[key] for key in plot_variation_values.keys()])
        }
        print("Variant keys:")
        for key in axes.keys():
            print(key)
        plotting_config.update({
            "name":         "main",
            "axes":         axes,
        })
        if time:
            self.plot_time(plotting_config)
        if boxplots:
            self.plot_boxplots(plotting_config)
        if gradients:
            self.plot_loss_and_gradient(plotting_config)
        if losses:
            self.plot_goal_losses(plotting_config)
        if collisions:
            self.plot_collisions(plotting_config)
        if runs:
            for ax_key in axes.keys():
                self.plot_state_runs(plotting_config, ax_key)

    # ------------------------------------------------------------------------------------------------

    def visualize_graph(self, aicon:Type[AICON], save:bool=True, show:bool=False):
        save_path = os.path.join(self.record_dir, 'configs') if save else None
        G = nx.DiGraph()

        pos = {}
        ai_nodes=[]
        estimator_nodes=[]
        measurement_model_nodes=[]
        observation_nodes=[]

        # Add nodes and edges for Active Interconnections
        for ai_key, ai in aicon.AIs.items():
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
        for mm_key, mm in aicon.MMs.items():
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
            plt.savefig(save_path + f"{aicon.type}_graph.png")
        if show:
            plt.show()
            input("Press Enter to continue...")

# ==================================================================================

class Plotter:
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
    
    def plot_sensor_failures(self, axs, plotting_config, indices, axspan:tuple):
        sensor_failures = []
        if all("Divergence" in var["smrs"] for var in plotting_config["axes"].values()) and all(any("visual_angle_dot" in loss_key for loss_key in var["observation_loss"].keys()) for var in plotting_config["axes"].values()):
            sensor_failures.append("divergence")
        elif all("Triangulation" in var["smrs"] for var in plotting_config["axes"].values()) and all(any("offset_angle_dot" in loss_key for loss_key in var["observation_loss"].keys()) for var in plotting_config["axes"].values()):
            sensor_failures.append("triangulation")
        elif all(ax['distance_sensor']=="distsensor" for ax in plotting_config["axes"].values()) and all(any("distance" in loss_key for loss_key in var["observation_loss"].keys()) for var in plotting_config["axes"].values()):
            sensor_failures.append("distance")
        elif all(ax['distance_sensor']=="distsensor" for ax in plotting_config["axes"].values()) and any(any("distance" in loss_key for loss_key in var["observation_loss"].keys()) for var in plotting_config["axes"].values()):
            print("WARN: not all variations have distance sensor failures")
        if len(set([var["desired_distance"] for var in plotting_config["axes"].values()])) > 1:
            print("WARN: not all variations have the same desired distance!")
        for i in range(len(indices)):
            axs[0][i].axhline(y=0, color='black', linestyle='solid', linewidth=2)
            axs[1][i].axhline(y=0, color='black', linestyle='solid', linewidth=2)
            axs[2][i].axhline(y=0, color='black', linestyle='solid', linewidth=2)
            if len(sensor_failures) > 0:
                axs[0][i].axvspan(axspan[0], axspan[1], color='grey', alpha=0.3, label=f'distance sensor failure')
                axs[1][i].axvspan(axspan[0], axspan[1], color='grey', alpha=0.3, label=f'distance sensor failure')
                axs[2][i].axvspan(axspan[0], axspan[1], color='grey', alpha=0.3, label=f'distance sensor failure')

    # ======================================= plotting ==========================================

    @staticmethod
    def plot_mean_stddev(subplot: plt.Axes, means: np.ndarray, stddevs: np.ndarray, collisions:List[Tuple[float,float]], label:str, plotting_dict:Dict[str,str], x_axis=None):
        style_dict = {key: plotting_dict['style'][label][key] for key in ['label', 'color', 'linestyle', 'linewidth']}
        
        if x_axis is None:
            x_axis = range(len(means))
        else:
            x_axis = x_axis[:len(means)]
        subplot.plot(
            x_axis,
            means,
            **style_dict
        )
        stddev_kwargs = {"color": style_dict["color"]}
        subplot.fill_between(
            x_axis, 
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
    def compute_mean_and_stddev(data: List[np.ndarray], col=None) -> Tuple[np.ndarray, np.ndarray, List[Tuple[int, float]]]:
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
        if col is not None:
            for i,c in enumerate(col):
                if c[-1]:
                    collisions.append((len(c) - 1, data[i][-1]))
        return np.array(means), np.array(stddevs), collisions
    
    # --------------------------------------------------------------------------------------------------

    def plot_state_boxplots(self, plotting_config:Dict[str,Dict[str,Tuple[List[int],List[str],List[Tuple[float,float]]]]], abs:bool=False, save_path:str=None, show:bool=False):
        len_data = max(len(run_data["step"]) for variation in self.variations.values() for run_data in variation['data'].values())
        xbounds = (plotting_config["xbounds"][0], min(plotting_config["xbounds"][1], len_data-1))

        if 'config_colors' in plotting_config and 'motion_colors' in plotting_config:
            for config_key in plotting_config["config_colors"]:
                for motion_key, motion_color in plotting_config["motion_colors"].items():
                    if f"{config_key}_{motion_key}" in plotting_config["style"]:
                        plotting_config["style"][f"{config_key}_{motion_key}"]["color"] = motion_color[0]
                        plotting_config["style"][f"{config_key}_{motion_key}"]["boxcolor"] = motion_color[1]

        for state_id, config in plotting_config["states"].items():
            ybounds = config.get("boxybounds", None)
            indices = np.array(config["indices"])
            if len(indices) == 1:
                task_fig, task_axs = plt.subplots(1, 1, figsize=(7, 6))
                err_fig, err_axs = plt.subplots(1, 1, figsize=(7, 6))
                uctty_fig, uctty_axs = plt.subplots(1, 1, figsize=(7, 6))
                axs = [[ax] for ax in [task_axs, err_axs, uctty_axs]]
            else:
                fig, axs = plt.subplots(3, len(indices), figsize=(6 * len(indices), 21))
                fig_abs, axs_abs = plt.subplots(3, len(indices), figsize=(6 * len(indices), 21))

            states = {}
            errors = {}
            ucttys = {}
            
            for label, variation in plotting_config["axes"].items():
                if 'motion' in label:
                    continue
                self.set_variation(variation)

                var_states = [run_data["estimators"][state_id]["env_state"][xbounds[0]:min(xbounds[1]+1,len_data),indices] for run_key, run_data in self.variations[self.current_variation_id]['data'].items() if run_key not in plotting_config["exclude_runs"]]
                desired_distance = [run_data["desired_distance"]*np.ones_like(run_data["estimators"][state_id]["env_state"][xbounds[0]:min(xbounds[1]+1,len_data),indices]) for run_key, run_data in self.variations[self.current_variation_id]['data'].items() if run_key not in plotting_config["exclude_runs"]]

                var_tasks = [state-desired for state, desired in zip(var_states, desired_distance)]
                var_ucttys = [run_data["estimators"][state_id]["uncertainty"][xbounds[0]:min(xbounds[1]+1,len_data),indices] for run_key, run_data in self.variations[self.current_variation_id]['data'].items() if run_key not in plotting_config["exclude_runs"]]
                var_errors = [run_data["estimators"][state_id]["estimation_error"][xbounds[0]:min(xbounds[1]+1,len_data),indices] for run_key, run_data in self.variations[self.current_variation_id]['data'].items() if run_key not in plotting_config["exclude_runs"]]
                var_collisions = [run_data["collision"] for run_key, run_data in self.variations[self.current_variation_id]['data'].items() if run_key not in plotting_config["exclude_runs"]]

                if abs:
                    var_tasks = [np.abs(task) for task in var_tasks]
                    var_errors = [np.abs(error) for error in var_errors]

                states[label] = np.array([np.mean(run) for run in var_tasks if len(run) >= xbounds[1]])
                errors[label] = np.array([np.mean(run) for run in var_errors if len(run) >= xbounds[1]])
                ucttys[label] = np.array([np.mean(run) for run in var_ucttys if len(run) >= xbounds[1]])

                if abs:
                    states[label] = np.abs(states[label])
                    errors[label] = np.abs(errors[label])

            if 'extended' not in save_path:
                for i in range(len(indices)):
                    for j, data in enumerate([states, errors, ucttys]):
                        for label in plotting_config["axes"].keys():
                            label_key = 'boxlabel' if 'boxlabel' in plotting_config['style'][label] else 'label'
                            plot_label = plotting_config['style'][label][label_key]
                            # use the label index as the x-position
                            label_index = list(plotting_config["style"].keys()).index(label)
                            axs[j][i].boxplot(
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
                        axs[j][i].tick_params(axis='x', labelsize=14)
                        max_val = max(data[label].max() for label in plotting_config["axes"].keys())
                        axs[j][i].set_ylim(ybounds[j][i] if ybounds is not None else (0.0, max_val * 1.1))
                        axs[j][i].axhline(y=0, color='black', linestyle='solid', linewidth=1)
                        axs[j][i].set_title("\\textbf{" + f"{['Task Error', 'Estimation Error', 'Estimation Uncertainty'][j]}" + "}", fontsize=16)
                        axs[j][i].set_ylabel(['Distance Offset from $d^\\ast$', 'Distance Estimation Error', 'Distance Estimate Standard Deviation'][j], fontsize=16)
                        axs[j][i].grid(True)#, axis='y')

            else:
                groups = ["nosmrs", "tri", "div", "both"] if 'exp2' in save_path else ["aicon", "trc"]
                group_labels = ["No SMRs", "Tri.", "Div.", "Tri. + Div."] if 'exp2' in save_path else ["AICON", "TRC"]
                motion_keys = ["stationary", "linear", "sine", "flight", "chase"]
                motion_labels = ["T stationary", "T moves linearly", "T moves on sine path", "T flees from R", "T chases R"]

                stylelist = []
                for group in groups:
                    for motion_key in motion_keys:
                        stylelist.append(f"{group}_{motion_key}")

                for i in range(len(indices)):
                    for j, data in enumerate([states, errors, ucttys]):
                        for label in plotting_config["axes"].keys():
                            group_index = next((i for i, group in enumerate(groups) if group in label), -1)
                            if group_index == -1:
                                print("WARN: group not found in label")
                                continue

                            label_key = 'boxlabel' if 'boxlabel' in plotting_config['style'][label] else 'label'
                            plot_label = plotting_config['style'][label][label_key]
                            # use the group index and label index as the x-position
                            
                            label_index = stylelist.index(label)
                            position = group_index * (len(plotting_config["axes"]) // len(groups)) + label_index % (len(plotting_config["axes"]) // len(groups)) + 1
                            axs[j][i].boxplot(
                                data[label],
                                positions=[position],
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
                        axs[j][i].tick_params(axis='x', labelsize=14)
                        max_val = max(data[label].max() for label in plotting_config["axes"].keys())
                        axs[j][i].set_ylim(ybounds[j][i] if ybounds is not None else (0.0, max_val * 1.1))
                        axs[j][i].axhline(y=0, color='black', linestyle='solid', linewidth=1)
                        axs[j][i].set_title("\\textbf{" + f"{['Task Error', 'Estimation Error', 'Estimation Uncertainty'][j]}" + "}", fontsize=16)
                        axs[j][i].set_ylabel(['Distance Offset from $d^\\ast$', 'Distance Estimation Error', 'Distance Estimate Standard Deviation'][j], fontsize=16)
                        axs[j][i].grid(True, axis='y')

                        # Set group labels on x-axis
                        group_positions = [group_index * (len(plotting_config["axes"]) // len(groups)) + (len(plotting_config["axes"]) // len(groups)) // 2 + 1 for group_index in range(len(groups))]
                        axs[j][i].set_xticks(group_positions)
                        axs[j][i].set_xticklabels(group_labels, fontsize=14)

                        # Add vertical lines to separate groups
                        for group_index in range(1, len(groups)):
                            axs[j][i].axvline(x=group_index * (len(plotting_config["axes"]) // len(groups)) + 0.5, color='grey', linestyle='--')

                        axs[j][i].legend(
                            handles=[
                                plt.Line2D([0], [0], color=plotting_config['style'][groups[0]+'_'+motion_keys[i]]['boxcolor'], marker='s', markersize=10, linestyle='None', label=motion_labels[i])
                            for i in range(len(motion_keys))],
                            loc='upper left' if 'exp1' in save_path or 'wind' in save_path else 'upper right',
                            fontsize=14
                        )

            for name, fig in zip(["task", "error", "uctty"], [task_fig, err_fig, uctty_fig]):
                path = os.path.join(save_path, "records/box/" + ("abs_" if abs and name!="uctty" else "") + f"{name}") if save_path is not None else None
                self.save_fig(fig, path, show)

    # --------------------------------------------------------------------------------------------------

    def plot_accumulated_motion_boxplots(self, plotting_config:Dict[str,Dict[str,Tuple[List[int],List[str],List[Tuple[float,float]]]]], abs:bool=False, save_path:str=None, show:bool=False):
        len_data = max(len(run_data["step"]) for variation in self.variations.values() for run_data in variation['data'].values())
        xbounds = (plotting_config["xbounds"][0], min(plotting_config["xbounds"][1], len_data-1))
        if 'config_colors' in plotting_config and 'motion_colors' in plotting_config:
            for config_key in plotting_config["config_colors"]:
                for motion_key, motion_color in plotting_config["motion_colors"].items():
                    if f"{config_key}_{motion_key}" in plotting_config["style"]:
                        plotting_config["style"][f"{config_key}_{motion_key}"].update(plotting_config["style"][f"{config_key}_linear"])
                        plotting_config["style"][f"{config_key}_{motion_key}"]["color"] = motion_color[0]
                        plotting_config["style"][f"{config_key}_{motion_key}"]["boxcolor"] = motion_color[1]
        
        for state_id, config in plotting_config["states"].items():
            ybounds = config.get("boxybounds", None)
            indices = np.array(config["indices"])
            if len(indices) == 1:
                task_fig, task_axs = plt.subplots(1, 1, figsize=(7, 6))
                err_fig, err_axs = plt.subplots(1, 1, figsize=(7, 6))
                uctty_fig, uctty_axs = plt.subplots(1, 1, figsize=(7, 6))
                axs = [[ax] for ax in [task_axs, err_axs, uctty_axs]]
            else:
                fig, axs = plt.subplots(3, len(indices), figsize=(6 * len(indices), 21))

            states = {}
            errors = {}
            ucttys = {}
            
            for label, variation in plotting_config["axes"].items():
                if 'motion' in label:
                    continue
                self.set_variation(variation)

                var_states = [run_data["estimators"][state_id]["env_state"][xbounds[0]:min(xbounds[1]+1,len_data),indices] for run_key, run_data in self.variations[self.current_variation_id]['data'].items() if run_key not in plotting_config["exclude_runs"]]
                desired_distance = [run_data["desired_distance"]*np.ones_like(run_data["estimators"][state_id]["env_state"][xbounds[0]:min(xbounds[1]+1,len_data),indices]) for run_key, run_data in self.variations[self.current_variation_id]['data'].items() if run_key not in plotting_config["exclude_runs"]]

                var_task_errors = [state-desired_distance for state, desired_distance in zip(var_states, desired_distance)]
                var_ucttys = [run_data["estimators"][state_id]["uncertainty"][xbounds[0]:min(xbounds[1]+1,len_data),indices] for run_key, run_data in self.variations[self.current_variation_id]['data'].items() if run_key not in plotting_config["exclude_runs"]]
                var_errors = [run_data["estimators"][state_id]["estimation_error"][xbounds[0]:min(xbounds[1]+1,len_data),indices] for run_key, run_data in self.variations[self.current_variation_id]['data'].items() if run_key not in plotting_config["exclude_runs"]]
                var_collisions = [run_data["collision"] for run_key, run_data in self.variations[self.current_variation_id]['data'].items() if run_key not in plotting_config["exclude_runs"]]

                if abs:
                    var_task_errors = [np.abs(err) for err in var_task_errors]
                    var_errors = [np.abs(err) for err in var_errors]

                states[label] = np.array([np.mean(task_error) for task_error, col in zip(var_task_errors, var_collisions) if len(col) >= xbounds[1]])
                errors[label] = np.array([np.mean(run) for run, col in zip(var_errors, var_collisions) if len(col) >= xbounds[1]])
                ucttys[label] = np.array([np.mean(run) for run, col in zip(var_ucttys, var_collisions) if len(col) >= xbounds[1]])

            if 'exp1_extended' in save_path:
                groups = ["aicon", "trc"]
            elif 'exp2' in save_path and 'extended' in save_path:
                groups = ["nosmrs", "tri", "div", "both"]
            else:
                print("Invalid Experiment. Abort.")
                return
            alt_states = {f"{group}_stationary": states[f"{group}_stationary"] for group in groups}
            alt_states.update({f"{group}_motion": np.concatenate([states[label] for label in states if group in label and "stationary" not in label]) for group in groups})
            alt_errors = {f"{group}_stationary": errors[f"{group}_stationary"] for group in groups}
            alt_errors.update({f"{group}_motion": np.concatenate([errors[label] for label in errors if group in label and "stationary" not in label]) for group in groups})
            alt_ucttys = {f"{group}_stationary": ucttys[f"{group}_stationary"] for group in groups}
            alt_ucttys.update({f"{group}_motion": np.concatenate([ucttys[label] for label in ucttys if group in label and "stationary" not in label]) for group in groups})
            
            with open(os.path.join(save_path, "records/stats.txt"), "w") as f:
                for label in alt_states.keys():
                    mean_alt_state = np.mean(alt_states[label])
                    f.write(f"Mean abs_alt_state for {label}: {mean_alt_state:.4f}\n")

                prefixes = set(label.split('_')[0] for label in alt_states.keys())
                comparisons = [(f"{prefix}_stationary", f"{prefix}_motion", prefix.capitalize()) for prefix in prefixes]

                for stationary_label, motion_label, name in comparisons:
                    stationary_mean = np.mean(alt_states[stationary_label])
                    motion_mean = np.mean(alt_states[motion_label])
                    percentual_increase = ((motion_mean - stationary_mean) / stationary_mean) * 100
                    f.write(f"{name}: task error increase stationary -> motion: {percentual_increase:.2f}%\n")

                for prefix in prefixes:
                    stationary_mean = np.mean(alt_states[f"{prefix}_stationary"])
                    motion_mean = np.mean(alt_states[f"{prefix}_motion"])
                    for other_prefix in prefixes:
                        if other_prefix != prefix:
                            other_stationary_mean = np.mean(alt_states[f"{other_prefix}_stationary"])
                            other_motion_mean = np.mean(alt_states[f"{other_prefix}_motion"])
                            percentual_increase_stationary = ((other_stationary_mean - stationary_mean) / stationary_mean) * 100
                            percentual_increase_motion = ((other_motion_mean - motion_mean) / motion_mean) * 100
                            f.write(
                                f"{prefix} vs. {other_prefix}: task error increase stationary: "
                                f"{percentual_increase_stationary:.2f}%, motion: {percentual_increase_motion:.2f}%\n"
                            )

            for i, data in enumerate([alt_states, alt_errors, alt_ucttys]):
                ax = axs[i][0]
                groups = ["nosmrs", "tri", "div", "both"] if 'exp2' in save_path else ["aicon", "trc"]
                group_labels = ["No SMRs", "Tri.", "Div.", "Tri. + Div."] if 'exp2' in save_path else ["AICON", "TRC"]
                motion_keys = ["stationary", "motion"]
                motion_labels = ["Stationary", "Motion"]

                # We'll place each group in a "block" on the x-axis, with two boxes (stationary, motion) per group.
                for group_index, group_name in enumerate(groups):
                    for motion_index, motion_key in enumerate(motion_keys):
                        label = f"{group_name}_{motion_key}"
                        if label not in data:
                            print(f"WARN: '{label}' not found in data")
                            continue

                        # Position = (group_index block) + (which motion in that block) + 1
                        position = group_index * len(motion_keys) + motion_index + 1
                        
                        label_key = 'boxlabel' if 'boxlabel' in plotting_config['style'][label] else 'label'
                        plot_label = plotting_config['style'][label][label_key]
                        #plot_label = f"{group_labels[group_index]} {motion_labels[motion_index]}"

                        ax.boxplot(
                            data[label],
                            positions=[position],
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
                            widths=0.5
                        )

                # For each group label on the x-axis, pick the midpoint of its two boxes
                group_positions = [
                    group_index * len(motion_keys) + (len(motion_keys) / 2) + 0.5
                    for group_index in range(len(groups))
                ]
                ax.set_xticks(group_positions)
                ax.set_xticklabels(group_labels, fontsize=14)

                # Vertical lines separating group blocks
                for sep_index in range(1, len(groups)):
                    ax.axvline(x=sep_index * len(motion_keys) + 0.5, color='grey', linestyle='--')

                # Legend for stationary vs. motion
                ax.legend(
                    handles=[
                        plt.Line2D([0], [0],
                                   color=plotting_config['style'][f"{groups[0]}_{motion_keys[m_idx]}"]['boxcolor'],
                                   marker='s', markersize=10, linestyle='None',
                                   label=motion_labels[m_idx])
                        for m_idx in range(len(motion_keys))
                    ],
                    loc=('upper left' if 'exp1' in save_path else 'upper right'),
                    fontsize=14
                )

                ax.tick_params(axis='x', labelsize=14)
                max_val = max(max(data[label]) for label in alt_states.keys())
                ax.set_ylim(0.0, max_val * 1.1)
                ax.set_title("\\textbf{" + f"{['Absolute Task (Distance) Error', 'Absolute Estimation Error', 'Estimation Uncertainty'][i]}" + "}", fontsize=16)
                ax.set_ylabel(['Distance Offset from $d^\\ast$', 'Distance Estimation Error', 'Distance Estimate Standard Deviation'][i], fontsize=16)
                ax.grid(True)#, axis='y')
                # ax.tick_params(axis='x', labelsize=14)
                # max_val = max(data[alt_label].max() for alt_label in alt_states.keys())
                # ax.set_ylim(0.0, max_val * 1.1)
                # #alt_ax.axhline(y=0, color='black', linestyle='solid', linewidth=1)
                # ax.set_title("\\textbf{" + f"{['Absolute Task (Distance) Error', 'Absolute Estimation Error', 'Estimation Uncertainty'][i]}" + "}", fontsize=16)
                # ax.set_ylabel(['Distance Offset from $d^\\ast$', 'Distance Estimation Error', 'Distance Estimate Standard Deviation'][i], fontsize=16)
                # ax.grid(True)#, axis='y')


            for name, fig in zip(["task", "error", "uctty"], [task_fig, err_fig, uctty_fig]):
                path = os.path.join(
                    save_path,
                    "records/box/accumulated_" + ("abs_" if abs and name != "uctty" else "") + f"{name}"
                ) if save_path is not None else None
                self.save_fig(fig, path, show)

    # --------------------------------------------------------------------------------------------------

    def plot_collisions(self, plotting_config:Dict[str,Dict[str,Tuple[List[int],List[str],List[Tuple[float,float]]]]], save_path:str=None, show:bool=False):
        if 'config_colors' in plotting_config and 'motion_colors' in plotting_config:
            for config_key in plotting_config["config_colors"]:
                for motion_key, motion_color in plotting_config["motion_colors"].items():
                    if f"{config_key}_{motion_key}" in plotting_config["style"]:
                        plotting_config["style"][f"{config_key}_{motion_key}"]["color"] = motion_color[0]

        len_data = max(len(run_data["step"]) for variation in self.variations.values() for run_data in variation['data'].values())
        xbounds = (plotting_config["xbounds"][0], min(plotting_config["xbounds"][1], len_data-1))
        
        ratio_collisions = {}
        total_runs = len(self.variations[self.current_variation_id]['data']) - len(plotting_config["exclude_runs"])
        for label, variation in plotting_config["axes"].items():
            self.set_variation(variation)
            var_collisions = [run_data["collision"] for run_key, run_data in self.variations[self.current_variation_id]['data'].items() if run_key not in plotting_config["exclude_runs"]]
            ratio_collisions[label] = sum(collisions_run[:min(len(collisions_run),xbounds[1])].sum() for collisions_run in var_collisions) / total_runs
        if not all(col == 0 for col in ratio_collisions.values()):
            fig, ax = plt.subplots(1, 1, figsize=(7, 6))
            ax.grid(axis='y')

            if 'extended' in save_path:
                groups = ["nosmrs", "tri", "div", "both"]
                group_labels = ["No SMRs", "Tri.", "Div.", "Tri. + Div."]
                motion_keys = ["stationary", "linear", "sine", "flight", "chase"]
                motion_labels = ["T stationary", "T moves linearly", "T moves on sine path", "T flees from R", "T chases R"]

                stylelist = []
                for group in groups:
                    for motion_key in motion_keys:
                        stylelist.append(f"{group}_{motion_key}")

                for group_index, group in enumerate(groups):
                    for motion_index, motion_key in enumerate(motion_keys):
                        label = f"{group}_{motion_key}"
                        ax.bar(
                            group_index * len(motion_keys) + motion_index,
                            ratio_collisions[label],
                            color=plotting_config['style'][label]['color']
                        )

                group_positions = [group_index * len(motion_keys) + len(motion_keys) // 2 for group_index in range(len(groups))]
                ax.set_xticks(group_positions)
                ax.set_xticklabels(group_labels, fontsize=14)

                for group_index in range(1, len(groups)):
                    ax.axvline(x=group_index * len(motion_keys) - 0.5, color='grey', linestyle='--')

                ax.legend(
                    handles=[
                        plt.Line2D([0], [0], color=plotting_config['style'][groups[0]+'_'+motion_keys[i]]['color'], marker='s', markersize=10, linestyle='None', label=motion_labels[i])
                    for i in range(len(motion_keys))],
                    loc='upper right',
                    fontsize=14,
                )

            else:
                labels = []
                for i, label in enumerate(plotting_config['style'].keys()):
                    ax.bar(
                        i,
                        ratio_collisions[label],
                        color=plotting_config['style'][label]['color']
                    )
                    labels.append(plotting_config['style'][label]['boxlabel' if 'boxlabel' in plotting_config['style'][label] else 'label'])
                ax.set_xticks(range(len(labels)))
                ax.set_xticklabels(labels)
            
            ax.tick_params(axis='x', labelsize=12)
            ax.set_title("\\textbf{Collisions }", fontsize=20)
            ax.set_ylabel("Avg. Collisions per Run", fontsize=16)
            ax.set_ylim(0, 1)#0.4)

            path = os.path.join(save_path, f"records/collisions") if save_path is not None else None
            self.save_fig(fig, path, show)

    # --------------------------------------------------------------------------------------------------

    def plot_states(self, plotting_config:Dict[str,Dict[str,Tuple[List[int],List[str],List[Tuple[float,float]]]]], abs:bool=False, save_path:str=None, show:bool=False):
        if 'config_colors' in plotting_config and 'motion_colors' in plotting_config:
            for config_key, config_color in plotting_config["config_colors"].items():
                for motion_key in plotting_config["motion_colors"]:
                    if f"{config_key}_{motion_key}" in plotting_config["style"]:
                        plotting_config["style"][f"{config_key}_{motion_key}"]["color"] = config_color
        
        len_data = max(len(run_data["step"]) for variation in self.variations.values() for run_data in variation['data'].values())
        xbounds = (plotting_config["xbounds"][0], min(plotting_config["xbounds"][1], len_data-1))

        for state_id, config in plotting_config["states"].items():
            indices = np.array(config["indices"])
            ybounds = config["ybounds"]
            
            absybounds = config.get("absybounds", None)
            if len(indices) == 1:
                task_fig, task_ax = plt.subplots(1, 1, figsize=(7, 6))
                err_fig, err_ax = plt.subplots(1, 1, figsize=(7, 6))
                uctty_fig, uctty_ax = plt.subplots(1, 1, figsize=(7, 6))
                axs = [[ax] for ax in [task_ax, err_ax, uctty_ax]]
            else:
                fig, axs = plt.subplots(3, len(indices), figsize=(7 * len(indices), 18))

            x_axis = np.arange(0, len_data * 0.05, 0.05)

            # HACK: plot sensor failures
            obsloss: dict = list(plotting_config["axes"].values())[0]["observation_loss"]
            if len(obsloss) > 0 and all(list(obsloss.values())[0][i][0] < len(x_axis) for i in range(len(list(obsloss.values())[0]))):
                self.plot_sensor_failures(axs, plotting_config, indices, (x_axis[list(obsloss.values())[0][0][0]], x_axis[min(list(obsloss.values())[0][0][1],len(x_axis)-1)]))

            for label in plotting_config["style"].keys():
                if "motion" in label:
                    continue
                variation = plotting_config["axes"][label]
                self.set_variation(variation)

                # NOTE: calculate mean_stddev with collisions and plot collisions
                # states = [run_data["estimators"][state_id]["env_state"][:,indices] for run_key, run_data in self.variations[self.current_variation_id]['data'].items() if run_key not in plotting_config["exclude_runs"]]
                # ucttys = [run_data["estimators"][state_id]["uncertainty"][:,indices] for run_key, run_data in self.variations[self.current_variation_id]['data'].items() if run_key not in plotting_config["exclude_runs"]]
                # errors = [run_data["estimators"][state_id]["estimation_error"][:,indices] for run_key, run_data in self.variations[self.current_variation_id]['data'].items() if run_key not in plotting_config["exclude_runs"]]
                #collisions = [run_data["collision"] for run_key, run_data in self.variations[self.current_variation_id]['data'].items() if run_key not in plotting_config["exclude_runs"]]

                # NOTE: calculate mean_stddev without collisions
                states = [run_data["estimators"][state_id]["env_state"][xbounds[0]:min(xbounds[1]+1,len_data),indices] for run_key, run_data in self.variations[self.current_variation_id]['data'].items() if run_key not in plotting_config["exclude_runs"] and not len(run_data["collision"])<=xbounds[1]]
                ucttys = [run_data["estimators"][state_id]["uncertainty"][xbounds[0]:min(xbounds[1]+1,len_data),indices] for run_key, run_data in self.variations[self.current_variation_id]['data'].items() if run_key not in plotting_config["exclude_runs"] and not len(run_data["collision"])<=xbounds[1]]
                errors = [run_data["estimators"][state_id]["estimation_error"][xbounds[0]:min(xbounds[1]+1,len_data),indices] for run_key, run_data in self.variations[self.current_variation_id]['data'].items() if run_key not in plotting_config["exclude_runs"] and not len(run_data["collision"])<=xbounds[1]]
                collisions = [run_data["collision"][xbounds[0]:min(xbounds[1]+1,len_data)] for run_key, run_data in self.variations[self.current_variation_id]['data'].items() if run_key not in plotting_config["exclude_runs"] and not len(run_data["collision"])<=xbounds[1]]

                desired_distance = [desired_distance*np.ones_like(np.array(states[i])) for i,desired_distance in enumerate([run_data["desired_distance"] for run_key, run_data in self.variations[self.current_variation_id]['data'].items() if run_key not in plotting_config["exclude_runs"] and not len(run_data["collision"])<=xbounds[1]])]
                states = [states[i]-desired_distance[i] for i in range(len(states))]

                if abs:
                    # calculate absolute arrors
                    states = [np.abs(states[i]) for i in range(len(states))]
                    errors = [np.abs(errors[i]) for i in range(len(errors))]

                state_means, state_stddevs, state_collisions = self.compute_mean_and_stddev(states, collisions)
                # HACK: overwrite distances to target at collisions, cause they slightly vary due to how much "inside" the agent is in the target at the step
                for i, col in enumerate(state_collisions):
                    state_collisions[i] = (col[0], 0.0)
                error_means, error_stddevs, error_collisions = self.compute_mean_and_stddev(errors, collisions)
                uctty_means, uctty_stddevs, uctty_collisions = self.compute_mean_and_stddev(ucttys, collisions)

                for i in range(len(indices)):
                    self.plot_mean_stddev(axs[0][i], state_means[xbounds[0]:min(xbounds[1]+1,len(state_means)), i], state_stddevs[xbounds[0]:min(xbounds[1]+1,len(state_means)), i], state_collisions, label, plotting_config, x_axis)
                    self.plot_mean_stddev(axs[1][i], error_means[xbounds[0]:min(xbounds[1]+1,len(state_means)), i], error_stddevs[xbounds[0]:min(xbounds[1]+1,len(state_means)), i], error_collisions, label, plotting_config, x_axis)
                    self.plot_mean_stddev(axs[2][i], uctty_means[xbounds[0]:min(xbounds[1]+1,len(state_means)), i], uctty_stddevs[xbounds[0]:min(xbounds[1]+1,len(state_means)), i], uctty_collisions, label, plotting_config, x_axis)    

            titles   = ["\\textbf{Task Error}", "\\textbf{Estimation Error}", "\\textbf{Estimation Uncertainty}"]
            y_labels = ["Distance Offset from $d^\\ast$", "Distance Estimation Error", "Distance Estimate Standard Deviation"]
            max_steps = max(len(run_data["step"]) for variation in self.variations.values() for run_data in variation['data'].values())
            for j in range(3):
                axs[j][i].set_title(titles[j], fontsize=20)
                axs[j][i].set_ylabel(y_labels[j], fontsize=20)
                axs[j][i].set_xlabel("Time $[s]$", fontsize=20)
                axs[j][i].grid(True)
                axs[j][i].legend(fontsize=14)
                axs[j][i].set_xlim(xbounds[0]*0.05, min(xbounds[1], max_steps)*0.05)
                if abs: 
                    if absybounds is not None:
                        axs[j][i].set_ylim(absybounds[j][i])
                    else:
                        if j == 0:
                            axs[j][i].set_ylim(0, ybounds[j][i][1] - self.variations[self.current_variation_id]['data'][1]["desired_distance"])
                        else:
                            axs[j][i].set_ylim(0, ybounds[j][i][1]-ybounds[j][i][0])
                else:
                    axs[j][i].set_ylim(ybounds[j][i])
            # save / show
            for name, fig in zip(["task", "error", "uctty"], [task_fig, err_fig, uctty_fig]):
                path = os.path.join(save_path, "records/time/" + ("abs_" if abs and name!="uctty" else "") + f"{name}") if save_path is not None else None
                self.save_fig(fig, path, show)

    # --------------------------------------------------------------------------------------------------

    def plot_accumulated_motion_states(self, plotting_config:Dict[str,Dict[str,Tuple[List[int],List[str],List[Tuple[float,float]]]]], save_path:str=None, show:bool=False):
        len_data = max(len(run_data["step"]) for variation in self.variations.values() for run_data in variation['data'].values())
        xbounds = (plotting_config["xbounds"][0], min(plotting_config["xbounds"][1], len_data-1))

        for state_id, config in plotting_config["states"].items():
            indices = np.array(config["indices"])
            ybounds = config["absybounds"] if "absybounds" in config else config["ybounds"]
            ybounds = config["boxybounds"] if "boxybounds" in config else config["ybounds"]

            x_axis = np.arange(0, len_data * 0.05, 0.05)

            states = {}
            ucttys = {}
            errors = {}
            alt_collisions = {}
            for label in plotting_config["style"].keys():
                if 'motion' in label:
                    continue
                variation = plotting_config["axes"][label]
                self.set_variation(variation)

                states[label] = [run_data["estimators"][state_id]["env_state"][xbounds[0]:min(xbounds[1]+1,len_data),indices] for run_key, run_data in self.variations[self.current_variation_id]['data'].items() if run_key not in plotting_config["exclude_runs"] and not len(run_data["collision"])<=xbounds[1]]
                ucttys[label] = [run_data["estimators"][state_id]["uncertainty"][xbounds[0]:min(xbounds[1]+1,len_data),indices] for run_key, run_data in self.variations[self.current_variation_id]['data'].items() if run_key not in plotting_config["exclude_runs"] and not len(run_data["collision"])<=xbounds[1]]
                errors[label] = [run_data["estimators"][state_id]["estimation_error"][xbounds[0]:min(xbounds[1]+1,len_data),indices] for run_key, run_data in self.variations[self.current_variation_id]['data'].items() if run_key not in plotting_config["exclude_runs"] and not len(run_data["collision"])<=xbounds[1]]
                alt_collisions[label] = [run_data["collision"][xbounds[0]:min(xbounds[1]+1,len_data)] for run_key, run_data in self.variations[self.current_variation_id]['data'].items() if run_key not in plotting_config["exclude_runs"] and not len(run_data["collision"])<=xbounds[1]]

                desired_distance = [desired_distance*np.ones_like(np.array(states[label][i])) for i,desired_distance in enumerate([run_data["desired_distance"] for run_key, run_data in self.variations[self.current_variation_id]['data'].items() if run_key not in plotting_config["exclude_runs"] and not len(run_data["collision"])<=xbounds[1]])]
                states[label] = [states[label][i]-desired_distance[i] for i in range(len(states[label]))]

                # calculate absolute arrors
                states[label] = [np.abs(states[label][i]) for i in range(len(states[label]))]
                errors[label] = [np.abs(errors[label][i]) for i in range(len(errors[label]))]

            # HACK
            if 'wind_extended' in save_path:
                global_shortest_length = min(
                    min(len(run) for run in states[lbl])
                    for lbl in states
                )
                for lbl in states:
                    states[lbl] = [run[:global_shortest_length] for run in states[lbl]]
                    errors[lbl] = [run[:global_shortest_length] for run in errors[lbl]]
                    ucttys[lbl] = [run[:global_shortest_length] for run in ucttys[lbl]]
                    alt_collisions[lbl] = [run[:global_shortest_length] for run in alt_collisions[lbl]]
                x_axis = x_axis[:global_shortest_length]

            if 'exp1_extended' in save_path:
                groups = ["aicon", "trc"]
            elif 'exp2' in save_path and 'extended' in save_path:
                groups = ["nosmrs", "tri", "div", "both"]
            else:
                print("Invalid Experiment. Abort.")
                return

            alt_states = {f"{group}_stationary": states[f"{group}_stationary"] for group in groups}
            alt_states.update({f"{group}_motion": np.concatenate([states[label] for label in states if group in label and "stationary" not in label]) for group in groups})
            alt_errors = {f"{group}_stationary": errors[f"{group}_stationary"] for group in groups}
            alt_errors.update({f"{group}_motion": np.concatenate([errors[label] for label in errors if group in label and "stationary" not in label]) for group in groups})
            alt_ucttys = {f"{group}_stationary": ucttys[f"{group}_stationary"] for group in groups}
            alt_ucttys.update({f"{group}_motion": np.concatenate([ucttys[label] for label in ucttys if group in label and "stationary" not in label]) for group in groups})

            task_fig, task_ax = plt.subplots(1, 1, figsize=(7, 6))
            err_fig, err_ax = plt.subplots(1, 1, figsize=(7, 6))
            uctty_fig, uctty_ax = plt.subplots(1, 1, figsize=(7, 6))
            axs = [[ax] for ax in [task_ax, err_ax, uctty_ax]]

            # HACK: plot sensor failures
            obsloss: dict = list(plotting_config["axes"].values())[0]["observation_loss"]
            if len(obsloss) > 0 and all(list(obsloss.values())[0][i][0] < len(x_axis) for i in range(len(list(obsloss.values())[0]))):
                self.plot_sensor_failures(axs, plotting_config, indices, (x_axis[list(obsloss.values())[0][0][0]], x_axis[min(list(obsloss.values())[0][0][1],len(x_axis)-1)]))

            for i, data in enumerate([alt_states, alt_errors, alt_ucttys]):
                ax = axs[i][0]
                for label in data.keys():
                    means, stddevs, collisions = self.compute_mean_and_stddev(np.array(data[label]))
                    self.plot_mean_stddev(ax, means[xbounds[0]:min(xbounds[1]+1,len(means)),0], stddevs[xbounds[0]:min(xbounds[1]+1,len(stddevs)),0], collisions, label, plotting_config, x_axis)

                ax.set_title("\\textbf{" + f"{['Task State', 'Estimation Error', 'Estimation Uncertainty'][i]}" + "}", fontsize=16)
                ax.set_ylabel(["Distance Offset from $d^\\ast$", "Distance Estimation Error", "Distance Estimate Standard Deviation"][i], fontsize=16)

            for ax in range(3):
                #axs[ax][0].set_ylim(0.0, 20.0) if 'exp1' in save_path else axs[ax][0].set_ylim(ybounds[ax][0])
                axs[ax][0].set_ylim(ybounds[ax][0])
                axs[ax][0].set_xlim(xbounds[0]*0.05, min(xbounds[1], len_data)*0.05)
                axs[ax][0].set_xlabel("Time $[s]$", fontsize=16)
                axs[ax][0].legend(fontsize=14, loc='upper left')
                axs[ax][0].grid(True)

            for name, fig in zip(["task", "error", "uctty"], [task_fig, err_fig, uctty_fig]):
                path = os.path.join(save_path, "records/time/accumulated_" + f"{name}") if save_path is not None else None
                self.save_fig(fig, path, show)

    # --------------------------------------------------------------------------------------------------

    def plot_state_runs(self, plotting_config:Dict[str,Dict[str,Tuple[List[int],List[str],List[Tuple[float,float]]]]], axs_id: str, runs: list[int], save_path:str=None, show:bool=False):
        len_data = max(len(run_data["step"]) for variation in self.variations.values() for run_data in variation['data'].values())
        xbounds = (plotting_config["xbounds"][0], min(plotting_config["xbounds"][1], len_data-1))
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
                axs[0][i].set_xlim(xbounds[0], min(xbounds[1], len_data))
                axs[1][i].set_ylim(ybounds[1][i])
                axs[1][i].set_xlim(xbounds[0], min(xbounds[1], len_data))
                axs[2][i].set_ylim(ybounds[2][i])
                axs[2][i].set_xlim(xbounds[0], min(xbounds[1], len_data))

            # save / show
            path = os.path.join(save_path, f"records/runs/{axs_id}") if save_path is not None else None
            self.save_fig(fig, path, show)

    # --------------------------------------------------------------------------------------------------

    def plot_goal_losses(self, plotting_config:Dict[str,Dict[str,Tuple[List[int],List[str],List[Tuple[float,float]]]]], plot_subgoals:bool, save_path:str=None, show:bool=False):
        for label, style in list(plotting_config["style"].items()):
            plotting_config["style"][f"{label} total loss"] = style
            for subgoal_key in self.variations[self.current_variation_id]['data'][1]["goal_loss"]:
                plotting_config["style"][f"{label} {subgoal_key} loss"] = style
        # Plot goal losses
        fig_goal, ax = plt.subplots(1, 1, figsize=(7, 6))
        
        len_data = max(len(run_data["step"]) for variation in self.variations.values() for run_data in variation['data'].values())
        xbounds = (plotting_config["xbounds"][0], min(plotting_config["xbounds"][1], len_data-1))
        
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

    # --------------------------------------------------------------------------------------------------

    def plot_losses_and_gradients(self, plotting_config:Dict[str,Dict[str,Tuple[List[int],List[str],List[Tuple[float,float]]]]], save_path:str=None, show:bool=False):
        subgoal_labels = ["Total", "Task", "Uncertainty"]
        
        for i, (ax_label, style) in enumerate(list(plotting_config["style"].items())):
            plotting_config["style"][f"{ax_label} frontal"] = {
                'label':     'axial motion (divergence) control',
                'linestyle': 'solid',
                'color':     'c',
                'linewidth': 2
            }
            plotting_config["style"][f"{ax_label} lateral"] = {
                'label':     'orthogonal motion (triangulation) control',
                'linestyle': 'solid',
                'color':     'm',
                'linewidth': 2
            }
            plotting_config["style"][f"{ax_label} loss"] = {
                'label':     'distance belief uncertainty',
                'linestyle': '--',
                'color':     'black',
                'linewidth':  2
            }
        
        len_data = max(len(run_data["step"]) for variation in self.variations.values() for run_data in variation['data'].values())
        xbounds = (plotting_config["xbounds"][0], min(plotting_config["xbounds"][1], len_data-1))
        x_axis = np.arange(0, len_data * 0.05, 0.05)

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
            
            uctty = np.array([run_data["estimators"]["PolarTargetPos"]["uncertainty"] for run_key, run_data in self.variations[self.current_variation_id]['data'].items() if run_key not in plotting_config["exclude_runs"]])
            # TODO: utilize to get correlations of gradients and context variables
            # if "sine" in ax_label:
            #     correlation_range = xbounds
            #     datapoints = np.array([np.abs(uctty_grad[0, correlation_range[0]:correlation_range[1], 0]), uctty[0, correlation_range[0]:correlation_range[1], 2]])
            #     correlation = np.corrcoef(datapoints)
            #     print(f"{ax_label}: abs frontal grad / frontal target vel uncertainty: {correlation[0, 1]}")
            #     datapoints = np.array([uctty_grad[0, correlation_range[0]:correlation_range[1], 0], uctty[0, correlation_range[0]:correlation_range[1], 2]])
            #     correlation = np.corrcoef(datapoints)
            #     print(f"{ax_label}: frontal grad / frontal target vel uncertainty: {correlation[0, 1]}")
            #     datapoints = np.array([np.abs(uctty_grad[0, correlation_range[0]:correlation_range[1], 0]), uctty[0, correlation_range[0]:correlation_range[1], 3]])
            #     correlation = np.corrcoef(datapoints)
            #     print(f"{ax_label}: abs frontal grad / lateral target vel uncertainty: {correlation[0, 1]}")
            #     datapoints = np.array([uctty_grad[0, correlation_range[0]:correlation_range[1], 0], uctty[0, correlation_range[0]:correlation_range[1], 3]])
            #     correlation = np.corrcoef(datapoints)
            #     print(f"{ax_label}: frontal grad / lateral target vel uncertainty: {correlation[0, 1]}")
            #     datapoints = np.array([np.abs(uctty_grad[0, correlation_range[0]:correlation_range[1], 1]), uctty[0, correlation_range[0]:correlation_range[1], 2]])
            #     correlation = np.corrcoef(datapoints)
            #     print(f"{ax_label}: abs lateral grad / frontal target vel uncertainty: {correlation[0, 1]}")
            #     datapoints = np.array([uctty_grad[0, correlation_range[0]:correlation_range[1], 1], uctty[0, correlation_range[0]:correlation_range[1], 2]])
            #     correlation = np.corrcoef(datapoints)
            #     print(f"{ax_label}: lateral grad / frontal target vel uncertainty: {correlation[0, 1]}")
            #     datapoints = np.array([np.abs(uctty_grad[0, correlation_range[0]:correlation_range[1], 1]), uctty[0, correlation_range[0]:correlation_range[1], 3]])
            #     correlation = np.corrcoef(datapoints)
            #     print(f"{ax_label}: abs lateral grad / lateral target vel uncertainty: {correlation[0, 1]}")
            #     datapoints = np.array([uctty_grad[0, correlation_range[0]:correlation_range[1], 1], uctty[0, correlation_range[0]:correlation_range[1], 3]])
            #     correlation = np.corrcoef(datapoints)
            #     print(f"{ax_label}: lateral grad / lateral target vel uncertainty: {correlation[0, 1]}")
            #     datapoints = np.array([np.abs(uctty_grad[0, correlation_range[0]:correlation_range[1], 0]), uctty[0, correlation_range[0]:correlation_range[1], 4]])
            #     correlation = np.corrcoef(datapoints)
            #     print(f"{ax_label}: abs frontal grad / radius uncertainty: {correlation[0, 1]}")
            #     datapoints = np.array([uctty_grad[0, correlation_range[0]:correlation_range[1], 0], uctty[0, correlation_range[0]:correlation_range[1], 4]])
            #     correlation = np.corrcoef(datapoints)
            #     print(f"{ax_label}: frontal grad / radius uncertainty: {correlation[0, 1]}")
            #     datapoints = np.array([np.abs(uctty_grad[0, correlation_range[0]:correlation_range[1], 1]), uctty[0, correlation_range[0]:correlation_range[1], 4]])
            #     correlation = np.corrcoef(datapoints)
            #     print(f"{ax_label}: abs lateral grad / radius uncertainty: {correlation[0, 1]}")
            #     datapoints = np.array([uctty_grad[0, correlation_range[0]:correlation_range[1], 1], uctty[0, correlation_range[0]:correlation_range[1], 4]])
            #     correlation = np.corrcoef(datapoints)
            #     print(f"{ax_label}: lateral grad / radius uncertainty: {correlation[0, 1]}")
            #     datapoints = np.array([np.abs(uctty_grad[0, correlation_range[0]:correlation_range[1], 0]), np.array(uctty_loss[0])[correlation_range[0]:correlation_range[1]]])
            #     correlation = np.corrcoef(datapoints)
            #     print(f"{ax_label}: abs frontal grad / general uctty loss: {correlation[0, 1]}")
            #     datapoints = np.array([uctty_grad[0, correlation_range[0]:correlation_range[1], 0], np.array(uctty_loss[0])[correlation_range[0]:correlation_range[1]]])
            #     correlation = np.corrcoef(datapoints)
            #     print(f"{ax_label}: frontal grad / general uctty loss: {correlation[0, 1]}")
            #     datapoints = np.array([np.abs(uctty_grad[0, correlation_range[0]:correlation_range[1], 1]), np.array(uctty_loss[0])[correlation_range[0]:correlation_range[1]]])
            #     correlation = np.corrcoef(datapoints)
            #     print(f"{ax_label}: abs lateral grad / general uctty loss: {correlation[0, 1]}")
            #     datapoints = np.array([uctty_grad[0, correlation_range[0]:correlation_range[1], 1], np.array(uctty_loss[0])[correlation_range[0]:correlation_range[1]]])
            #     correlation = np.corrcoef(datapoints)
            #     print(f"{ax_label}: lateral grad / general uctty loss: {correlation[0, 1]}")
            #     datapoints = np.array([uctty_grad[0, correlation_range[0]:correlation_range[1], 0], uctty_grad[0, correlation_range[0]:correlation_range[1], 1]])
            #     correlation = np.corrcoef(datapoints)
            #     print(f"{ax_label}: frontal grad / lateral grad: {correlation[0, 1]}")
            #     datapoints = np.array([np.abs(uctty_grad[0, correlation_range[0]:correlation_range[1], 0]), np.abs(uctty_grad[0, correlation_range[0]:correlation_range[1], 1])])
            #     correlation = np.corrcoef(datapoints)
            #     print(f"{ax_label}: abs frontal grad / abs lateral grad: {correlation[0, 1]}")


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

                    if "trc" in ax_label:
                        self.plot_mean_stddev(ax, -np.abs(grad_means_frontal), grad_stddevs_frontal, grad_collisions_frontal, f"{ax_label} frontal", plotting_config, x_axis)
                    else:
                        self.plot_mean_stddev(ax, grad_means_frontal, grad_stddevs_frontal, grad_collisions_frontal, f"{ax_label} frontal", plotting_config, x_axis)
                    self.plot_mean_stddev(ax, grad_means_lateral, grad_stddevs_lateral, grad_collisions_lateral, f"{ax_label} lateral", plotting_config, x_axis)
                    ax.set_title("\\textbf{" + ("AICON " if "aicon" in ax_label else "TRC ") + "SMR Control Components}", fontsize=20)
                    
                    # self.plot_mean_stddev(ax, grad_means_frontal, grad_stddevs_frontal, grad_collisions_frontal, f"{ax_label} frontal", plotting_config)
                    # self.plot_mean_stddev(ax, grad_means_lateral, grad_stddevs_lateral, grad_collisions_lateral, f"{ax_label} lateral", plotting_config)
                    # ax.set_title("\\textbf{Uncertainty Gradient}", fontsize=20)
                    ax.set_ylabel("Control Output", fontsize=20)
                    ax.set_xlabel("Time $[s]$", fontsize=20)
                    ax.grid(True)
                    ax.legend(fontsize=16, loc='lower left')
                    ax.set_xlim(xbounds[0]*0.05, min(xbounds[1], len_data)*0.05)
                    #ax.set_ylim(min_y + 0.1*(min_y-max_y), max_y + 0.1*(max_y-min_y))
                    ax.set_ylim(-6,4)

                    ax2 = ax.twinx()
                    uctty_means, uctty_stddevs, uctty_collisions = self.compute_mean_and_stddev(uctty[:,:,0], collisions)
                    self.plot_mean_stddev(ax2, uctty_means, uctty_stddevs, uctty_collisions, f"{ax_label} loss", plotting_config, x_axis)
                    ax2.set_ylabel("Distance Belief Uncertainty", fontsize=20)
                    ax2.legend(fontsize=16, loc='lower left', bbox_to_anchor=(0, 0.16))
                    ax2.set_ylim(-6, 4)

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
                    axs[k][j].set_xlim(xbounds[0], min(xbounds[1], len_data))

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
            print(f"Saving plot to {save_path}...")
            fig.savefig(save_path+'.pdf', format='pdf')
        if show:
            fig.show()
        plt.close('all')
