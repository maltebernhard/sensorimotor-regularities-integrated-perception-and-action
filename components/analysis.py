from datetime import datetime
import os
import pickle
import sys
import time
from typing import Dict, List, Tuple, Type
import numpy as np
import torch
from tqdm import tqdm
import yaml
import matplotlib.pyplot as plt
import networkx as nx

from components.aicon import AICON
from components.logger import AICONLogger, VariationLogger
# from models.old.base.aicon import BaseAICON
# from models.old.experiment1.aicon import Experiment1AICON
# from models.old.experiment_foveal_vision.aicon import ExperimentFovealVisionAICON
# from models.old.divergence.aicon import DivergenceAICON
# from models.old.global_vels.aicon import GlobalVelAICON
# from models.old.single_estimator.aicon import SingleEstimatorAICON
from models.smc_ais.aicon import SMCAICON

# from models.old.even_older.experiment_estimator.aicon import ContingentEstimatorAICON
# from models.old.even_older.experiment_visibility.aicon import VisibilityAICON
import multiprocessing as mp

# ========================================================================================================

class Runner:
    def __init__(self, run_config: dict, base_env_config: dict, variation: dict, variation_id=None, wandb_config:dict=None):
        self.model = "SMC"

        self.env_config = self.generate_env_config(base_env_config, variation)
        self.aicon_type = {
            "smcs":            variation["smcs"],
            "controller":      variation["controller"],
            "distance_sensor": variation["distance_sensor"],
        }

        self.num_steps = run_config['num_steps']
        self.seed = run_config['seed']
        self.initial_action = run_config['initial_action']

        self.render = run_config['render']
        self.video_record_path = None
        self.prints = run_config['prints']
        self.step_by_step = run_config['step_by_step']
        
        self.logger: VariationLogger = VariationLogger(variation, variation_id, wandb_config) if variation_id is not None else None
        self.aicon = self.create_model()

        self.num_run = self.logger.run if self.logger is not None else 0

    def generate_env_config(self, base_env_config, variation):
        with open("environment/env_config.yaml") as file:
            env_config = yaml.load(file, Loader=yaml.FullLoader)
            env_config["num_obstacles"] = base_env_config["num_obstacles"]
            env_config["robot_sensor_angle"] = base_env_config["sensor_angle_deg"] / 180 * np.pi
            env_config["timestep"] = base_env_config["timestep"]
            env_config["observation_noise"] = variation["sensor_noise"]
            env_config["moving_target"] = variation["moving_target"]
            env_config["observation_loss"] = variation["observation_loss"]
            env_config["fv_noise"] = variation["fv_noise"]
            env_config["target_distance"] = variation["desired_distance"]
            env_config["action_mode"] = 3 if variation["control"] == "vel" else 1
            env_config["wind"] = variation["wind"]
        return env_config

    def run(self):
        self.num_run += 1
        if self.logger is not None:
            self.logger.run = self.num_run
        self.aicon.run(
            timesteps=self.num_steps,
            env_seed=self.seed+self.num_run-1,
            initial_action=torch.tensor(self.initial_action),
            render=self.render,
            prints=self.prints,
            step_by_step=self.step_by_step,
            logger=self.logger,
            video_path=self.video_record_path,
        )
    
    def create_model(self):
        if self.model == "SMC":                    return SMCAICON(self.env_config, self.aicon_type)
        # elif self.model == "Base":                   return BaseAICON(self.env_config)
        # elif self.model == "Experiment1":            return Experiment1AICON(self.env_config, self.aicon_type)
        # elif self.model == "ExperimentFovealVision": return ExperimentFovealVisionAICON(self.env_config)
        # elif self.model == "SingleEstimator":        return SingleEstimatorAICON(self.env_config)
        # elif self.model == "Divergence":             return DivergenceAICON(self.env_config)
        # elif self.model == "GlobalVel":              return GlobalVelAICON(self.env_config)

        # elif self.model == "Divergence":             return DivergenceAICON(self.env_config)
        # elif self.model == "Estimator":              return ContingentEstimatorAICON(self.env_config)
        # elif self.model == "Visibility":             return VisibilityAICON(self.env_config)
        else:
            raise ValueError(f"Model Type {self.model} not recognized")

def run_variation(base_run_config, base_env_config, variation_id, variation, wandb_config, num_runs, queue:mp.Queue, counter):
    # TODO: think about parallelizing even single seeded runs
    # TODO: measure parallelization performance
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

class Analysis:
    def __init__(self, experiment_config: dict):
        self.base_env_config: dict = experiment_config['base_env_config']
        
        self.base_run_config: dict = experiment_config['base_run_config']
        self.base_run_config['render'] = False
        self.base_run_config['prints'] = 0
        self.base_run_config['step_by_step'] = False

        self.experiment_config = experiment_config
        self.name = experiment_config["name"]
        self.model = experiment_config["model_type"]
        self.num_runs = experiment_config["num_runs"]

        self.variations = experiment_config["variations"]
        
        self.logger = AICONLogger(self.variations)        
        self.record_dir = f"records/{datetime.now().strftime('%Y_%m_%d_%H_%M')}_{self.name}/"
        self.record_videos = experiment_config["record_videos"]

        self.wandb_config = None
        if self.experiment_config['wandb'] and self.experiment_config.get("wandb_group", None) is None:
            self.experiment_config['wandb_group'] = f"{datetime.now().strftime('%Y_%m_%d_%H_%M')}"
            self.wandb_config = {
                "wandb_project": f"aicon-{self.experiment_config['name']}",
                "wandb_group": self.experiment_config['wandb_group'],
            }

    def add_and_run_variations(self, variations: List[dict]):
        self.variations += [variation for variation in variations if not any(all(variation[sub_key]==var[sub_key] for sub_key in var) for var in self.variations)]
        self.logger.add_variations(variations)
        self.experiment_config["variations"] = self.variations
        self.run_analysis(variations)

    def run_analysis(self, variations: List[dict] = None):
        if variations is None:
            variations = self.variations
        total_configs = len(variations)
        total_runs = total_configs * self.num_runs
        mp.set_start_method('spawn', force=True)
        with tqdm(total=total_runs, desc="Running Analysis", position=0, leave=True, ) as pbar:
            counter = mp.Value('i', 0)
            data_queue = mp.Queue()
            processes = []
            active_processes = []
            for variation in variations:
                var_id = self.logger.set_variation(variation)
                p = mp.Process(target=run_variation, args=(self.base_run_config, self.base_env_config, var_id, variation, self.wandb_config, self.num_runs, data_queue, counter))
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

    def run_demo(self, variation: dict, run_number, step_by_step:bool=False, record_video=False):
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
        run_config['render'] = True
        run_config['prints'] = 1
        run_config['step_by_step'] = step_by_step
        runner = Runner(
            variation = variation,
            run_config = run_config,
            base_env_config = self.base_env_config,
        )
        runner.num_run = run_number-1
        if record_video:
            self.logger.set_variation(variation)
            config_id = self.logger.current_variation_id
            video_path = self.record_dir + f"/records/variation{config_id}_seed{runner.seed+runner.num_run-1}.mp4"
            runner.video_record_path = video_path
        runner.run()

    def plot_states(self, plotting_config: Dict[str,Tuple[List[int],List[str]]], save: bool=False, show: bool=True):
        """
        Plots the logged states according to the state dictionary.
        Args:
            plotting_config (Dict[str, Tuple[List[int], List[str]]], optional): 
                A dictionary where keys are estimator names and values are tuples containing 
                a list of state indices and a list of corresponding labels to plot.
        """
        self.logger.plot_states(plotting_config, save_path=self.record_dir if save else None, show=show)

    def plot_state_runs(self, plotting_config: Dict[str,Tuple[List[int],List[str]]], config_id: str, runs: list[int]=None, save: bool=False, show: bool=True):
        self.logger.plot_state_runs(plotting_config, config_id, runs, save_path=self.record_dir if save else None, show=show)

    def plot_goal_losses(self, plotting_config:dict, plot_subgoals:bool=False, save:bool=True, show:bool=False):
        self.logger.plot_goal_losses(plotting_config, plot_subgoals, save_path=self.record_dir if save else None, show=show)

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
    
