from datetime import datetime
import os
import sys
import time
from typing import Dict, List, Tuple, Type
import torch
from tqdm import tqdm
import yaml
import matplotlib.pyplot as plt
import networkx as nx

from components.aicon import AICON
from components.logger import AICONLogger, VariationLogger
from models.old.base.aicon import BaseAICON
from models.old.experiment1.aicon import Experiment1AICON
from models.old.experiment_foveal_vision.aicon import ExperimentFovealVisionAICON
from models.old.divergence.aicon import DivergenceAICON
from models.old.global_vels.aicon import GlobalVelAICON
from models.old.single_estimator.aicon import SingleEstimatorAICON
from models.smc_ais.aicon import SMCAICON

from models.old.even_older.experiment_estimator.aicon import ContingentEstimatorAICON
from models.old.even_older.experiment_visibility.aicon import VisibilityAICON

# ========================================================================================================

class Runner:
    def __init__(self, model: str, run_config: dict, env_config: dict, aicon_type: str = None, logger: VariationLogger = None):
        self.model = model
        if aicon_type is not None:
            self.aicon_type = aicon_type
        self.env_config = env_config

        self.num_steps = run_config['num_steps']
        self.seed = run_config['seed']
        self.initial_action = run_config['initial_action']

        self.render = run_config['render']
        self.video_record_path = None
        self.prints = run_config['prints']
        self.step_by_step = run_config['step_by_step']
        
        self.logger = logger
        self.aicon = self.create_model()

        self.num_run = logger.run if logger is not None else 0

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

        elif self.model == "Base":                   return BaseAICON(self.env_config)
        elif self.model == "Experiment1":            return Experiment1AICON(self.env_config, self.aicon_type)
        elif self.model == "ExperimentFovealVision": return ExperimentFovealVisionAICON(self.env_config)
        elif self.model == "SingleEstimator":        return SingleEstimatorAICON(self.env_config)
        elif self.model == "Divergence":             return DivergenceAICON(self.env_config)
        elif self.model == "GlobalVel":              return GlobalVelAICON(self.env_config)

        elif self.model == "Divergence":             return DivergenceAICON(self.env_config)
        elif self.model == "Estimator":              return ContingentEstimatorAICON(self.env_config)
        elif self.model == "Visibility":             return VisibilityAICON(self.env_config)
        else:
            raise ValueError(f"Model Type {self.model} not recognized")

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

        if self.experiment_config['wandb'] and self.experiment_config.get("wandb_group", None) is None:
            self.experiment_config['wandb_group'] = f"{datetime.now().strftime('%Y_%m_%d_%H_%M')}"
        self.logger.set_wandb_config(f"aicon-{self.experiment_config['name']}", self.experiment_config['wandb_group'])

    def add_and_run_variations(self, variations: List[dict]):
        self.variations += variations
        self.logger.add_variations(variations)
        self.logger.set_wandb_config(f"aicon-{self.experiment_config['name']}", self.experiment_config['wandb_group'])
        self.experiment_config["variations"] = self.variations
        self.run_analysis(variations)

    def run_analysis(self, variations: List[dict] = None):
        if variations is None:
            variations = self.variations
        total_configs = len(variations)
        total_runs = total_configs * self.num_runs
        completed_configs = 0

        
        with tqdm(total=total_runs, desc="Running Analysis", position=0, leave=True) as pbar, tqdm.external_write_mode(file=sys.stdout):
            for variation in variations:
                self.logger.set_variation(variation)
                env_config = self.base_env_config.copy()
                env_config["observation_noise"] = variation["sensor_noise"]
                env_config["moving_target"] = variation["moving_target"]
                env_config["observation_loss"] = variation["observation_loss"]
                env_config["fv_noise"] = variation["fv_noise"]
                config_id = self.logger.current_variation_id
                aicon_type = {
                    "smcs": variation["smcs"],
                    "control": variation["control"],
                    "distance_sensor": variation["distance_sensor"],
                }
                runner = Runner(
                    model = self.model,
                    aicon_type = aicon_type,
                    run_config = self.base_run_config,
                    env_config = env_config,
                    logger = self.logger.variation_loggers[config_id],
                )
                for run in range(self.num_runs):
                    if self.record_videos and run == self.num_runs-1:
                        runner.render = True
                        video_path = self.record_dir + f"/records/variation{config_id}_seed{runner.seed+runner.num_run-1}.mp4"
                        runner.video_record_path = video_path
                    runner.run()
                    pbar.update(1)
                    pbar.set_description(f"{completed_configs+1}:{runner.num_run}/{total_configs}:{self.num_runs}")
                completed_configs += 1

        os.makedirs(os.path.join(self.record_dir, 'configs'), exist_ok=True)
        os.makedirs(os.path.join(self.record_dir, 'records'), exist_ok=True)
        #self.visualize_graph(aicon=runner.aicon, save=True, show=False)
        self.save()
    
    def save(self):
        #print("Saving Data ...")
        with open(os.path.join(self.record_dir, 'configs/experiment_config.yaml'), 'w') as file:
            yaml.dump(self.experiment_config, file)
        self.logger.save(self.record_dir)
        #print("Data Saved.")

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
        env_config = self.base_env_config.copy()
        env_config["observation_noise"] = variation["sensor_noise"]
        env_config["moving_target"] = variation["moving_target"]
        env_config["observation_loss"] = variation["observation_loss"]
        env_config["fv_noise"] = variation["fv_noise"]
        run_config = self.base_run_config.copy()
        run_config['render'] = True
        run_config['prints'] = 1
        run_config['step_by_step'] = step_by_step
        aicon_type = {
            "smcs": variation["smcs"],
            "control": variation["control"],
            "distance_sensor": variation["distance_sensor"],
        }
        runner = Runner(
            model = self.model,
            aicon_type = aicon_type,
            run_config = run_config,
            env_config = env_config
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

    @staticmethod
    def load(folder: str):
        with open(os.path.join(folder, 'configs/experiment_config.yaml'), 'r') as file:
            experiment_config = yaml.load(file, Loader=yaml.FullLoader)
        analysis = Analysis(experiment_config)
        analysis.logger.load(folder)
        analysis.record_dir = folder
        return analysis