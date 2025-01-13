from datetime import datetime
import os
from typing import Dict, List, Tuple
import torch
import yaml
from experiment_base.aicon import BaseAICON
from experiment_control.aicon import ControlAICON
from experiment_divergence.aicon import DivergenceAICON
from experiment_estimator.aicon import ContingentEstimatorAICON
from experiment_foveal_vision.aicon import FovealVisionAICON
from experiment_general.aicon import GeneralTestAICON
from experiment_goal.aicon import ContingentGoalAICON
from experiment_interconnection.aicon import ContingentInterconnectionAICON
from experiment_foveal_vision.aicon import FovealVisionAICON
from experiment_visibility.aicon import VisibilityAICON

# ========================================================================================================

class Analysis:
    def __init__(self, experiment_config: dict, env_config: dict):
        self.aicon_type = experiment_config['aicon_type']
        self.num_runs = experiment_config["num_runs"]
        self.num_steps = experiment_config['num_steps']
        self.seed = experiment_config['seed']
        self.initial_action = experiment_config['initial_action']
        self.render = experiment_config['render']
        self.prints = experiment_config['prints']
        self.step_by_step = experiment_config['step_by_step']
        self.record_data = experiment_config['record_data']
        self.record_video = experiment_config['record_video']
        self.plotting_config = experiment_config['plotting_config']
        self.experiment_config = experiment_config
        self.env_config = env_config
        self.aicon = self.select_aicon_type(self.aicon_type)(env_config)
        self.record_dir = f"records/{datetime.now().strftime('%Y_%m_%d_%H_%M')}_{self.aicon_type}"

    def run(self):
        for run in range(self.num_runs):
            self.aicon.run(
                timesteps=self.num_steps,
                env_seed=self.seed + run,
                initial_action=torch.tensor(self.initial_action, device=self.aicon.device),
                render=self.render if run<self.num_runs-1 else (self.record_data and self.record_video),
                prints=self.prints,
                step_by_step=self.step_by_step,
                record_path=self.record_dir if self.record_data else None,
            )
        if self.record_data:
            self.save()

    def save(self):
        os.makedirs(os.path.join(self.record_dir, 'configs'), exist_ok=True)
        with open(os.path.join(self.record_dir, 'configs/env_config.yaml'), 'w') as file:
            yaml.dump(self.env_config, file)
        with open(os.path.join(self.record_dir, 'configs/experiment_config.yaml'), 'w') as file:
            yaml.dump(self.experiment_config, file)
        os.makedirs(os.path.join(self.record_dir, 'records'), exist_ok=True)
        self.aicon.logger.save(self.record_dir)
        self.plot_states(save=True, show=False)
        self.plot_goal_losses(save=True, show=False)
        self.visualize_graph(save=True, show=False)

    def plot_states(self, plotting_config:dict=None, save:bool=False, show:bool=True):
        """
        Plots the logged states according to the state dictionary.
        Args:
            plotting_config (Dict[str, Tuple[List[int], List[str]]], optional): 
                A dictionary where keys are estimator names and values are tuples containing 
                a list of state indices and a list of corresponding labels to plot.
        """
        config = self.plotting_config if plotting_config is None else plotting_config
        self.aicon.logger.plot_states(config, save_path=self.record_dir if save else None, show=show)

    def plot_goal_losses(self, save:bool=True, show:bool=False):
        self.aicon.logger.plot_goal_losses(save_path=self.record_dir if save else None, show=show)

    def visualize_graph(self, save:bool=True, show:bool=False):
        self.aicon.visualize_graph(save_path=os.path.join(self.record_dir, 'configs') if save else None, show=show)

    @staticmethod
    def load(folder: str):
        with open(os.path.join(folder, 'configs/env_config.yaml'), 'r') as file:
            env_config = yaml.safe_load(file)
        with open(os.path.join(folder, 'configs/experiment_config.yaml'), 'r') as file:
            experiment_config = yaml.safe_load(file)
        analysis = Analysis(experiment_config, env_config)
        analysis.aicon.logger.load(os.path.join(folder, 'records/data.yaml'))
        analysis.record_dir = folder
        return analysis

    def select_aicon_type(self, typestring:str):
        if   typestring == "Base":            return BaseAICON
        elif typestring == "GeneralTest":     return GeneralTestAICON
        elif typestring == "FovealVision":    return FovealVisionAICON
        elif typestring == "Divergence":      return DivergenceAICON
        elif typestring == "Goal":            return ContingentGoalAICON
        elif typestring == "Estimator":       return ContingentEstimatorAICON
        elif typestring == "Interconnection": return ContingentInterconnectionAICON
        elif typestring == "Control":         return ControlAICON
        elif typestring == "Visibility":      return VisibilityAICON
        else:
            raise ValueError(f"AICON Type {typestring} not recognized")