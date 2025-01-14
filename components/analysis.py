from datetime import datetime
import os
from typing import Dict, List, Tuple, Type
import torch
from tqdm import tqdm
import yaml
from components.aicon import AICON
from components.logger import AICONLogger
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

class Runner:
    def __init__(self, aicon_type: str, run_config: dict, env_config: dict, logger:AICONLogger = None):
        self.aicon_type = aicon_type
        self.env_config = env_config

        self.num_steps = run_config['num_steps']
        self.seed = run_config['seed']
        self.initial_action = run_config['initial_action']

        self.render = run_config['render']
        self.prints = run_config['prints']
        self.step_by_step = run_config['step_by_step']
        
        self.logger = logger
        self.aicon = self.select_aicon_type(self.aicon_type)(env_config)

        self.num_run = 0

    def run(self):
        self.num_run += 1
        if self.logger is not None:
            self.logger.run = self.num_run
        self.aicon.run(
            timesteps=self.num_steps,
            env_seed=self.seed+self.num_run-1,
            initial_action=torch.tensor(self.initial_action, device=self.aicon.device),
            render=self.render,
            prints=self.prints,
            step_by_step=self.step_by_step,
            logger=self.logger,
        )
    
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

class Analysis:
    def __init__(self, experiment_config: dict):
        self.base_env_config = experiment_config['base_env_config']
        self.base_run_config = experiment_config['base_run_config']
        self.base_run_config['render'] = False
        self.base_run_config['prints'] = 0
        self.base_run_config['step_by_step'] = False

        self.experiment_config = experiment_config
        self.num_runs = experiment_config["num_runs"]

        # Variations
        self.aicon_type_config: List[str] = experiment_config["aicon_type_config"]
        self.sensor_noise_config: List[dict] = experiment_config["sensor_noise_config"]
        self.moving_target_config: List[bool] = experiment_config["moving_target_config"]
        self.observation_loss_config: List[Dict[str,Tuple[float,float]]] = experiment_config["observation_loss_config"]
        
        self.logger = AICONLogger()
        self.record_dir = f"records/{datetime.now().strftime('%Y_%m_%d_%H_%M')}"

    def run_analysis(self):
        os.makedirs(os.path.join(self.record_dir, 'configs'), exist_ok=True)
        os.makedirs(os.path.join(self.record_dir, 'records'), exist_ok=True)
        total_runs = len(self.aicon_type_config) * len(self.sensor_noise_config) * len(self.moving_target_config) * len(self.observation_loss_config) * self.num_runs
        with tqdm(total=total_runs, desc="Running Analysis", position=0, leave=True) as pbar:
            for aicon_type in self.aicon_type_config:
                for sensor_noise in self.sensor_noise_config:
                    for moving_target in self.moving_target_config:
                        for observation_loss in self.observation_loss_config:
                            # TODO: maybe observation_loss tuples can't be configured in logger
                            self.logger.set_config(aicon_type, sensor_noise, moving_target, observation_loss)
                            env_config = self.base_env_config.copy()
                            env_config["observation_noise"] = sensor_noise
                            env_config["moving_target"] = moving_target
                            env_config["observation_loss"] = observation_loss

                            runner = Runner(aicon_type, self.base_run_config, env_config, self.logger)
                            for run in range(self.num_runs):
                                runner.run()
                                pbar.update(1)
            self.visualize_graph(aicon=runner.aicon, save=True, show=False)
        self.save()
    
    def save(self):
        with open(os.path.join(self.record_dir, 'configs/env_config.yaml'), 'w') as file:
            yaml.dump(self.base_env_config, file)
        with open(os.path.join(self.record_dir, 'configs/experiment_config.yaml'), 'w') as file:
            yaml.dump(self.experiment_config, file)
        self.logger.save(self.record_dir)

    def plot_states(self, plotting_config: Dict[str,Tuple[List[int],List[str]]], save: bool=False, show: bool=True):
        """
        Plots the logged states according to the state dictionary.
        Args:
            plotting_config (Dict[str, Tuple[List[int], List[str]]], optional): 
                A dictionary where keys are estimator names and values are tuples containing 
                a list of state indices and a list of corresponding labels to plot.
        """
        self.logger.plot_states(plotting_config, save_path=self.record_dir if save else None, show=show)

    def plot_goal_losses(self, save:bool=True, show:bool=False):
        self.logger.plot_goal_losses(save_path=self.record_dir if save else None, show=show)

    def visualize_graph(self, aicon:Type[AICON], save:bool=True, show:bool=False):
        aicon.visualize_graph(save_path=os.path.join(self.record_dir, 'configs') if save else None, show=show)

    @staticmethod
    def load(folder: str):
        with open(os.path.join(folder, 'configs/env_config.yaml'), 'r') as file:
            env_config = yaml.safe_load(file)
        with open(os.path.join(folder, 'configs/experiment_config.yaml'), 'r') as file:
            experiment_config = yaml.safe_load(file)
        analysis = Analysis(experiment_config, env_config)
        analysis.logger.load(folder)
        analysis.record_dir = folder
        return analysis