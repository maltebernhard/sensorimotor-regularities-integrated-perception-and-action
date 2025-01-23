from datetime import datetime
import os
import sys
from typing import Dict, List, Tuple, Type
import torch
from tqdm import tqdm
import yaml
from components.aicon import AICON
from components.logger import AICONLogger
from models.experiment1.aicon import Experiment1AICON
from models.experimental.aicon import ExperimantalAICON
from models.experiment_foveal_vision.aicon import ExperimentFovealVisionAICON

from models.old.experiment_divergence.aicon import DivergenceAICON
from models.old.experiment_estimator.aicon import ContingentEstimatorAICON
from models.old.experiment_visibility.aicon import VisibilityAICON

# ========================================================================================================

class Runner:
    def __init__(self, model: str, run_config: dict, env_config: dict, aicon_type: str = None, logger: AICONLogger = None):
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

        self.num_run = 0

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
        if   self.model == "Experiment1":     return Experiment1AICON(self.env_config, self.aicon_type)
        elif self.model == "Experimental":    return ExperimantalAICON(self.env_config)
        elif self.model == "ExperimentFovealVision": return ExperimentFovealVisionAICON(self.env_config)

        elif self.model == "Divergence":      return DivergenceAICON(self.env_config)
        elif self.model == "Estimator":       return ContingentEstimatorAICON(self.env_config)
        elif self.model == "Visibility":      return VisibilityAICON(self.env_config)
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
        self.model = experiment_config["model_type"]
        self.num_runs = experiment_config["num_runs"]

        # Variations
        self.aicon_type_config: List[str] = experiment_config["aicon_type_config"]
        self.sensor_noise_config: List[dict] = experiment_config["sensor_noise_config"]
        self.moving_target_config: List[bool] = experiment_config["moving_target_config"]
        self.observation_loss_config: List[Dict[str,Tuple[float,float]]] = experiment_config["observation_loss_config"]
        self.foveal_vision_noise_config: List[dict] = experiment_config["foveal_vision_noise_config"]
        
        self.logger = AICONLogger()
        self.record_dir = f"records/{datetime.now().strftime('%Y_%m_%d_%H_%M')}/"
        self.record_videos = experiment_config["record_videos"]

    def run_analysis(self):
        os.makedirs(os.path.join(self.record_dir, 'configs'), exist_ok=True)
        os.makedirs(os.path.join(self.record_dir, 'records'), exist_ok=True)
        total_runs = len(self.aicon_type_config) * len(self.sensor_noise_config) * len(self.moving_target_config) * len(self.observation_loss_config) * self.num_runs
        with tqdm(total=total_runs, desc="Running Analysis", position=0, leave=True) as pbar, tqdm.external_write_mode(file=sys.stdout):
            for aicon_type in self.aicon_type_config:
                for sensor_noise in self.sensor_noise_config:
                    for moving_target in self.moving_target_config:
                        for observation_loss in self.observation_loss_config:
                            for foveal_vision_noise in self.foveal_vision_noise_config:
                                self.logger.set_config(aicon_type, sensor_noise, moving_target, observation_loss, foveal_vision_noise)
                                env_config = self.base_env_config.copy()
                                env_config["observation_noise"] = sensor_noise
                                env_config["moving_target"] = moving_target
                                env_config["observation_loss"] = observation_loss
                                env_config["foveal_vision_noise"] = foveal_vision_noise
                                config_id = self.logger.config
                                runner = Runner(
                                    model = self.model,
                                    aicon_type = aicon_type,
                                    run_config = self.base_run_config,
                                    env_config = env_config,
                                    logger = self.logger
                                )
                                for run in range(self.num_runs):
                                    if self.record_videos and run == self.num_runs-1:
                                        runner.render = True
                                        video_path = self.record_dir + f"/records/{config_id[0]}_{config_id[1][0]}_{config_id[1][1]}_{config_id[1][2]}_{config_id[1][3]}.mp4"
                                        runner.video_record_path = video_path
                                    runner.run()
                                    pbar.update(1)
        self.visualize_graph(aicon=runner.aicon, save=True, show=False)
        self.save()
    
    def save(self):
        with open(os.path.join(self.record_dir, 'configs/experiment_config.yaml'), 'w') as file:
            yaml.dump(self.experiment_config, file)
        self.logger.save(self.record_dir)

    def run_demo(self, aicon_type: str, sensor_noise: dict, moving_target: bool, observation_loss: dict, foveal_vision_noise: dict, run_number, record_video=False):
        env_config = self.base_env_config.copy()
        env_config["observation_noise"] = sensor_noise
        env_config["moving_target"] = moving_target
        env_config["observation_loss"] = observation_loss
        env_config["foveal_vision_noise"] = foveal_vision_noise
        run_config = self.base_run_config.copy()
        run_config['render'] = True
        run_config['prints'] = 1
        run_config['step_by_step'] = False
        runner = Runner(
            model = self.model,
            aicon_type = aicon_type,
            run_config = run_config,
            env_config = env_config
        )
        runner.num_run = run_number-1
        if record_video:
            self.logger.set_config(aicon_type, sensor_noise, moving_target, observation_loss, foveal_vision_noise)
            config_id = self.logger.config
            video_path = self.record_dir + f"/records/{config_id[0]}_{config_id[1][0]}_{config_id[1][1]}_{config_id[1][2]}_{config_id[1][3]}.mp4"
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

    def plot_state_runs(self, plotting_config: Dict[str,Tuple[List[int],List[str]]], config_id: str, save: bool=False, show: bool=True):
        self.logger.plot_state_runs(plotting_config, config_id, save_path=self.record_dir if save else None, show=show)

    def plot_goal_losses(self, plotting_config:dict, save:bool=True, show:bool=False):
        self.logger.plot_goal_losses(plotting_config, save_path=self.record_dir if save else None, show=show)

    def visualize_graph(self, aicon:Type[AICON], save:bool=True, show:bool=False):
        aicon.visualize_graph(save_path=os.path.join(self.record_dir, 'configs') if save else None, show=show)

    @staticmethod
    def load(folder: str):
        with open(os.path.join(folder, 'configs/experiment_config.yaml'), 'r') as file:
            experiment_config = yaml.safe_load(file)
        analysis = Analysis(experiment_config)
        analysis.logger.load(folder)
        analysis.record_dir = folder
        return analysis