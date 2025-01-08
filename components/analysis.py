import torch
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
    def __init__(self, type, experiment_config: dict, env_config: dict):
        self.type = type
        self.num_runs = experiment_config["num_runs"]
        self.num_steps = experiment_config['num_steps']
        self.seed = experiment_config['seed']
        self.initial_action = experiment_config['initial_action']
        self.render = experiment_config['render']
        self.prints = experiment_config['prints']
        self.step_by_step = experiment_config['step_by_step']
        self.record_data = experiment_config['record_data']

        if   type == "GeneralTest":     self.aicon = GeneralTestAICON(env_config)
        elif type == "FovealVision":    self.aicon = FovealVisionAICON(env_config)
        elif type == "Divergence":      self.aicon = DivergenceAICON(env_config)
        elif type == "Goal":            self.aicon = ContingentGoalAICON(env_config)
        elif type == "Estimator":       self.aicon = ContingentEstimatorAICON(env_config)
        elif type == "Interconnection": self.aicon = ContingentInterconnectionAICON(env_config)
        elif type == "Control":         self.aicon = ControlAICON(env_config)
        elif type == "Visibility":      self.aicon = VisibilityAICON(env_config)
        else:
            raise ValueError(f"AICON Type {type} not recognized")

    def run(self):
        for run in range(self.num_runs):
            self.aicon.run(
                timesteps=self.num_steps,
                env_seed=self.seed + run,
                initial_action=torch.tensor(self.initial_action, device=self.aicon.device),
                render=self.render,
                prints=self.prints,
                step_by_step=self.step_by_step,
                record_data=self.record_data,
            )
        if self.record_data:
            self.plot()
            self.aicon.save()

    # TODO: implement general version
    def plot(self):
        self.aicon.logger.plot_estimation_error("PolarTargetPos", {"distance": 0, "offset_angle": 1}, save_path=self.aicon.record_dir)
        self.aicon.logger.plot_state("PolarTargetPos", save_path=self.aicon.record_dir)
        self.aicon.logger.plot_estimation_error("RobotVel", save_path=self.aicon.record_dir)
        self.aicon.logger.plot_state("RobotVel", save_path=self.aicon.record_dir)