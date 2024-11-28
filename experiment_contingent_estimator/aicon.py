import numpy as np
import yaml
import torch
from typing import Dict

from components.aicon import AICON
from environment.gaze_fix_env import GazeFixEnv

# ========================================================================================================

class SimpleVelTestAICON(AICON):
    def __init__(self):
        super().__init__()

        config = 'config/env_config.yaml'
        with open(config) as file:
            env_config = yaml.load(file, Loader=yaml.FullLoader)
            env_config["action_mode"] = 3
        self.set_env(GazeFixEnv(env_config))

        estimators = {}
        #self.set_estimators(estimators)

        active_interconnections = {}
        #self.set_active_interconnections(active_interconnections)

        goals = {}
        #self.set_goals(goals)

        self.reset()

    def reset(self):
        raise NotImplementedError

    def eval_predict(self, action, buffer_dict):
        raise NotImplementedError

    def eval_step(self, action, new_step = False):
        raise NotImplementedError
    
    def render(self):
        raise NotImplementedError

    def compute_action(self, gradients):
        raise NotImplementedError
    
    def print_states(self):
        raise NotImplementedError