from typing import Dict
from abc import abstractmethod
import gymnasium as gym
import numpy as np

# =================================================================================

class Observation:
    def __init__(self, low, high, callable) -> None:
        self.low = low
        self.high = high
        self.calculate = callable

    def calculate_value(self):
        self.value = self.calculate()
        return self.value

# =================================================================================

class BaseEnv(gym.Env):
    def __init__(self):
        super().__init__()
        self.observations: Dict[str, Observation] = {}
    
    @abstractmethod
    def step(self, partial_action):
        raise NotImplementedError

    @abstractmethod
    def reset(self, seed=None, **kwargs):
        super().reset(seed=seed)
        np.random.seed(seed)

    @abstractmethod
    def render(self):
        raise NotImplementedError

    @abstractmethod
    def close(self):
        raise NotImplementedError
    
    # --------------------------------------------------

    @abstractmethod
    def generate_observation_space(self):
        raise NotImplementedError
    
    @abstractmethod
    def generate_action_space(self):
        raise NotImplementedError

