from abc import ABC, abstractmethod
import torch

class State:
    def __init__(self, state_mean):
        self.state_mean: torch.Tensor = state_mean
        self.state_cov: torch.Tensor = torch.zeros((len(state_mean), len(state_mean)))

class RecursiveEstimator(ABC):
    def __init__(self, state_dim):
        self.state_dim = state_dim

    def set_state(self, state: State):
        self.state = state

    @abstractmethod
    def update(self, measurement, control_input):
        pass