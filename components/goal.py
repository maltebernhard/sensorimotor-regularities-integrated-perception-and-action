
import torch
from abc import ABC, abstractmethod
from torch.nn import Module

from components.estimator import RecursiveEstimator

class Goal(ABC, Module):
    def __init__(self, estimator: RecursiveEstimator, desired_state: torch.Tensor):
        super().__init__()
        self._estimator = estimator
        self.desired_state = desired_state
        # assert self._estimator.state_mean.shape == self.desired_state.shape, \
        #     "Estimator state mean and desired state must have the same dimensions"

    @property
    def current_state(self) -> torch.Tensor:
        return self._estimator.state_mean
    
    @abstractmethod
    def loss_function(self, current_state_mean, current_state_cov, buffer_dict) -> float:
        raise NotImplementedError
    
    def loss_function_from_buffer(self, buffer_dict):
        return self.loss_function(buffer_dict[self._estimator.id]['state_mean'], buffer_dict[self._estimator.id]['state_cov'], buffer_dict)

# ==================================================================================

class GoToTargetGoal(Goal):
    def __init__(self, estimator: RecursiveEstimator):
        super().__init__(estimator, torch.zeros_like(estimator.state_mean))

    def loss_function(self, current_state_mean, current_state_cov, buffer_dict):
        return ((current_state_mean[:2] - self.desired_state[:2]).pow(2) + 0.5 * torch.trace(current_state_cov[:2,:2])).sum()
    
class StopGoal(Goal):
    def __init__(self, estimator: RecursiveEstimator):
        super().__init__(estimator, torch.zeros_like(estimator.state_mean))

    def loss_function(self, current_state_mean, current_state_cov, buffer_dict):
        return ((current_state_mean[:3] - self.desired_state[:3]).pow(2) + 0.1 * torch.trace(current_state_cov[:3,:3])).sum()
    
class AvoidObstacleGoal(Goal):
    def __init__(self, estimator: RecursiveEstimator):
        super().__init__(estimator, torch.zeros_like(estimator.state_mean))

    def loss_function(self, current_state_mean: torch.Tensor, current_state_cov, buffer_dict):
        estimated_dist_to_obstacle = torch.maximum(current_state_mean[:2].norm() - buffer_dict[self._estimator.id.split('Pos')[0]+'Rad']['state_mean'], 0.0001*torch.ones(1, device=current_state_mean.device))
        loss_mean = (1000 / estimated_dist_to_obstacle).sum()
        loss_cov = 0.1 * torch.trace(current_state_cov[:2,:2]).sum()
        return loss_mean + loss_cov