
from typing import Dict
import torch
from abc import ABC, abstractmethod
from torch.nn import Module

from components.estimator import RecursiveEstimator

class Goal(ABC, Module):
    def __init__(self, device):
        super().__init__()
        self.device = device
    
    @abstractmethod
    def loss_function(self, buffer_dict) -> float:
        raise NotImplementedError
    
    def loss_function_from_buffer(self, buffer_dict):
        return self.loss_function(buffer_dict)

# ==================================================================================

class PolarGoToTargetGoal(Goal):
    def __init__(self, device):
        super().__init__(device)

    def loss_function(self, buffer_dict: Dict[str, Dict[str, torch.Tensor]]) -> float:
        current_state_mean = buffer_dict['PolarTargetPos']['state_mean']
        current_state_cov = buffer_dict['PolarTargetPos']['state_cov']
        loss_mean = torch.concat([
            1.0 * torch.atleast_1d(current_state_mean[0]),
            1.0 * torch.atleast_1d(current_state_mean[1]),
        ]).pow(2).sum()
        loss_cov = 2.0 * current_state_cov[1,1]
        #loss_cov += 0.0 * current_state_cov[3,3]
        return loss_mean + loss_cov

class GoToTargetGoal(Goal):
    def __init__(self, device):
        super().__init__(device)

    def loss_function(self, buffer_dict: Dict[str, Dict[str, torch.Tensor]]):
        current_state_mean = buffer_dict['TargetPos']['state_mean']
        current_state_cov = buffer_dict['TargetPos']['state_cov']
        loss_mean = torch.concat([
            3.0 * current_state_mean[:2],
            #torch.atleast_1d(1.0 * torch.abs(torch.atan2(current_state_mean[1], current_state_mean[0])))
        ]).pow(2).sum()
        loss_cov = 10.0 * torch.trace(current_state_cov[:2,:2]).pow(2)
        return loss_mean + loss_cov
    
class GazeFixationGoal(Goal):
    def __init__(self, device):
        super().__init__(device)

    def loss_function(self, buffer_dict: Dict[str, Dict[str, torch.Tensor]]):
        current_state_mean = buffer_dict['PolarTargetPos']['state_mean']
        current_state_cov = buffer_dict['PolarTargetPos']['state_cov']
        loss_mean = torch.concat([
            torch.atleast_1d(current_state_mean[1]),
            #torch.atleast_1d(current_state_mean[3])
        ]).pow(2).sum()
        loss_cov = 0.5 * current_state_cov[1,1]
        #loss_cov += 0.5 * current_state_cov[3,3]
        return loss_mean + loss_cov
    
class StopGoal(Goal):
    def __init__(self, device):
        super().__init__(device)

    def loss_function(self, buffer_dict: Dict[str, Dict[str, torch.Tensor]]):
        current_state_mean = buffer_dict['RobotVel']['state_mean']
        current_state_cov = buffer_dict['RobotVel']['state_cov']
        loss_mean = (current_state_mean[:3]).pow(2).sum()
        loss_cov = 0.1 * torch.trace(current_state_cov[:3,:3])
        return loss_mean + loss_cov
    
class AvoidObstacleGoal(Goal):
    def __init__(self, device, i_obstacle):
        super().__init__(device)
        self.i_obstacle = i_obstacle

    def loss_function(self, buffer_dict: Dict[str, Dict[str, torch.Tensor]]):
        current_state_mean = buffer_dict[f'Obstacle{self.i_obstacle}Pos']['state_mean']
        current_state_cov = buffer_dict[f'Obstacle{self.i_obstacle}Pos']['state_cov']
        estimated_dist_to_obstacle = torch.maximum(current_state_mean[:2].norm() - buffer_dict[f'Obstacle{self.i_obstacle}Rad']['state_mean'], 0.0001*torch.ones(1, device=self.device))
        loss_mean = (800 / estimated_dist_to_obstacle).sum()
        loss_cov = 0.2 * torch.trace(current_state_cov[:2,:2]).sum()
        return loss_mean + loss_cov