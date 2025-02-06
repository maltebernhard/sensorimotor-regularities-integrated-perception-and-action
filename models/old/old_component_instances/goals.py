from typing import Dict
import torch
from components.goal import Goal

# ==================================================================================

class PolarGoToTargetGoal(Goal):
    def __init__(self):
        super().__init__()
        self.desired_distance = 0.0

    def loss_function(self, buffer_dict: Dict[str, Dict[str, torch.Tensor]]):
        loss_mean = torch.concat([
            1e0 * torch.atleast_1d(buffer_dict['PolarTargetPos']['mean'][0] - self.desired_distance),
        ]).pow(2).sum()
        loss_cov = 2e0 * torch.trace(buffer_dict['PolarTargetPos']['cov'])
        return loss_mean + loss_cov
    
class CartesianGoToTargetGoal(Goal):
    def __init__(self):
        super().__init__()

    def loss_function(self, buffer_dict: Dict[str, Dict[str, torch.Tensor]]):
        current_state_mean = buffer_dict['CartesianTargetPos']['mean']
        current_state_cov = buffer_dict['CartesianTargetPos']['cov']
        loss_mean = torch.concat([
            1.0 * current_state_mean[:2],
        ]).pow(2).sum()
        loss_cov = 1.0 * torch.trace(current_state_cov[:2,:2]).pow(2)
        return loss_mean + loss_cov
    
class GazeFixationGoal(Goal):
    def __init__(self):
        super().__init__()

    def loss_function(self, buffer_dict: Dict[str, Dict[str, torch.Tensor]]):
        current_state_mean = buffer_dict['PolarTargetPos']['mean']
        current_state_cov = buffer_dict['PolarTargetPos']['cov']
        loss_mean = torch.concat([
            torch.atleast_1d(current_state_mean[1]),
        ]).pow(2).sum()
        loss_cov = 1.0 * current_state_cov[1,1]
        return loss_mean + loss_cov