import torch
from components.goal import Goal
from typing import Dict

# ==================================================================================

class PolarGoToTargetGoal(Goal):
    def __init__(self):
        super().__init__()
        self.desired_distance = 0.0

    def loss_function(self, buffer_dict: Dict[str, Dict[str, torch.Tensor]]):
        loss_mean = torch.concat([
            1e0 * torch.atleast_1d(buffer_dict['PolarTargetPos']['state_mean'][0] - self.desired_distance),
        ]).pow(2).sum()
        loss_cov = 2e0 * torch.trace(buffer_dict['PolarTargetPos']['state_cov'])
        return loss_mean + loss_cov

class PolarGoToTargetGazeFixationGoal(Goal):
    def __init__(self):
        super().__init__()
        self.desired_distance = 0.0

    def loss_function(self, buffer_dict: Dict[str, Dict[str, torch.Tensor]]):
        loss_mean = torch.concat([
            # Translation
            1e0 * torch.atleast_1d(buffer_dict['PolarTargetPos']['state_mean'][0] - self.desired_distance),
            # GazeFixation
            1e1 * torch.atleast_1d(buffer_dict['PolarTargetPos']['state_mean'][1]),
            1.0 * torch.atleast_1d(buffer_dict['PolarTargetPos']['state_mean'][3] * torch.clamp(1.0 - torch.abs(buffer_dict['PolarTargetPos']['state_mean'][1]), min=0.0)),
        ]).pow(2).sum()
        loss_cov = 1e0 * torch.trace(buffer_dict['PolarTargetPos']['state_cov'])
        return loss_mean + loss_cov
    
class PolarGoToTargetFovealVisionGoal(Goal):
    def __init__(self):
        super().__init__()
        self.desired_distance = 0.0

    def loss_function(self, buffer_dict: Dict[str, Dict[str, torch.Tensor]]):
        loss_mean = torch.concat([
            1e0 * torch.atleast_1d(buffer_dict['PolarTargetPos']['state_mean'][0] - self.desired_distance),
        ]).pow(2).sum()
        cov = buffer_dict['PolarTargetPos']['state_cov'].diag()
        loss_cov = 2e0 * (cov[0] + cov[2]) + 2e0 * (cov[1] + cov[3])
        return loss_mean + loss_cov