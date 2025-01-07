import torch
from components.goal import Goal
from typing import Dict

# ==================================================================================

class PolarGoToTargetGoal(Goal):
    def __init__(self, device):
        super().__init__(device)
        self.desired_distance = 0.0

    def loss_function(self, buffer_dict: Dict[str, Dict[str, torch.Tensor]]):
        loss_mean = torch.concat([
            # Translation
            1e0 * torch.atleast_1d(buffer_dict['PolarTargetPos']['state_mean'][0] - self.desired_distance),
            #1e0 * torch.atleast_1d(buffer_dict['PolarTargetPos']['state_mean'][2] / (torch.abs(buffer_dict['PolarTargetPos']['state_mean'][0] - self.desired_distance) + 1.0)),
            # GazeFixation
            1e1 * torch.atleast_1d(buffer_dict['PolarTargetPos']['state_mean'][1]),
            1.0 * torch.atleast_1d(buffer_dict['PolarTargetPos']['state_mean'][3] * torch.clamp(1.0 - torch.abs(buffer_dict['PolarTargetPos']['state_mean'][1]), min=0.0)),
        ]).pow(2).sum()
        loss_cov = 1e0 * torch.trace(buffer_dict['PolarTargetPos']['state_cov'])
        return loss_mean + loss_cov
        # NOTE: if only mean or only cov is used, the gradient induces strong rotation when the target is not in sight
        # return loss_mean
        # return loss_cov