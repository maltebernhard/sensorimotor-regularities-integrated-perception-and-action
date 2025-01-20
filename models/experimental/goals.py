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
            1.0 * torch.atleast_1d(buffer_dict['PolarTargetGlobalPos']['state_mean'][0] - self.desired_distance),
        ]).pow(2).sum()
        loss_cov = 1e0 * torch.trace(buffer_dict['PolarTargetGlobalPos']['state_cov'])
        return loss_mean + loss_cov