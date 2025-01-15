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
            torch.abs(5e0 * torch.atleast_1d(buffer_dict['PolarTargetPos']['state_mean'][0] - self.desired_distance)),
        ]).sum()
        loss_cov = 1e0 * torch.trace(buffer_dict['PolarTargetPos']['state_cov'])
        return loss_mean + loss_cov