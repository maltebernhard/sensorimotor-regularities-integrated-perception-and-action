import torch
from components.goal import Goal
from typing import Dict

# ==================================================================================

class PolarGoToTargetGoal(Goal):
    def __init__(self):
        super().__init__()
        self.desired_distance = 0.0

    def loss_function(self, buffer_dict: Dict[str, Dict[str, torch.Tensor]]):
        # penalty for distance to target
        estimated_distance = buffer_dict['PolarTargetPos']['state_mean'][0]
        loss_mean = torch.concat([
            2e0 * torch.atleast_1d(estimated_distance - self.desired_distance),
        ]).pow(2).sum()
        cov = buffer_dict['PolarTargetPos']['state_cov'].diag()
        # penalty for uncertainty in distance
        loss_cov_distance1 = 1e0 * cov[0]# / max(1.0, (estimated_distance/100)) 
        return {
            "distance                ": loss_mean,
            "distance_uncertainty    ": loss_cov_distance1,
        }