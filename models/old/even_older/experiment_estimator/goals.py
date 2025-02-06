import torch
from components.goal import Goal
from typing import Dict

# ==================================================================================

class SpecificGoToTargetGoal(Goal):
    def __init__(self):
        super().__init__()
        self.desired_distance = 0.0

    def loss_function(self, buffer_dict: Dict[str, Dict[str, torch.Tensor]]):
        loss_mean = torch.concat([
            1.0 * torch.atleast_1d(buffer_dict['RobotState']['mean'][3] - self.desired_distance),
            #1e-1 * torch.atleast_1d(buffer_dict['PolarTargetPos']['mean'][2] / (torch.abs(buffer_dict['PolarTargetPos']['mean'][0] - self.desired_distance) + 1.0)),
        ]).pow(2).sum()
        loss_cov = 1e0 * torch.trace(buffer_dict['RobotState']['cov'])
        #loss_cov = 0.0
        return loss_mean + loss_cov
    
    