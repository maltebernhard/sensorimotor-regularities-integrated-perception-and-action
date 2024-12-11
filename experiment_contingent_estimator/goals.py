import torch
from components.goal import Goal
from typing import Dict

# ==================================================================================

class SpecificGoToTargetGoal(Goal):
    def __init__(self, device):
        super().__init__(device)
        self.desired_distance = 0.0

    def loss_function(self, buffer_dict: Dict[str, Dict[str, torch.Tensor]]):
        loss_mean = torch.concat([
            1.0 * torch.atleast_1d(buffer_dict['RobotState']['state_mean'][3] - self.desired_distance),
            #1e-1 * torch.atleast_1d(buffer_dict['PolarTargetPos']['state_mean'][2] / (torch.abs(buffer_dict['PolarTargetPos']['state_mean'][0] - self.desired_distance) + 1.0)),
        ]).pow(2).sum()
        loss_cov = 1e0 * torch.trace(buffer_dict['RobotState']['state_cov'])
        #loss_cov = 0.0
        return loss_mean + loss_cov
    
    