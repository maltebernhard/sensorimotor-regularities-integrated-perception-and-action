import torch
from components.goal import Goal
from typing import Dict

# ==================================================================================

class GoToTargetGoal(Goal):
    def __init__(self, device):
        super().__init__(device)

    def loss_function(self, buffer_dict: Dict[str, Dict[str, torch.Tensor]]):
        loss_mean = buffer_dict['RobotState']['state_mean'][6].pow(2)
        loss_cov = 0.0
        return loss_mean + loss_cov