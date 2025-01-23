import math
import time
import torch
from components.goal import Goal
from typing import Dict

# ==================================================================================
    
class PolarGoToTargetFovealVisionGoal(Goal):
    def __init__(self):
        super().__init__()
        self.desired_distance = 0.0

    def loss_function(self, buffer_dict: Dict[str, Dict[str, torch.Tensor]]):
        currtime = time.time()
        relevance_target = (math.sin(currtime/2) + 1.0) / 2.0
        relevance_obstacle = 1.0 - relevance_target

        relevance_target = round(relevance_target)
        relevance_obstacle = round(relevance_obstacle)

        target_loss_mean = 2e0 * torch.atleast_1d(buffer_dict['PolarTargetPos']['state_mean'][0] - self.desired_distance)
        obstacle_loss_mean = 2e0 * torch.atleast_1d(buffer_dict['PolarObstacle1Pos']['state_mean'][0] - self.desired_distance)
        loss_mean = 0.0
        loss_mean += relevance_target * target_loss_mean.pow(2)
        loss_mean += relevance_obstacle * obstacle_loss_mean.pow(2)

        target_cov = buffer_dict['PolarTargetPos']['state_cov'].diag()
        obs_cov = buffer_dict['PolarObstacle1Pos']['state_cov'].diag()
        target_loss_cov = 2e4 * (target_cov[1] + target_cov[3]).sqrt() / buffer_dict['PolarTargetPos']['state_mean'][0]
        obstacle_loss_cov = 2e4 * (obs_cov[1] + obs_cov[3]).sqrt() / buffer_dict['PolarObstacle1Pos']['state_mean'][0]
        loss_cov = 0.0
        loss_cov += relevance_target * target_loss_cov
        loss_cov += relevance_obstacle * obstacle_loss_cov

        loss = 0.0
        loss += loss_mean.squeeze()
        loss += loss_cov.squeeze()

        return loss