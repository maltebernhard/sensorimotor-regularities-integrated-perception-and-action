import math
import torch
from components.goal import Goal
from typing import Dict

# ==================================================================================
    
class PolarGoToTargetFovealVisionGoal(Goal):
    def __init__(self):
        super().__init__()
        self.desired_distance = 0.0
        self.step = 0

    def loss_function(self, buffer_dict: Dict[str, Dict[str, torch.Tensor]]):
        self.step += 1
        currtime = self.step * 0.05

        relevance_target = (math.sin(currtime/2) + 1.0) / 2.0
        relevance_obstacle = 1.0 - relevance_target
        relevance_target = round(relevance_target)
        relevance_obstacle = round(relevance_obstacle)

        target_loss_mean =   2e0 * (buffer_dict['PolarTargetPos']['state_mean'][0] - self.desired_distance)
        obstacle_loss_mean = 2e0 * (buffer_dict['PolarObstacle1Pos']['state_mean'][0] - self.desired_distance)

        loss_mean_1 = relevance_target * target_loss_mean.pow(2)
        loss_mean_2 = relevance_obstacle * obstacle_loss_mean.pow(2)

        tar_cov = buffer_dict['PolarTargetPos']['state_cov'].diag()
        obs_cov = buffer_dict['PolarObstacle1Pos']['state_cov'].diag()
        tar_loss_cov = 2e4 * (tar_cov[1] + tar_cov[3]).sqrt() / buffer_dict['PolarTargetPos']['state_mean'][0]
        obs_loss_cov = 2e4 * (obs_cov[1] + obs_cov[3]).sqrt() / buffer_dict['PolarObstacle1Pos']['state_mean'][0]

        loss_cov_1 = relevance_target * tar_loss_cov
        loss_cov_2 = relevance_obstacle * obs_loss_cov

        return {
            "target_1_distance":    loss_mean_1,
            "target_2_distance":    loss_mean_2,
            "target_1_uncertainty": loss_cov_1,
            "target_2_uncertainty": loss_cov_2,
        }