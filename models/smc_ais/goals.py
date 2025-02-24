import torch
from components.goal import Goal
from typing import Dict

# ==================================================================================

class PolarGoToTargetGoal(Goal):
    def __init__(self):
        super().__init__()
        self.desired_distance = 0.0
        self.num_obstacles = 0

    def loss_function(self, buffer_dict: Dict[str, Dict[str, torch.Tensor]]):
        # penalty for distance to target
        estimated_distance = buffer_dict['PolarTargetPos']['mean'][0]
        loss_target_distance = 5e0 * (estimated_distance - self.desired_distance).pow(2)
        if estimated_distance < self.desired_distance:
            #loss_mean = (100/self.desired_distance*(estimated_distance - buffer_dict['PolarTargetPos']['mean'][-1])).pow(2)
            loss_target_distance += (1/(torch.max(torch.stack([estimated_distance - buffer_dict['PolarTargetPos']['mean'][-1], torch.tensor(1e-6)])) / self.desired_distance)-1.0).pow(2)
        cov = buffer_dict['PolarTargetPos']['cov'].diag()
        # penalty for uncertainty in distance
        loss_target_uncertainty = 1e1 * cov[0].sqrt()

        if self.num_obstacles == 0:
            return {
                "target_distance": loss_target_distance,
                "target_distance_uncertainty": loss_target_uncertainty,
            }
        
        loss_obstacle_distance =    torch.zeros(1)
        loss_obstacle_uncertainty = torch.zeros(1)
        for i in range(self.num_obstacles):
            estimated_distance = buffer_dict[f'PolarObstacle{i+1}Pos']['mean'][0]
            if estimated_distance < 10.0:
                loss_obstacle_distance += 1 / torch.max(torch.stack([
                    (estimated_distance - buffer_dict[f'PolarObstacle{i+1}Pos']['mean'][-1]) / 10.0,
                    torch.tensor(1e-6)
                ])) - 1.0 + 1e1 * (10.0 - estimated_distance).pow(2)
                uctty = buffer_dict[f'PolarObstacle{i+1}Pos']['cov'].diag()[0].sqrt()
                loss_obstacle_uncertainty += 1e1 * uctty #* (10.0 - estimated_distance)

        return {
            "target_distance":               loss_target_distance,
            "target_distance_uncertainty":   loss_target_uncertainty,
            "obstacle_distance":             loss_obstacle_distance.squeeze(),
            "obstacle_distance_uncertainty": loss_obstacle_uncertainty.squeeze(),
        }