import torch
from components.goal import Goal
from typing import Dict

# ==================================================================================

class PolarGoToTargetGoal(Goal):
    def __init__(self, device):
        super().__init__(device)
        self.desired_distance = 0.0

    def loss_function(self, buffer_dict: Dict[str, Dict[str, torch.Tensor]]) -> float:
        polar_target_pos = buffer_dict['PolarTargetPos']['state_mean']
        robot_vel = buffer_dict['RobotVel']['state_mean']
        polar_target_pos_uncertainty = buffer_dict['PolarTargetPos']['state_cov']
        # more efficient: action decay
        trans_vel_penalty = torch.atleast_1d(torch.abs(robot_vel[:2].norm()-self.desired_distance))# / torch.clamp(polar_target_pos[0], min=0.01) if polar_target_pos[0] < 10 else torch.zeros(1)
        angular_vel_penalty = torch.atleast_1d(robot_vel[2])# / torch.clamp(0.1 * torch.abs(polar_target_pos[1]), min=0.1) if torch.abs(polar_target_pos[1]) < 0.5 else torch.zeros(1)
        
        loss_mean = torch.concat([
            1.0 * torch.atleast_1d(polar_target_pos[0] - self.desired_distance),
            1e1 * torch.atleast_1d(polar_target_pos[1]),
            #1e1 * trans_vel_penalty,
            #1e1 * angular_vel_penalty,
        ]).pow(2).sum()
        loss_cov = 1e0 * torch.trace(polar_target_pos_uncertainty).pow(2)
        #print(f"Loss Mean: {loss_mean} | Loss Cov: {loss_cov}")
        return loss_mean + loss_cov
    
class AvoidObstacleGoal(Goal):
    def __init__(self, device, i_obstacle):
        super().__init__(device)
        self.i_obstacle = i_obstacle

    def loss_function(self, buffer_dict: Dict[str, Dict[str, torch.Tensor]]):
        current_state_mean = buffer_dict[f'CartesianObstacle{self.i_obstacle}Pos']['state_mean']
        current_state_cov = buffer_dict[f'CartesianObstacle{self.i_obstacle}Pos']['state_cov']
        estimated_dist_to_obstacle = torch.maximum(current_state_mean[:2].norm() - buffer_dict[f'Obstacle{self.i_obstacle}Rad']['state_mean'], 0.0001*torch.ones(1, device=self.device))
        loss_mean = (800 / estimated_dist_to_obstacle).sum()
        loss_cov = 0.2 * torch.trace(current_state_cov[:2,:2]).sum()
        return loss_mean + loss_cov