import torch
from components.estimator import RecursiveEstimator

# ==================================== Specific Implementations ==============================================

class Polar_Distance_Estimator(RecursiveEstimator):
    """
    Estimator for Target state x:
    x[0]: target distance
    x[1]: target distance dot
    """
    def __init__(self, device, id: str):
        super().__init__(id, 2, device)
        self.default_state = torch.tensor([10.0, 0.0], device=device)
        self.default_cov = 1e3 * torch.eye(2, device=device)
        self.default_motion_noise = torch.eye(2, device=device) * torch.tensor([5e-1, 1e0], device=device)

    def forward_model(self, x_mean: torch.Tensor, cov: torch.Tensor, u: torch.Tensor):
        timestep = u[0]
        offset_angle = x_mean[1]
        robot_target_frame_rotation_matrix = torch.stack([
            torch.stack([torch.cos(-offset_angle), -torch.sin(-offset_angle)]),
            torch.stack([torch.sin(-offset_angle), torch.cos(-offset_angle)]),
        ]).squeeze()
        robot_target_frame_vel = torch.matmul(robot_target_frame_rotation_matrix, u[1:3])

        ret_mean = torch.empty_like(x_mean)
        ret_mean[0] = x_mean[0] + (- robot_target_frame_vel[0]) * timestep
        ret_mean[1] = - robot_target_frame_vel[0]
        return ret_mean, cov
    
class Polar_Angle_Estimator(RecursiveEstimator):
    """
    Estimator for Target state x:
    x[0]: target angle
    x[1]: target angle dot
    """
    def __init__(self, device, id: str):
        super().__init__(id, 2, device)
        self.default_state = torch.tensor([0.1, 0.0], device=device)
        self.default_cov = 1e3 * torch.eye(2, device=device)
        self.default_motion_noise = torch.eye(2, device=device) * torch.tensor([1e-2, 1e-1], device=device)

    def forward_model(self, x_mean: torch.Tensor, cov: torch.Tensor, u: torch.Tensor):
        timestep = u[0]
        offset_angle = x_mean[1]
        robot_target_frame_rotation_matrix = torch.stack([
            torch.stack([torch.cos(-offset_angle), -torch.sin(-offset_angle)]),
            torch.stack([torch.sin(-offset_angle), torch.cos(-offset_angle)]),
        ]).squeeze()
        robot_target_frame_vel = torch.matmul(robot_target_frame_rotation_matrix, u[1:3])

        ret_mean = torch.empty_like(x_mean)
        ret_mean[0] = (x_mean[0] - u[3] * timestep + torch.pi) % (2 * torch.pi) - torch.pi
        ret_mean[1] = - u[3]
        return ret_mean, cov

class Obstacle_Rad_Estimator(RecursiveEstimator):
    """
    Estimator for obstacle radius state x:
    x[0]: obstacle radius
    """
    def __init__(self, device, id: str):
        super().__init__(id, 1, device)
        self.default_state = torch.tensor([1.0], device=device)
        self.default_cov = 1e1 * torch.eye(1, device=device)
        self.default_motion_noise = 1e-2 * torch.eye(1, device=device)

    def forward_model(self, x_mean, cov: torch.Tensor, u):
        return x_mean, cov
    
class Target_Visibility_Estimator(RecursiveEstimator):
    """
    Estimator for target visibility state x:
    x[0]: target visibility
    """
    def __init__(self, device, id: str):
        super().__init__(id, 1, device)
        self.default_state = torch.tensor([0.0], device=device)
        self.default_cov = 0.1 * torch.eye(1, device=device)
        self.default_motion_noise = 1e-3 * torch.eye(1, device=device)

    def forward_model(self, x_mean, cov: torch.Tensor, u):
        ret_mean = x_mean - 0.1
        if ret_mean < 0:
            ret_mean = torch.zeros(1)
        return ret_mean, cov