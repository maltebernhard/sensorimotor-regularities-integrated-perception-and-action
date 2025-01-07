import torch

from components.estimator import RecursiveEstimator
    
# ==================================================================================

# class Polar_Pos_Rad_Estimator(RecursiveEstimator):
#     """
#     Estimator for Target state x:
#     x[0]: target distance
#     x[1]: target offset angle
#     x[2]: del target distance
#     x[3]: del target offset angle
#     x[4]: target radius
#     """
#     def __init__(self, device, id: str):
#         super().__init__(id, 5, device)
#         self.default_state = torch.tensor([10.0, 0.1, 0.0, 0.0, 1.0], device=device)
#         self.default_cov = 1e3 * torch.eye(5, device=device)
#         self.default_motion_noise = torch.eye(5, device=device) * torch.tensor([5e-1, 1e-2, 1e0, 1e-1, 1e-2], device=device)

#     def forward_model(self, x_mean: torch.Tensor, cov: torch.Tensor, u: torch.Tensor):
#         timestep = u[0]
#         offset_angle = x_mean[1]
#         robot_target_frame_rotation_matrix = torch.stack([
#             torch.stack([torch.cos(-offset_angle), -torch.sin(-offset_angle)]),
#             torch.stack([torch.sin(-offset_angle), torch.cos(-offset_angle)]),
#         ]).squeeze()
#         robot_target_frame_vel = torch.matmul(robot_target_frame_rotation_matrix, u[1:3])

#         ret_mean = torch.empty_like(x_mean)
#         ret_mean[0] = x_mean[0] + (- robot_target_frame_vel[0]) * timestep
#         ret_mean[1] = (x_mean[1] + (- robot_target_frame_vel[1]/x_mean[0] - u[3]) * timestep + torch.pi) % (2 * torch.pi) - torch.pi
#         ret_mean[2] = - robot_target_frame_vel[0]
#         ret_mean[3] = - robot_target_frame_vel[1]/x_mean[0] - u[3]
#         ret_mean[4] = x_mean[4]
#         return ret_mean, cov
    
class Vis_Angle_Estimator(RecursiveEstimator):
    """
    NOTE: Since this implementation differentiates between Measurement Models based on sensor measurements, which get called only after a step,
    and Active Interconnections, which get called in gradient computation as well, we need to implement a dummy estimator for visual angle measurement to use it in Active Interconenctions.
     
    Dummy Estimator for Visual Angle Measurement:
    x[0]: visual angle
    x[1]: visual angle dot
    """
    def __init__(self, device, id):
        super().__init__(id, 2, device)
        self.default_state = torch.tensor([0.0, 0.0], device=device)
        self.default_cov = 1e3 * torch.eye(2, device=device)
        self.default_motion_noise = 1e-3 * torch.eye(2, device=device)

    def forward_model(self, x_mean: torch.Tensor, cov: torch.Tensor, u: torch.Tensor):
        return x_mean, cov

class Rad_Estimator(RecursiveEstimator):
    """
    Estimator for Object radius r:
    x[0]: object radius
    """
    def __init__(self, device, id: str):
        super().__init__(id, 1, device)
        self.default_state = torch.tensor([1.0], device=device)
        self.default_cov = 1e3 * torch.eye(1, device=device)
        self.default_motion_noise = torch.eye(1, device=device) * 1e-2

    def forward_model(self, x_mean: torch.Tensor, cov: torch.Tensor, u: torch.Tensor):
        # NOTE: This implementation assumes full knowledge of the object radius, resulting in good distance estimation
        # ret_mean = torch.empty_like(x_mean)
        # ret_cov = torch.zeros_like(cov)
        # ret_mean[0] = 1.0
        # return ret_mean, ret_cov
        return x_mean, cov