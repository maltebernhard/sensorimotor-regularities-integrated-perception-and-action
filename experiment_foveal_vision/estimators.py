import torch
from components.estimator import RecursiveEstimator

# ==================================== Specific Implementations ==============================================
    
# class Target_Visibility_Estimator(RecursiveEstimator):
#     """
#     Estimator for target visibility state x:
#     x[0]: target visibility
#     """
#     def __init__(self, device, id: str):
#         super().__init__(id, 1, device)
#         self.default_state = torch.tensor([1.0], device=device)
#         self.default_cov = 1e1 * torch.eye(1, device=device)
#         self.default_motion_noise = 1e-2 * torch.eye(1, device=device)

#     def forward_model(self, x_mean, cov: torch.Tensor, u):
#         return torch.clamp(x_mean - 0.1, min=0.0), cov
    
class Foveal_Angle_Estimator(RecursiveEstimator):
    """
    Estimator for foveal angle state x:
    x[0]: foveal angle
    x[1]: foveal angle velocity
    """
    def __init__(self, device, id: str):
        super().__init__(id, 2, device)
        self.default_state = torch.tensor([0.0, 0.0], device=device)
        self.default_cov = 1e-2 * torch.eye(2, device=device)
        self.default_motion_noise = 1e-2 * torch.eye(2, device=device)

    def forward_model(self, x_mean, cov: torch.Tensor, u):
        ret_mean = torch.empty_like(x_mean)
        ret_mean[0] = (x_mean[0] - u[3] * u[0] + torch.pi) % (2*torch.pi) - torch.pi
        ret_mean[1] = - u[3]
        ret_cov = torch.empty_like(cov)
        ret_cov[0,0] = cov[0,0] + ret_mean[0] / torch.pi * 100
        ret_cov[1,1] = cov[1,1] + ret_mean[0] / torch.pi * 100
        return ret_mean, ret_cov