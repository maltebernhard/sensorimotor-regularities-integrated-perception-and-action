import torch
from components.estimator import RecursiveEstimator

# ==================================== Specific Implementations ==============================================
    
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
        self.default_state = torch.tensor([1.0], device=device)
        self.default_cov = 1e1 * torch.eye(1, device=device)
        self.default_motion_noise = 1e-2 * torch.eye(1, device=device)

    def forward_model(self, x_mean, cov: torch.Tensor, u):
        return torch.clamp(x_mean - 0.1, min=0.0), cov