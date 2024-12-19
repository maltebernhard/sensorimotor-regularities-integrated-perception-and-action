import torch

from components.estimator import RecursiveEstimator
    
# ==================================================================================

class Object_Visual_Angle_Estimator(RecursiveEstimator):
    """
    Estimator for object visual angle state x:
    x[0]: object visual angle
    """
    def __init__(self, device, id: str):
        super().__init__(id, 1, device)
        self.default_state = torch.tensor([0.1], device=device)
        self.default_cov = 1e1 * torch.eye(1, device=device)
        self.default_motion_noise = 1e-2 * torch.eye(1, device=device)

    def forward_model(self, x_mean, cov: torch.Tensor, u):
        return x_mean, cov