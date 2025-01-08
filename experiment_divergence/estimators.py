import torch

from components.estimator import RecursiveEstimator
    
# ==================================================================================
    
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