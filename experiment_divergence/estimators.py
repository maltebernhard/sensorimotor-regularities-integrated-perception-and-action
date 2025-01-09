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
    def __init__(self, object_name:str="Target"):
        super().__init__(f'{object_name}VisAngle', 2)
        self.default_state = torch.tensor([0.0, 0.0])
        self.default_cov = 1e3 * torch.eye(2)
        self.default_motion_noise = 1e-3 * torch.eye(2)

    def forward_model(self, x_mean: torch.Tensor, cov: torch.Tensor, u: torch.Tensor):
        return x_mean, cov