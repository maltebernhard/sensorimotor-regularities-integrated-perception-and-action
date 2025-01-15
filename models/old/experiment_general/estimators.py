import torch
from components.estimator import RecursiveEstimator

# ==================================== Specific Implementations ==============================================
    
class Visibility_Estimator(RecursiveEstimator):
    """
    Estimator for target visibility state x:
    x[0]: target visibility
    """
    def __init__(self, object_name:str="Target"):
        super().__init__(f'{object_name}Visibility', 1)
        self.default_state = torch.tensor([0.0])
        self.default_cov = 0.1 * torch.eye(1)
        self.default_motion_noise = 1e-3 * torch.eye(1)

    def forward_model(self, x_mean, cov: torch.Tensor, u):
        ret_mean = x_mean - 0.1
        if ret_mean < 0:
            ret_mean = torch.zeros(1)
        return ret_mean, cov