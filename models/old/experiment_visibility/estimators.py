import torch
from components.estimator import RecursiveEstimator
from components.helpers import rotate_vector_2d

# ==================================== Specific Implementations ==============================================

class Polar_Distance_Estimator(RecursiveEstimator):
    """
    Estimator for Target state x:
    x[0]: target distance
    x[1]: target distance dot
    """
    def __init__(self, object_name:str="Target"):
        super().__init__(f"Polar{object_name}Distance", 2)
        self.default_state = torch.tensor([10.0, 0.0])
        self.default_cov = 1e3 * torch.eye(2)
        self.default_motion_noise = torch.eye(2) * torch.tensor([5e-1, 1e0])

    def forward_model(self, x_mean: torch.Tensor, cov: torch.Tensor, u: torch.Tensor):
        timestep = u[0]
        rtf_vel = rotate_vector_2d(x_mean[1], u[1:3])
        ret_mean = torch.empty_like(x_mean)
        ret_mean[0] = x_mean[0] + (- rtf_vel[0]) * timestep
        ret_mean[1] = - rtf_vel[0]
        return ret_mean, cov
    
class Polar_Angle_Estimator(RecursiveEstimator):
    """
    Estimator for Target state x:
    x[0]: target angle
    x[1]: target angle dot
    """
    def __init__(self, object_name:str="Target"):
        super().__init__(f'Polar{object_name}Angle', 2)
        self.default_state = torch.tensor([0.1, 0.0])
        self.default_cov = 1e3 * torch.eye(2)
        self.default_motion_noise = torch.eye(2) * torch.tensor([1e-2, 1e-1])

    def forward_model(self, x_mean: torch.Tensor, cov: torch.Tensor, u: torch.Tensor):
        timestep = u[0]
        ret_mean = torch.empty_like(x_mean)
        ret_mean[0] = (x_mean[0] - u[3] * timestep + torch.pi) % (2 * torch.pi) - torch.pi
        ret_mean[1] = - u[3]
        return ret_mean, cov
    
class Target_Visibility_Estimator(RecursiveEstimator):
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