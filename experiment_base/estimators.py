import torch
from components.estimator import RecursiveEstimator
from components.helpers import rotate_vector_2d

# ==================================================================================

class Polar_Pos_Estimator_Vel(RecursiveEstimator):
    """
    Estimator for Target state x:
    x[0]: target distance
    x[1]: target offset angle
    x[2]: del target distance
    x[3]: del target offset angle
    """
    def __init__(self, object_name:str="Target"):
        super().__init__(f'Polar{object_name}Pos', 4)
        self.default_state = torch.tensor([10.0, 0.0, 0.0, 0.0])
        self.default_cov = 1e3 * torch.eye(4)
        self.default_motion_noise = torch.eye(4) * torch.tensor([1e0, 1e0, 1e0, 1e0])
        #self.update_uncertainty = torch.eye(4) * torch.tensor([1e-1, 1e-1, 1e-1, 1e-1])

    def forward_model(self, x_mean: torch.Tensor, cov: torch.Tensor, u: torch.Tensor):
        timestep = u[0]
        rtf_vel = rotate_vector_2d(x_mean[1], u[1:3])

        ret_mean = torch.empty_like(x_mean)
        ret_mean[0] = x_mean[0] + (x_mean[2] - rtf_vel[0]) * timestep
        ret_mean[1] = (x_mean[1] + (x_mean[3] - rtf_vel[1]/x_mean[0] - u[3]) * timestep + torch.pi) % (2 * torch.pi) - torch.pi
        ret_mean[2] = x_mean[2]
        ret_mean[3] = x_mean[3]
        return ret_mean, cov