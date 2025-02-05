import torch
from components.estimator import RecursiveEstimator
from components.helpers import rotate_vector_2d

# ==================================== Specific Implementations ==============================================

class Polar_Pos_Estimator(RecursiveEstimator):
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
        self.default_motion_noise = torch.eye(4) * torch.tensor([5e-1, 5e-2, 5e-1, 5e-2])
        self.update_uncertainty = torch.eye(4) * torch.tensor([1e-1, 1e-2, 1e-1, 1e-2])

    def forward_model(self, x_mean: torch.Tensor, cov: torch.Tensor, u: torch.Tensor):
        ret_mean = torch.empty_like(x_mean)
        timestep = u[0]

        old_rtf_vel = rotate_vector_2d(-x_mean[1], u[1:3])
        new_distance = x_mean[0] - old_rtf_vel[0] * timestep
        new_angle = (x_mean[1] + (- old_rtf_vel[1]/x_mean[0] - u[3]) * timestep + torch.pi) % (2 * torch.pi) - torch.pi
        new_rtf_vel = rotate_vector_2d(-new_angle, u[1:3])

        ret_mean[0] = new_distance
        ret_mean[1] = new_angle
        ret_mean[2] = - new_rtf_vel[0]
        ret_mean[3] = - new_rtf_vel[1]/new_distance# - u[3]
        
        return ret_mean, cov
