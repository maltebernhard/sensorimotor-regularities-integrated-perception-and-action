import torch
from components.estimator import RecursiveEstimator
from components.helpers import rotate_vector_2d

# ==================================== Specific Implementations ==============================================

class PolarPos_Tri_Div_Estimator(RecursiveEstimator):
    """
    Estimator for Target state x:
    x[0]: target distance
    x[1]: target offset angle
    x[2]: target distance dot
    x[3]: target offset angle dot
    x[4]: target visual angle
    x[5]: target visual angle dot
    """
    def __init__(self, object_name:str="Target"):
        super().__init__(f'Polar{object_name}Pos', 6)
        self.default_state = torch.tensor([10.0, 0.0, 0.0, 0.0, 0.0, 0.0])
        self.default_cov = 1e3 * torch.eye(6)
        self.default_motion_noise = torch.eye(6) * torch.tensor([1e-1, 1e-2, 1e-1, 1e-2, 1e-2, 1e-2])
        self.update_uncertainty = torch.eye(6) * torch.tensor([1e-1, 1e-2, 1e-1, 1e-2, 1e-2, 1e-2])

    def forward_model(self, x_mean: torch.Tensor, cov: torch.Tensor, u: torch.Tensor):
        ret_mean = torch.empty_like(x_mean)
        timestep = u[0]

        pre_step_rtf_vel = rotate_vector_2d(x_mean[1], u[1:3])
        ret_mean[0] = torch.abs(x_mean[0] + (- pre_step_rtf_vel[0]) * timestep)
        ret_mean[1] = (x_mean[1] + (- pre_step_rtf_vel[1]/x_mean[0] - u[3]) * timestep + torch.pi) % (2 * torch.pi) - torch.pi

        post_step_rtf_vel = rotate_vector_2d(ret_mean[1], u[1:3])
        ret_mean[2] = - post_step_rtf_vel[0]

        estimated_angle_dot = - post_step_rtf_vel[1]/x_mean[0] - u[3]
        ret_mean[3] = estimated_angle_dot
        
        pre_step_vis_angle_dot = -2 / x_mean[0] * torch.tan(x_mean[4]/2) * (- pre_step_rtf_vel[0])
        new_vis_angle = x_mean[4] + pre_step_vis_angle_dot * timestep
        ret_mean[4] = new_vis_angle
        post_step_vis_angle_dot = -2 / ret_mean[0] * torch.tan(ret_mean[4]/2) * (- post_step_rtf_vel[0])
        ret_mean[5] = post_step_vis_angle_dot

        return ret_mean, cov
