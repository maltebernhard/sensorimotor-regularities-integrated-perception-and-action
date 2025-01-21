import torch
from components.estimator import RecursiveEstimator
from components.helpers import rotate_vector_2d

# ==================================== Specific Implementations ==============================================

class Polar_Pos_FovealVision_Estimator_Vel(RecursiveEstimator):
    """
    Estimator for Target state x:
    x[0]: target distance
    x[1]: target offset angle
    x[2]: del target distance
    x[3]: del target offset angle
    """
    def __init__(self, object_name:str="Target", foveal_vision_noise:dict={}, sensor_angle:float=2*torch.pi):
        super().__init__(f'Polar{object_name}Pos', 4)
        self.default_state = torch.tensor([10.0, 0.0, 0.0, 0.0])
        self.default_cov = 1e3 * torch.eye(4)
        self.default_motion_noise = torch.eye(4) * torch.tensor([1e-1, 5e-1, 1e-1, 5e-1])
        self.update_uncertainty = torch.eye(4) * torch.tensor([1e-1, 1e-2, 1e-1, 1e-2])
        self.foveal_vision_noise = foveal_vision_noise
        self.sensor_angle = sensor_angle

    @staticmethod
    def smooth_abs(x, margin=1.0):
        abs_x = torch.abs(x)
        smooth_part = 0.5 * (x**2) / margin
        linear_part = abs_x - 0.5 * margin
        return torch.where(abs_x <= margin, smooth_part, linear_part)

    def forward_model(self, x_mean: torch.Tensor, cov: torch.Tensor, u: torch.Tensor):
        timestep = u[0]
        rtf_vel = rotate_vector_2d(x_mean[1], u[1:3])

        ret_mean = torch.empty_like(x_mean)
        ret_mean[0] = x_mean[0] + (- rtf_vel[0]) * timestep
        ret_mean[1] = (x_mean[1] + (- rtf_vel[1]/x_mean[0] - u[3]) * timestep + torch.pi) % (2 * torch.pi) - torch.pi
        ret_mean[2] = - rtf_vel[0]
        ret_mean[3] = - rtf_vel[1]/x_mean[0] - u[3]
        ret_cov = cov.clone()

        if ret_mean[1].clone().abs() < self.sensor_angle/2:
            #ret_cov[1,1] = (self.foveal_vision_noise["target_offset_angle"] * self.smooth_abs(ret_mean[1]/(self.sensor_angle/2), margin=1e-3)).pow(2)
            #ret_cov[3,3] = (self.foveal_vision_noise["target_offset_angle_dot"] * self.smooth_abs(ret_mean[1]/(self.sensor_angle/2), margin=1e-3)).pow(2)
            ret_cov[1,1] = self.smooth_abs(ret_mean[1], margin=1e-3)
            ret_cov[3,3] = self.smooth_abs(ret_mean[1], margin=1e-3)
        return ret_mean, ret_cov