import torch
from components.estimator import RecursiveEstimator
from components.helpers import rotate_vector_2d

# ==================================== Specific Implementations ==============================================

class Robot_Vel_Estimator(RecursiveEstimator):
    def __init__(self):
        super().__init__("RobotVel", 3)
        self.default_state = torch.tensor([0.0, 0.0, 0.0])
        self.default_cov = 1e1 * torch.eye(3)
        self.default_motion_noise = torch.eye(3) * torch.tensor([1e-1, 1e-1, 5e-2])
        self.update_uncertainty: torch.Tensor = torch.eye(self.state_dim) * torch.tensor([1e-1, 1e-1, 2e-2])

    def forward_model(self, x_mean: torch.Tensor, cov: torch.Tensor, u: torch.Tensor):
        """
        Args:
            x_mean: Mean of the state
                x[0]: frontal vel       v_x
                x[1]: lateral vel       v_y
                x[2]: rotational vel    w
                
            u: Control input
                u[0]: del_t
                u[1]: frontal vel       v_x
                u[2]: lateral vel       v_y
                u[3]: rotational vel    w
            """
        rotation = - u[3] * u[0]
        rotation_matrix = torch.stack([
            torch.stack([torch.cos(rotation), -torch.sin(rotation)]),
            torch.stack([torch.sin(rotation), torch.cos(rotation)]),
        ]).squeeze()
        new_vel = torch.matmul(rotation_matrix, u[1:3])

        return torch.stack([
            new_vel[0],
            new_vel[1],
            u[3],
        ]).squeeze(), cov

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
        # NOTE: Unfortunately, Triangulation behavior only works well when angle uncertainty goes up faster than distance uncertainty, otherwise strong oscillation
        #       Can't tackle this with update uncertainty as it is not used when updating the estimator itself w/ Triangulation AI
        self.default_motion_noise = torch.eye(4) * torch.tensor([1e-1, 5e-1, 1e-1, 5e-1])
        self.update_uncertainty = torch.eye(4) * torch.tensor([1e-1, 1e-2, 1e-1, 1e-2])

    def forward_model(self, x_mean: torch.Tensor, cov: torch.Tensor, u: torch.Tensor):
        timestep = u[0]
        rtf_vel = rotate_vector_2d(-x_mean[1], u[1:3])

        ret_mean = torch.empty_like(x_mean)
        ret_mean[0] = x_mean[0] + (- rtf_vel[0]) * timestep
        ret_mean[1] = (x_mean[1] + (- rtf_vel[1]/x_mean[0] - u[3]) * timestep + torch.pi) % (2 * torch.pi) - torch.pi
        ret_mean[2] = - rtf_vel[0]
        ret_mean[3] = - rtf_vel[1]/x_mean[0] - u[3]
        return ret_mean, cov

class Polar_Pos_FovealVision_Estimator(RecursiveEstimator):
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
        rtf_vel = rotate_vector_2d(-x_mean[1], u[1:3])

        ret_mean = torch.empty_like(x_mean)
        ret_mean[0] = x_mean[0] + (- rtf_vel[0]) * timestep
        ret_mean[1] = (x_mean[1] + (- rtf_vel[1]/x_mean[0] - u[3]) * timestep + torch.pi) % (2 * torch.pi) - torch.pi
        ret_mean[2] = - rtf_vel[0]
        ret_mean[3] = - rtf_vel[1]/x_mean[0] - u[3]
        ret_cov = cov.clone()

        if ret_mean[1].clone().abs() < self.sensor_angle/2:
            # TODO: Using the "real" noise will make the system unstable
            #ret_cov[1,1] = (ret_mean[1] * self.foveal_vision_noise["target_offset_angle"]/(self.sensor_angle/2)).pow(2)
            #ret_cov[3,3] = (ret_mean[1] * self.foveal_vision_noise["target_offset_angle_dot"]/(self.sensor_angle/2)).pow(2)
            ret_cov[1,1] = self.smooth_abs(ret_mean[1], margin=1e-3)
            ret_cov[3,3] = self.smooth_abs(ret_mean[1], margin=1e-3)
        return ret_mean, ret_cov