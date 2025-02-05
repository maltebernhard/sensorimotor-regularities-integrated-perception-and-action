import torch
from components.estimator import RecursiveEstimator
from components.helpers import rotate_vector_2d

# ==================================== Specific Implementations ==============================================

class Robot_Vel_Estimator(RecursiveEstimator):
    def __init__(self):
        super().__init__("RobotVel", 3)
        self.default_state = torch.tensor([0.0, 0.0, 0.0])
        self.default_cov = 1e-3 * torch.eye(3)
        self.default_motion_noise = torch.eye(3) * torch.tensor([1e-1, 1e-1, 5e-2])
        #self.update_uncertainty: torch.Tensor = torch.eye(self.state_dim) * torch.tensor([1e-1, 1e-1, 2e-2])

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
    x[2]: target radius
    """
    def __init__(self, object_name:str="Target"):
        super().__init__(f'Polar{object_name}Pos', 3)
        self.default_state = torch.tensor([10.0, 0.0, 0.1])
        self.default_cov = torch.eye(3) * torch.tensor([1e1, 1e1, 1e1])
        self.default_motion_noise = torch.eye(3) * torch.tensor([1e-1, 3e-2, 3e-2])
        self.update_uncertainty = torch.eye(3) * torch.tensor([1e-1, 1e-2, 1e-2])

    def forward_model(self, x_mean: torch.Tensor, cov: torch.Tensor, u: torch.Tensor):
        ret_mean = torch.empty_like(x_mean)
        timestep = u[0]

        old_rtf_vel = rotate_vector_2d(-x_mean[1], u[1:3])
        #target_radius = torch.sin(x_mean[2] / 2) * x_mean[0] if x_mean[2] < 2 * torch.pi else 1.0
        new_distance = torch.abs(x_mean[0] - old_rtf_vel[0] * timestep)
        new_angle = (x_mean[1] + (- old_rtf_vel[1]/x_mean[0] - u[3]) * timestep + torch.pi) % (2 * torch.pi) - torch.pi
        #new_visual_angle = 2 * torch.asin(target_radius / new_distance) if not target_radius / new_distance >= 1.0 else 2 * torch.pi

        ret_mean[0] = new_distance
        ret_mean[1] = new_angle
        ret_mean[2] = x_mean[2]
        
        return ret_mean, cov
