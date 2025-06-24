import torch
from components.aicon import RecursiveEstimator
from components.helpers import rotate_vector_2d

# ==================================== Specific Implementations ==============================================

class Robot_Vel_Estimator(RecursiveEstimator):
    # TODO: turn into 3d vel instead of 2d vel + rot
    def __init__(self):
        super().__init__("RobotVel", 3)
        self.default_state = torch.tensor([0.0, 0.0, 0.0])
        self.default_cov = 1e-3 * torch.eye(3)
        self.default_motion_noise = torch.eye(3) * torch.tensor([1e-1, 1e-1, 1e-2])
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
        ret_mean = torch.empty_like(x_mean)
        ret_mean[0] = new_vel[0]
        ret_mean[1] = new_vel[1]
        ret_mean[2] = u[3]
        return ret_mean, cov
    
class Polar_Pos_Estimator(RecursiveEstimator):
    # TODO: turn 3d
    """
    Estimator for Target state x:
    x[0]: target distance
    x[1]: target offset angle
    (x[2]: target distance dot)
    (x[3]: target offset angle dot)
    x[-1]: target radius
    """
    def __init__(self, object_name:str="Target", moving_object:bool=False):
        super().__init__(f'Polar{object_name}Pos', 5 if moving_object else 3)
        self.moving_object = moving_object
        if moving_object:
            self.default_state = torch.tensor([15.0, 0.0, 0.0, 0.0, 0.1])
            self.default_cov = torch.eye(5) * torch.tensor([1e2, 1e1, 1e1, 1e1, 1e1])
            self.default_motion_noise = torch.eye(5) * torch.tensor([5e-1, 3e-2, 5e-1, 1e-1, 5e-3])
            self.update_uncertainty = torch.eye(5) * torch.tensor([1e-1, 1e-2, 1e-2, 1e-2, 1e-2])
        else:
            self.default_state = torch.tensor([20.0, 0.0, 0.1])
            self.default_cov = torch.eye(3) * torch.tensor([1e2, 1e1, 1e1])
            self.default_motion_noise = torch.eye(3) * torch.tensor([1e-1, 3e-2, 3e-2])
            self.update_uncertainty = torch.eye(3) * torch.tensor([1e-1, 1e-2, 1e-2])

    def forward_model(self, x_mean: torch.Tensor, cov: torch.Tensor, u: torch.Tensor):
        ret_mean = torch.empty_like(x_mean)
        timestep = u[0]
        old_distance = x_mean[0]
        old_angle = x_mean[1]
        old_cartesian_robot_rtf_vel = rotate_vector_2d(-x_mean[1], u[1:3])

        if not self.moving_object:
            new_distance = torch.abs(old_distance - old_cartesian_robot_rtf_vel[0] * timestep)
            new_angle = (old_angle + (- old_cartesian_robot_rtf_vel[1]/new_distance - u[3]) * timestep + torch.pi) % (2 * torch.pi) - torch.pi

            ret_mean[0] = new_distance
            ret_mean[1] = new_angle
            ret_mean[2] = x_mean[2]

        else:
            old_target_distance_dot = x_mean[2]
            old_target_angle_dot = x_mean[3]
            old_cartesian_target_rtf_vel = torch.stack([old_target_distance_dot, old_target_angle_dot*old_distance])
            new_distance = torch.abs(old_distance + (old_target_distance_dot - old_cartesian_robot_rtf_vel[0]) * timestep)
            new_angle = (old_angle + (old_target_angle_dot - old_cartesian_robot_rtf_vel[1]/new_distance - u[3]) * timestep + torch.pi) % (2 * torch.pi) - torch.pi

            angular_shift = (old_target_angle_dot - old_cartesian_robot_rtf_vel[1]/new_distance) * timestep
            new_cartesian_target_rtf_vel = rotate_vector_2d(angular_shift, old_cartesian_target_rtf_vel)
            new_target_distance_dot = new_cartesian_target_rtf_vel[0]
            new_target_angle_dot = new_cartesian_target_rtf_vel[1] / new_distance

            ret_mean[0] = new_distance
            ret_mean[1] = new_angle
            ret_mean[2] = new_target_distance_dot
            ret_mean[3] = new_target_angle_dot
            ret_mean[4] = x_mean[4]
        
        return ret_mean, cov
