import torch
from components.aicon import RecursiveEstimator
from components.helpers import rotate_vector_2d, world_to_rtf

# ==================================== Specific Implementations ==============================================

class Robot_Vel_Estimator(RecursiveEstimator):
    def __init__(self):
        super().__init__("RobotVel", 3)
        self.default_state = torch.tensor([0.0, 0.0, 0.0])
        self.default_cov = 1e-3 * torch.eye(3)
        self.default_motion_noise = torch.eye(3) * torch.tensor([1e-2, 1e-2, 1e-2])
        #self.update_uncertainty: torch.Tensor = torch.eye(self.state_dim) * torch.tensor([1e-1, 1e-1, 2e-2])

    def forward_model(self, x_mean: torch.Tensor, cov: torch.Tensor, u: torch.Tensor):
        """
        Args:
            x_mean: Mean of the state
                x[0]: vel_x  v_x
                x[1]: vel_y  v_y
                x[2]: vel_z  v_z
                
            u: Control input
                u[0]: del_t
                u[1]: vel_x  v_x
                u[2]: vel_y  v_y
                u[3]: vel_z  v_z
            """
        ret_mean = torch.empty_like(x_mean)
        ret_mean[0] = u[1]
        ret_mean[1] = u[2]
        ret_mean[2] = u[3]
        return ret_mean, cov
    
class Polar_Pos_Estimator(RecursiveEstimator):
    # TODO: turn 3d
    """
    Estimator for Target state x:
    x[0]: target distance
    x[1]: target azimuth angle phi
    x[2]: target zenith angle theta
    (x[3]: target distance dot)
    (x[4]: target azimuth dot)
    (x[5]: target zenith dot)
    x[-1]: target radius
    """
    def __init__(self, object_name:str="Target", moving_object:bool=False):
        super().__init__(f'Polar{object_name}Pos', 7 if moving_object else 4)
        self.moving_object = moving_object
        if moving_object:
            self.default_state = torch.tensor([15.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.1])
            self.default_cov = torch.eye(7) * torch.tensor([1e2, 1e1, 1e1, 1e1, 1e1, 1e1, 1e1])
            self.default_motion_noise = torch.eye(7) * torch.tensor([5e-1, 1e-2, 1e-2, 5e-1, 1e-1, 1e-1, 5e-3])
            self.update_uncertainty = torch.eye(7) * torch.tensor([1e-1, 1e-2, 1e-2, 1e-2, 1e-2, 1e-2, 1e-2])
        else:
            self.default_state = torch.tensor([20.0, 0.0, 0.0, 0.1])
            self.default_cov = torch.eye(4) * torch.tensor([1e2, 1e1, 1e1, 1e1])
            self.default_motion_noise = torch.eye(4) * torch.tensor([1e-1, 1e-2, 1e-2, 5e-3])
            self.update_uncertainty = torch.eye(4) * torch.tensor([1e-1, 1e-2, 1e-2, 1e-2])

    def forward_model(self, x_mean: torch.Tensor, cov: torch.Tensor, u: torch.Tensor):
        ret_mean = torch.empty_like(x_mean)
        timestep = u[0]
        old_distance = x_mean[0]
        old_phi = x_mean[1]
        old_theta = x_mean[2]
        cartesian_robot_rtf_vel = world_to_rtf(u[1:4], old_phi, old_theta)

        if not self.moving_object:
            new_distance = torch.abs(old_distance - cartesian_robot_rtf_vel[0] * timestep)
            new_phi = (old_phi + (cartesian_robot_rtf_vel[1] / old_distance) * timestep + torch.pi) % (2 * torch.pi) - torch.pi
            new_theta = (old_theta + (cartesian_robot_rtf_vel[2] / old_distance) * timestep + torch.pi) % (2 * torch.pi) - torch.pi
            
            ret_mean[0] = new_distance
            ret_mean[1] = new_phi
            ret_mean[2] = new_theta
            return ret_mean, cov

        else:
            old_target_distance_dot = x_mean[2]
            old_target_phi_dot = x_mean[3]
            old_target_theta_dot = x_mean[4]
            old_cartesian_target_rtf_vel = torch.stack([old_target_distance_dot, old_target_phi_dot * old_distance, old_target_theta_dot * old_distance])
            new_distance = torch.abs(old_distance + (old_target_distance_dot - cartesian_robot_rtf_vel[0]) * timestep)
            new_phi = (old_phi + (old_target_phi_dot - cartesian_robot_rtf_vel[1] / new_distance) * timestep + torch.pi) % (2 * torch.pi) - torch.pi
            new_theta = (old_theta + (old_target_theta_dot - cartesian_robot_rtf_vel[2] / new_distance) * timestep + torch.pi) % (2 * torch.pi) - torch.pi

            ret_mean[0] = new_distance
            ret_mean[1] = new_phi
            ret_mean[2] = new_theta
            ret_mean[3] = old_target_distance_dot - cartesian_robot_rtf_vel[0]
            ret_mean[4] = old_target_phi_dot - cartesian_robot_rtf_vel[1] / new_distance
            ret_mean[5] = old_target_theta_dot - cartesian_robot_rtf_vel[2] / new_distance
            ret_mean[6] = x_mean[6]
            
        return ret_mean, cov
