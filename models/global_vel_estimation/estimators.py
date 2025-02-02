import torch
from components.estimator import RecursiveEstimator
from components.helpers import rotate_vector_2d

# ==================================================================================

class Robot_Vel_Estimator(RecursiveEstimator):
    def __init__(self):
        super().__init__("RobotVel", 3)
        self.default_state = torch.tensor([0.0, 0.0, 0.0])
        self.default_cov = 1e1 * torch.eye(3)
        self.default_motion_noise = torch.eye(3) * torch.tensor([3e-1, 3e-1, 1e-1])
        self.update_uncertainty: torch.Tensor = torch.eye(self.state_dim) * torch.tensor([9e-1, 9e-1, 6e-1])

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
    x[0]: target distance (robot frame)
    x[1]: target offset angle (robot frame)
    x[2]: radial vel (global frame)
    x[3]: angular vel (global frame)
    """
    def __init__(self, object_name:str="Target"):
        super().__init__(f'Polar{object_name}Pos', 4)
        self.default_state = torch.tensor([10.0, 0.0, 0.0, 0.0])
        self.default_cov = 1e3 * torch.eye(4)
        self.default_motion_noise = torch.eye(4) * torch.tensor([1e-1, 5e-1, 1e-1, 5e-1])
        self.update_uncertainty = torch.eye(4) * torch.tensor([1e-1, 1e-2, 1e-1, 1e-2])

    # TODO: Problem: Angular vel is used for Triangulation, but is not updated by control input
    # Angular Vel currently reflects the global angular velocity due to target movement
    # Measurement model however does not reflect that, it only updates to perceived angular vel
        # I can equip measurement update to account for robot rotation, but then it still mixes target translation and robot translation
    # Solution: I need to separately account for the target's own velocity (pure estimate, no direct measurement) and the angular vel caused by robot translation
        # If I calculate the angular vel caused by robot translation from the estimated distance, and then update the estimated distance with that estimate, is that redundant or an AI?
            # ANSWER: Redundant
    
    def forward_model(self, x_mean: torch.Tensor, cov: torch.Tensor, u: torch.Tensor):
        timestep = u[0]
        rtf_vel = rotate_vector_2d(-x_mean[1], u[1:3])
        ret_mean = torch.empty_like(x_mean)
        ret_mean[0] = x_mean[0] + (x_mean[2] - rtf_vel[0]) * timestep
        ret_mean[1] = (x_mean[1] + (x_mean[3] - rtf_vel[1]/x_mean[0] - u[3]) * timestep + torch.pi) % (2 * torch.pi) - torch.pi
        ret_mean[2] = x_mean[2]
        ret_mean[3] = x_mean[3]
        return ret_mean, cov