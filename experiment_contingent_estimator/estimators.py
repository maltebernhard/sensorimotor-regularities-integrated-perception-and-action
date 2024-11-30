import torch
from components.estimator import RecursiveEstimator

# ==================================== Specific Implementations ==============================================

class Robot_State_Estimator_Acc(RecursiveEstimator):
    def __init__(self, device):
        super().__init__("RobotState", 7, device)
        self.default_state = torch.tensor([0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 10.0], device=device)
        self.default_cov = 1e0 * torch.eye(7, device=device)
        self.default_motion_noise = 1e-1 * torch.eye(7, device=device)

    def forward_model(self, x_mean: torch.Tensor, cov: torch.Tensor, u: torch.Tensor):
        """
        Args:
            x_mean: Mean of the state
                x[0]: frontal vel       v_x
                x[1]: lateral vel       v_y
                x[2]: rotational vel    w
                x[3]: frontal acc       v_dot_x
                x[4]: lateral acc       v_dot_y
                x[5]: rotational acc    w_dot
                x[6]: target_distance   d
                
            u: Control input
                u[0]: del_t
                u[1]: frontal acc       v_dot_x
                u[2]: lateral acc       v_dot_y
                u[3]: rotational acc    w_dot
            """
        
        ret_mean = torch.empty_like(x_mean)
        ret_cov = cov

        d = x_mean[6] - x_mean[0] * u[0]

        ret_mean[0] = x_mean[0] + u[1] * u[0]
        ret_mean[1] = x_mean[1] + u[2] * u[0]
        #ret_mean[2] = x_mean[2] + x_mean[5] * u[0]
        ret_mean[2] = - x_mean[1] / d + u[3] * u[0]
        ret_mean[3] = u[1]
        ret_mean[4] = u[2]
        ret_mean[5] = u[3]
        #ret_mean[6] = x_mean[6]
        ret_mean[6] = d

        return ret_mean, ret_cov
    
class Robot_State_Estimator_Vel(RecursiveEstimator):
    def __init__(self, device):
        super().__init__("RobotState", 4, device)
        self.default_state = torch.tensor([0.0, 0.0, 0.0, 10.0], device=device)
        self.default_cov = 1e0 * torch.eye(4, device=device)
        self.default_motion_noise = 1e-1 * torch.eye(4, device=device)

    def forward_model(self, x_mean: torch.Tensor, cov: torch.Tensor, u: torch.Tensor):
        """
        Args:
            x_mean: Mean of the state
                x[0]: frontal vel       v_x
                x[1]: lateral vel       v_y
                x[2]: rotational vel    w
                x[3]: target_distance   d
                
            u: Control input
                u[0]: del_t
                u[1]: frontal vel       v_x
                u[2]: lateral vel       v_y
                u[3]: rotational vel    w
            """
        
        ret_mean = torch.empty_like(x_mean)

        d = x_mean[3] - x_mean[1] * u[0]

        return torch.stack([
            u[1],
            u[2],
            #u[3],
            - x_mean[1] / d,
            d
        ]).squeeze(), cov
    
class Polar_Pos_Estimator(RecursiveEstimator):
    """
    Estimator for Target state x:
    x[0]: target distance           r
    x[1]: target offset angle       theta
    x[2]: del target distance       r_dot
    x[3]: del target offset angle   theta_dot
    """
    def __init__(self, device, id: str):
        super().__init__(id, 4, device)
        self.default_state = torch.tensor([10.0, 0.0, 0.0, 0.0], device=device)
        self.default_cov = 1e3 * torch.eye(4, device=device)
        self.default_motion_noise = 1e0 * torch.eye(4, device=device)

    def forward_model(self, x_mean: torch.Tensor, cov: torch.Tensor, u: torch.Tensor):
        timestep = u[0]
        ret_mean = torch.empty_like(x_mean)
        ret_mean[:2] = x_mean[:2] + x_mean[2:] * timestep
        ret_mean[1] = (ret_mean[1] + torch.pi) % (2 * torch.pi) - torch.pi
        ret_mean[2:] = x_mean[2:]
        return ret_mean, cov