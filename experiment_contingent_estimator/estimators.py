import torch
from components.estimator import RecursiveEstimator

# ==================================== Specific Implementations ==============================================

class Robot_State_Estimator(RecursiveEstimator):
    def __init__(self, device):
        super().__init__("RobotState", 8, device)

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
            """
        
        ret_mean = torch.empty_like(x_mean)
        ret_cov = torch.empty_like(cov)

        d = x_mean[6] - x_mean[0] * u[0]

        ret_mean[0] = x_mean[0] + x_mean[3] * u[0]
        ret_mean[1] = x_mean[1] + x_mean[4] * u[0]
        #ret_mean[2] = x_mean[2] + x_mean[5] * u[0]
        ret_mean[2] = - x_mean[1] / d + x_mean[5] * u[0]
        ret_mean[3] = x_mean[3]
        ret_mean[4] = x_mean[4]
        ret_mean[5] = x_mean[5]
        #ret_mean[6] = x_mean[6]
        ret_mean[6] = d

        return ret_mean, ret_cov