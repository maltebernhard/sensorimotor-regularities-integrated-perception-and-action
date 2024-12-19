import torch
from components.estimator import RecursiveEstimator

# ==================================== Specific Implementations ==============================================

class Robot_State_Estimator_Vel(RecursiveEstimator):
    def __init__(self, device):
        super().__init__("RobotVel", 3, device)
        self.default_state = torch.tensor([0.0, 0.0, 0.0], device=device)
        self.default_cov = 1e1 * torch.eye(3, device=device)
        self.default_motion_noise = 1e0 * torch.eye(3, device=device)

    def forward_model(self, x_mean: torch.Tensor, cov: torch.Tensor, u: torch.Tensor):
        """
        Args:
            x_mean: Mean of the state
                x[0]: frontal vel           v_x
                x[1]: lateral vel           v_y
                x[2]: rotational vel        w
            u: Control input
                u[0]: del_t
                u[1]: frontal vel           v_x
                u[2]: lateral vel           v_y
                u[3]: rotational vel        w
        """
        return torch.stack([
            u[1],
            u[2],
            u[3],
        ]).squeeze(), cov

class Robot_State_Estimator_Acc(RecursiveEstimator):
    def __init__(self, device):
        super().__init__("RobotVel", 3, device)
        self.default_state = torch.tensor([0.0, 0.0, 0.0], device=device)
        self.default_cov = 1e1 * torch.eye(3, device=device)
        self.default_motion_noise = 1e0 * torch.eye(3, device=device)

    def forward_model(self, x_mean: torch.Tensor, cov: torch.Tensor, u: torch.Tensor):
        """
        Args:
            x_mean: Mean of the state
                x[0]: frontal vel           v_x
                x[1]: lateral vel           v_y
                x[2]: rotational vel        w
            u: Control input
                u[0]: del_t
                u[1]: frontal acc           v_x_dot
                u[2]: lateral acc           v_y_dot
                u[3]: rotational acc        w_dot
        """
        del_t = u[0]
        acc_x = u[1]
        acc_y = u[2]
        acc_rot = u[3]
        v_x_old = x_mean[0]
        v_y_old = x_mean[1]
        w_old = x_mean[2]

        # translational velocity update
        v_x = v_x_old + acc_x * del_t
        v_y = v_y_old + acc_y * del_t
        vel_trans = torch.stack([v_x, v_y])
        if vel_trans.norm() > 8.0:
            vel_trans = vel_trans / vel_trans.norm() * 8.0
        # rotational velocity update
        w = w_old + acc_rot * del_t
        if torch.abs(w) > 3.0:
            w = w / torch.abs(w) * 3.0

        del_rotation = - w * del_t
        del_rotation_matrix = torch.stack([
            torch.stack([torch.cos(del_rotation), -torch.sin(del_rotation)]),
            torch.stack([torch.sin(del_rotation), torch.cos(del_rotation)]),
        ]).squeeze()
        vel = torch.matmul(del_rotation_matrix, vel_trans)

        return torch.stack([
            vel[0],
            vel[1],
            w,
        ]).squeeze(), cov