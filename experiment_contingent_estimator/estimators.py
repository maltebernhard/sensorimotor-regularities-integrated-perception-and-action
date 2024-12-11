import torch
from components.estimator import RecursiveEstimator

# ==================================== Specific Implementations ==============================================

class Robot_State_Estimator_Vel(RecursiveEstimator):
    def __init__(self, device):
        super().__init__("RobotState", 7, device)
        self.default_state = torch.tensor([0.0, 0.0, 0.0, 10.0, 0.0, 0.0, 0.0], device=device)
        self.default_cov = 1e1 * torch.eye(7, device=device)
        self.default_motion_noise = 1e0 * torch.eye(7, device=device)

    def forward_model(self, x_mean: torch.Tensor, cov: torch.Tensor, u: torch.Tensor):
        """
        Args:
            x_mean: Mean of the state
                x[0]: frontal vel           v_x
                x[1]: lateral vel           v_y
                x[2]: rotational vel        w
                x[3]: target_distance       d
                x[4]: target_angle          theta
                x[5]: target_distance_dot   d_dot
                x[6]: target_angle_dot      theta_dot
            u: Control input
                u[0]: del_t
                u[1]: frontal vel           v_x
                u[2]: lateral vel           v_y
                u[3]: rotational vel        w
        """
        del_t = u[0]
        d_old = x_mean[3]
        theta = x_mean[4]
        d_dot = x_mean[5]
        theta_dot = x_mean[6]

        # translational velocity update
        v_x = u[1]
        v_y = u[2]
        # rotational velocity update
        #w = u[3]
        w = - v_y / d_old           # GazeFixation
        # target position update
        d = d_old - v_x * del_t
        theta = torch.zeros(1)      # GazeFixation
        d_dot = v_x
        theta_dot = torch.zeros(1)  # GazeFixation

        return torch.stack([
            torch.atleast_1d(v_x),
            torch.atleast_1d(v_y),
            torch.atleast_1d(w),
            torch.atleast_1d(d),
            torch.atleast_1d(theta),
            torch.atleast_1d(d_dot),
            torch.atleast_1d(theta_dot),
        ]).squeeze(), cov

class Robot_State_Estimator_Acc(RecursiveEstimator):
    def __init__(self, device):
        super().__init__("RobotState", 7, device)
        self.default_state = torch.tensor([0.0, 0.0, 0.0, 10.0, 0.0, 0.0, 0.0], device=device)
        self.default_cov = 1e1 * torch.eye(7, device=device)
        self.default_motion_noise = 1e0 * torch.eye(7, device=device)

    def forward_model(self, x_mean: torch.Tensor, cov: torch.Tensor, u: torch.Tensor):
        """
        Args:
            x_mean: Mean of the state
                x[0]: frontal vel           v_x
                x[1]: lateral vel           v_y
                x[2]: rotational vel        w
                x[3]: target_distance       d
                x[4]: target_angle          theta
                x[5]: target_distance_dot   d_dot
                x[6]: target_angle_dot      theta_dot
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
        d_old = x_mean[3]

        # translational velocity update
        v_x = v_x_old + acc_x * del_t
        v_y = v_y_old + acc_y * del_t
        vel_trans = torch.stack([v_x, v_y])
        if vel_trans.norm() > 8.0:
            vel_trans = vel_trans / vel_trans.norm() * 8.0
        # rotational velocity update
        #w = w_old + acc_rot * del_t
        w = - vel_trans[1] / d_old      # GazeFixation
        if torch.abs(w) > 3.0:
            w = w / torch.abs(w) * 3.0
        # target position update
        d = vel_trans[1] / w
        theta = torch.zeros(1)          # GazeFixation
        d_dot = vel_trans[0]
        theta_dot = torch.zeros(1)      # GazeFixation

        del_rotation = - w * del_t
        del_rotation_matrix = torch.stack([
            torch.stack([torch.cos(del_rotation), -torch.sin(del_rotation)]),
            torch.stack([torch.sin(del_rotation), torch.cos(del_rotation)]),
        ]).squeeze()
        vel = torch.matmul(del_rotation_matrix, vel_trans)

        return torch.stack([
            torch.atleast_1d(vel[0]),
            torch.atleast_1d(vel[1]),
            torch.atleast_1d(w),
            torch.atleast_1d(d),
            torch.atleast_1d(theta),
            torch.atleast_1d(d_dot),
            torch.atleast_1d(theta_dot),
        ]).squeeze(), cov
