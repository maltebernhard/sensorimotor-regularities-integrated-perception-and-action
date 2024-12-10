import torch
from components.estimator import RecursiveEstimator

# ==================================== Specific Implementations ==============================================

class Robot_Vel_Estimator_Vel(RecursiveEstimator):
    def __init__(self, device):
        super().__init__("RobotVel", 3, device)
        self.default_state = torch.tensor([0.0, 0.0, 0.0], device=device)
        self.default_cov = 1e1 * torch.eye(3, device=device)
        self.default_motion_noise = 1e-6 * torch.eye(3, device=device)

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

class Robot_Vel_Estimator_Acc(RecursiveEstimator):
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
    
class Polar_Pos_Estimator_Vel(RecursiveEstimator):
    """
    Estimator for Target state x:
    x[0]: target distance
    x[1]: target offset angle
    x[2]: del target distance
    x[3]: del target offset angle
    """
    def __init__(self, device, id: str):
        super().__init__(id, 4, device)
        self.default_state = torch.tensor([10.0, 0.0, 0.0, 0.0], device=device)
        self.default_cov = 1e3 * torch.eye(4, device=device)
        self.default_motion_noise = 1e-6 * torch.tensor([0.1, 0.0001, 0.1, 0.0001], device=device)

    def forward_model(self, x_mean: torch.Tensor, cov: torch.Tensor, u: torch.Tensor):
        timestep = u[0]
        offset_angle = x_mean[1]
        robot_target_frame_rotation_matrix = torch.stack([
            torch.stack([torch.cos(-offset_angle), -torch.sin(-offset_angle)]),
            torch.stack([torch.sin(-offset_angle), torch.cos(-offset_angle)]),
        ]).squeeze()
        robot_target_frame_vel = torch.matmul(robot_target_frame_rotation_matrix, u[1:3])

        # print("FORWARD:")
        # print("vel rot: ", u[3])
        # print("angle old: ", x_mean[1])
        # print("angle new: ", x_mean[1] + (- robot_target_frame_vel[1]/x_mean[0] - u[3]) * timestep)
        # print("angle step: ", (- robot_target_frame_vel[1]/x_mean[0] - u[3]) * timestep)

        ret_mean = torch.empty_like(x_mean)
        ret_mean[0] = x_mean[0] + (- robot_target_frame_vel[0]) * timestep
        ret_mean[1] = (x_mean[1] + (- robot_target_frame_vel[1]/x_mean[0] - u[3]) * timestep + torch.pi) % (2 * torch.pi) - torch.pi
        ret_mean[2] = - robot_target_frame_vel[0]
        ret_mean[3] = - robot_target_frame_vel[1]/x_mean[0] - u[3]
        return ret_mean, cov
    
class Polar_Pos_Estimator_Acc(RecursiveEstimator):
    """
    Estimator for Target state x:
    x[0]: target distance
    x[1]: target offset angle
    x[2]: del target distance
    x[3]: del target offset angle
    """
    def __init__(self, device, id: str):
        super().__init__(id, 4, device)
        self.default_state = torch.tensor([10.0, 0.0, 0.0, 0.0], device=device)
        self.default_cov = 1e3 * torch.eye(4, device=device)
        self.default_motion_noise = 1e0 * torch.eye(4, device=device)

    def forward_model(self, x_mean: torch.Tensor, cov: torch.Tensor, u: torch.Tensor):
        timestep = u[0]
        offset_angle = x_mean[1]
        robot_target_frame_rotation_matrix = torch.stack([
            torch.stack([torch.cos(-offset_angle), -torch.sin(-offset_angle)]),
            torch.stack([torch.sin(-offset_angle), torch.cos(-offset_angle)]),
        ]).squeeze()
        robot_target_frame_acc = torch.matmul(robot_target_frame_rotation_matrix, u[1:3])

        ret_mean = torch.empty_like(x_mean)
        ret_mean[0] = x_mean[0] + (x_mean[2] - robot_target_frame_acc[0] * timestep) * timestep
        ret_mean[1] = (x_mean[1] + (x_mean[3] - (robot_target_frame_acc[1]/x_mean[0] + u[3]) * timestep) * timestep + torch.pi) % (2 * torch.pi) - torch.pi
        ret_mean[2] = x_mean[2] - robot_target_frame_acc[0] * timestep
        ret_mean[3] = x_mean[3] - (robot_target_frame_acc[1]/x_mean[0] + u[3]) * timestep
        return ret_mean, cov

class Cartesian_Pos_Estimator(RecursiveEstimator):
    """
    Estimator for Target state x:
    x[0]: target frontal offset
    x[1]: target lateral offset
    x[2]: del target frontal offset
    x[3]: del target lateral offset
    x[4]: del robot target target frame rotation
    """
    def __init__(self, device, id: str):
        super().__init__(id, 5, device)
        self.default_state = torch.tensor([10.0, 0.0, 0.0, 0.0, 0.0], device=device)
        self.default_cov = 1e3 * torch.eye(5, device=device)
        self.default_motion_noise = 1e0 * torch.eye(5, device=device)

    def forward_model(self, x_mean, cov: torch.Tensor, u):
        ret_mean = torch.empty_like(x_mean)
        timestep = u[0]
        rotation_matrix = torch.stack([
            torch.stack([torch.cos(x_mean[4]*timestep), -torch.sin(x_mean[4]*timestep)]),
            torch.stack([torch.sin(x_mean[4]*timestep), torch.cos(x_mean[4]*timestep)]),
        ]).squeeze()
        ret_mean[:2] = torch.matmul(rotation_matrix, x_mean[:2] + timestep * x_mean[2:4])
        ret_mean[2:4] = torch.matmul(rotation_matrix, x_mean[2:4])
        ret_mean[4] = x_mean[4]

        ret_cov = torch.empty_like(cov)
        ret_cov[:2, :2] = torch.matmul(rotation_matrix, torch.matmul(cov[:2, :2], rotation_matrix.t()))
        ret_cov[2:4, 2:4] = torch.matmul(rotation_matrix, torch.matmul(cov[2:4, 2:4], rotation_matrix.t()))

        ret_cov[:2, 2:4] = torch.matmul(rotation_matrix, cov[:2, 2:4])
        ret_cov[2:4, :2] = torch.matmul(cov[2:4, :2], rotation_matrix.t())
        ret_cov[:4, 4] = cov[:4, 4]
        ret_cov[4, :4] = cov[4, :4]
        ret_cov[4, 4] = cov[4, 4]
        return ret_mean, ret_cov
    
class Obstacle_Rad_Estimator(RecursiveEstimator):
    """
    Estimator for obstacle radius state x:
    x[0]: obstacle radius
    """
    def __init__(self, device, id: str):
        super().__init__(id, 1, device)
        self.default_state = torch.tensor([1.0], device=device)
        self.default_cov = 1e1 * torch.eye(1, device=device)
        self.default_motion_noise = 1e-2 * torch.eye(1, device=device)

    def forward_model(self, x_mean, cov: torch.Tensor, u):
        return x_mean, cov