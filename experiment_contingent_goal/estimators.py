import torch
from components.estimator import RecursiveEstimator

# ==================================== Specific Implementations ==============================================

class Robot_Vel_Estimator_Acc(RecursiveEstimator):
    def __init__(self, device):
        super().__init__("RobotVel", 6, device)
        self.default_state = torch.tensor([0.0, 0.0, 0.0, 0.0, 0.0, 0.0], device=device)
        self.default_cov = 1e0 * torch.eye(6, device=device)
        self.default_motion_noise = 1e-1 * torch.eye(6, device=device)

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
                
            u: Control input
                u[0]: del_t
                u[1]: frontal acc       v_dot_x
                u[2]: lateral acc       v_dot_y
                u[3]: rotational acc    w_dot
            """
        
        ret_mean = torch.empty_like(x_mean)
        ret_cov = cov

        ret_mean[2] = x_mean[2] + u[3] * u[0]
        if abs(x_mean[2] + u[3] * u[0]) > 3.0:
            ret_mean[2] = (x_mean[2] + u[3] * u[0]) / abs(x_mean[2] + u[3] * u[0]) * 3.0
        ret_mean[3] = u[1]
        ret_mean[4] = u[2]
        ret_mean[5] = u[3]

        vel = x_mean[:2] + u[1:3] * u[0]

        rotation = -ret_mean[2]*u[0]
        rotation_matrix = torch.stack([
            torch.stack([torch.cos(rotation), -torch.sin(rotation)]),
            torch.stack([torch.sin(rotation), torch.cos(rotation)]),
        ]).squeeze()
        vel = torch.matmul(rotation_matrix, vel)

        ret_mean[:2] = vel
        if vel.norm() > 8.0:
            ret_mean[:2] = vel / vel.norm() * 8.0

        return ret_mean, ret_cov
    
class Robot_Vel_Estimator_Vel(RecursiveEstimator):
    def __init__(self, device):
        super().__init__("RobotVel", 3, device)
        self.default_state = torch.tensor([0.0, 0.0, 0.0], device=device)
        self.default_cov = 1e0 * torch.eye(3, device=device)
        self.default_motion_noise = 1e-1 * torch.eye(3, device=device)

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
        self.default_motion_noise = 1e0 * torch.eye(4, device=device)

    def forward_model(self, x_mean: torch.Tensor, cov: torch.Tensor, u: torch.Tensor):
        timestep = u[0]
        offset_angle = x_mean[1]
        robot_target_frame_rotation_matrix = torch.stack([
            torch.stack([torch.cos(-offset_angle), -torch.sin(-offset_angle)]),
            torch.stack([torch.sin(-offset_angle), torch.cos(-offset_angle)]),
        ]).squeeze()
        robot_target_frame_vel = torch.matmul(robot_target_frame_rotation_matrix, u[1:3])

        # print("Robot Vel: ", u[1:])
        # print("Robot Target Frame Vel: ", robot_target_frame_vel)

        ret_mean = torch.empty_like(x_mean)
        ret_mean[0] = x_mean[0] + (- robot_target_frame_vel[0]) * timestep
        ret_mean[1] = (x_mean[1] + (- robot_target_frame_vel[1]/x_mean[0] - u[3]) * timestep + torch.pi) % (2 * torch.pi) - torch.pi
        ret_mean[2] = - robot_target_frame_vel[0]
        ret_mean[3] = - torch.atan2(robot_target_frame_vel[1], x_mean[0]) - u[3]
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
        """
        Args:
            x_mean: Mean of the state
                x[0]: target distance       r
                x[1]: orientation offset    theta
                x[2]: distance dot          r_dot
                x[3]: orientation dot       theta_dot
                
            u: Control input
                u[0]: del_t
                u[1]: frontal acc       v_x
                u[2]: lateral acc       v_y
                u[3]: rotational acc    w
        """
        timestep = u[0]
        offset_angle = x_mean[1]
        robot_target_frame_rotation_matrix = torch.stack([
            torch.stack([torch.cos(-offset_angle), -torch.sin(-offset_angle)]),
            torch.stack([torch.sin(-offset_angle), torch.cos(-offset_angle)]),
        ]).squeeze()
        robot_target_frame_acc = torch.matmul(robot_target_frame_rotation_matrix, u[1:3])

        new_vel_rot = x_mean[3] - (robot_target_frame_acc[1]/x_mean[0] + u[3]) * timestep
        #new_vel_rot = x_mean[3] - u[3] * timestep

        if torch.abs(new_vel_rot) > 3.0:
            new_vel_rot = new_vel_rot / torch.abs(new_vel_rot) * 3.0

        ret_mean = torch.empty_like(x_mean)
        ret_mean[0] = x_mean[0] + (x_mean[2] - robot_target_frame_acc[0] * timestep) * timestep
        ret_mean[1] = (x_mean[1] + new_vel_rot * timestep + torch.pi) % (2 * torch.pi) - torch.pi

        ret_mean[2] = x_mean[2] - robot_target_frame_acc[0] * timestep
        ret_mean[3] = new_vel_rot
        return ret_mean, cov
