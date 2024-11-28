import torch
from components.estimator import RecursiveEstimator

# ==================================== Specific Implementations ==============================================

class Robot_Vel_Estimator(RecursiveEstimator):
    def __init__(self, device):
        super().__init__("RobotVel", 3, device)

    def forward_model(self, x_mean: torch.Tensor, cov: torch.Tensor, u: torch.Tensor):
        """
        Args:
            x_mean: Mean of the state
            u: Control input
                u[0]: Frontal acceleration  float
                u[1]: Lateral acceleration  float
                u[2]: Angular acceleration  float
                u[3]: del_t                 float
            """
        ret_mean = torch.empty_like(x_mean)
        vel_rot_new = x_mean[2] + u[2] * u[3]
        theta = vel_rot_new * u[3]
        rotation_matrix = torch.stack([
            torch.stack([torch.cos(-theta), -torch.sin(-theta)]),
            torch.stack([torch.sin(-theta), torch.cos(-theta)]),
        ]).squeeze()
        vel_trans_new = torch.matmul(rotation_matrix, x_mean[:2] + u[:2] * u[3])
        ret_mean[:2] = vel_trans_new
        ret_mean[2] = vel_rot_new
        return ret_mean, cov
    
class Polar_Pos_Estimator_External_Vel(RecursiveEstimator):
    """
    Estimator for Target state x:
    x[0]: target offset angle
    x[1]: target distance
    """
    def __init__(self, device, id: str, ):
        super().__init__(id, 2, device)

    def forward_model(self, x_mean: torch.Tensor, cov: torch.Tensor, u: torch.Tensor):
        ret_mean = torch.empty_like(x_mean)
        robot_vel = u[:2]

        robot_target_frame_rotation_matrix = torch.stack([
            torch.stack([torch.cos(-x_mean[1]), -torch.sin(-x_mean[1])]),
            torch.stack([torch.sin(-x_mean[1]), torch.cos(-x_mean[1])]),
        ]).squeeze()
        robot_target_frame_vel = torch.matmul(robot_target_frame_rotation_matrix, robot_vel)

        ret_mean[0] = (x_mean[0].pow(2) + (robot_target_frame_vel[1]*u[3]).pow(2)).sqrt() - robot_target_frame_vel[0] * u[3]
        ret_mean[1] = x_mean[1] - u[2] * u[3] - torch.atan2(robot_target_frame_vel[1] * u[3], x_mean[0])

        return ret_mean, cov
    
class Polar_Pos_Estimator_Internal_Vel(RecursiveEstimator):
    """
    Estimator for Target state x:
    x[0]: target offset angle
    x[1]: target distance
    x[2]: del target offset angle
    x[3]: del target distance
    """
    def __init__(self, device, id: str):
        super().__init__(id, 4, device)

    def forward_model(self, x_mean: torch.Tensor, cov: torch.Tensor, u: torch.Tensor):
        timestep = u[0]
        ret_mean = torch.empty_like(x_mean)
        ret_mean[:2] = x_mean[:2] + x_mean[2:] * timestep
        ret_mean[1] = (ret_mean[1] + torch.pi) % (2 * torch.pi) - torch.pi
        ret_mean[2:] = x_mean[2:]
        return ret_mean, cov

class Pos_Estimator_Internal_Vel(RecursiveEstimator):
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

# Same as Target Pos estimator - unnecessary
class Pos_Estimator_External_Vel(RecursiveEstimator):
    """
    Estimator for Target state x:
    x[0]: target frontal offset
    x[1]: target lateral offset
    """
    def __init__(self, device, id: str):
        super().__init__(id, 2, device)

    def forward_model(self, x_mean, cov: torch.Tensor, u):
        ret_mean = torch.empty_like(x_mean)
        robot_vel = u[:2]
        robot_vel_rot = u[2]
        timestep = u[3]
        rotation_matrix = torch.stack([
            torch.stack([torch.cos(-robot_vel_rot*timestep), -torch.sin(-robot_vel_rot*timestep)]),
            torch.stack([torch.sin(-robot_vel_rot*timestep), torch.cos(-robot_vel_rot*timestep)]),
        ]).squeeze()
        ret_mean = torch.matmul(rotation_matrix, x_mean - timestep * robot_vel)

        ret_cov = torch.empty_like(cov)
        ret_cov = torch.matmul(rotation_matrix, torch.matmul(cov, rotation_matrix.t()))
        return ret_mean, ret_cov
    
class Target_Pos_Estimator_No_Forward(RecursiveEstimator):
    """
    Estimator for Target state x:
    x[0]: target frontal offset
    x[1]: target lateral offset
    """
    def __init__(self, device):
        super().__init__("TargetPosNoForward", 2, device)

    def forward_model(self, x_mean, cov: torch.Tensor, u):
        return x_mean, cov
    
class Obstacle_Rad_Estimator(RecursiveEstimator):
    """
    Estimator for obstacle radius state x:
    x[0]: obstacle radius
    """
    def __init__(self, device, id: str):
        super().__init__(id, 1, device)
        self.set_static_motion_noise(torch.eye(1, device=device)*1e-1)

    def forward_model(self, x_mean, cov: torch.Tensor, u):
        return x_mean, cov

class Target_Distance_Estimator(RecursiveEstimator):
    """
    Estimator for Target state x:
    x[0]: target angular offset
    x[1]: target distance
    x[2]: del target angular offset
    x[3]: del target distance
    """
    def __init__(self, device):
        super().__init__("TargetDist", 2, device)

    def forward_model(self, x_mean, cov: torch.Tensor, u):
        ret_mean = torch.empty_like(x_mean)

        robot_vel_rot = u[2]
        target_angular_offset = x_mean[0]
        timestep = u[3]

        rotation_matrix = torch.stack([
            torch.stack([torch.cos(-target_angular_offset), -torch.sin(-target_angular_offset)]),
            torch.stack([torch.sin(-target_angular_offset), torch.cos(-target_angular_offset)]),
        ]).squeeze()
        target_frame_vel = torch.matmul(rotation_matrix, u[:2])
        ret_mean[0] = x_mean[0] - timestep * robot_vel_rot
        ret_mean[1] = x_mean[1] - timestep * target_frame_vel[0]
        return ret_mean, cov