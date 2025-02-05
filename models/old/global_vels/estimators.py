import torch
from components.estimator import RecursiveEstimator
from components.helpers import rotate_vector_2d

# ==================================== Specific Implementations ==============================================

class PolarPos_Tri_Div_Global_Estimator(RecursiveEstimator):
    """
    Estimator for Target state x:
    x[0]: target distance
    x[1]: target offset angle
    x[2]: target distance dot
    x[3]: target offset angle dot
    x[4]: target global radial vel
    x[5]: target global angular vel
    x[6]: target visual angle
    x[7]: target visual angle dot
    """
    def __init__(self, object_name:str="Target"):
        super().__init__(f'Polar{object_name}Pos', 8)
        self.default_state = torch.tensor([10.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0])
        self.default_cov = 1e3 * torch.eye(8)
        self.default_motion_noise = torch.eye(8) * torch.tensor([1e-1, 1e-2, 1e-1, 1e-2, 5e-1, 1e-1, 1e-2, 1e-2])
        self.update_uncertainty = torch.eye(8) * torch.tensor([1e-1, 1e-2, 1e-1, 1e-2, 1e-1, 1e-2, 1e-2, 1e-2])

    def forward_model(self, x_mean: torch.Tensor, cov: torch.Tensor, u: torch.Tensor):
        timestep = u[0]

        # update target distance and angle
        pre_step_distance = x_mean[0]
        pre_step_offset_angle = x_mean[1]
        pre_step_rtf_vel = rotate_vector_2d(-x_mean[1], u[1:3])

        pre_step_radial_object_vel = x_mean[4]
        pre_step_angular_object_vel = x_mean[5]

        pre_step_vis_angle = x_mean[6]

        pre_step_distance_dot = - pre_step_rtf_vel[0] + pre_step_radial_object_vel
        pre_step_angle_dot = - pre_step_rtf_vel[1]/pre_step_distance - u[3] + pre_step_angular_object_vel

        estimated_target_radius = torch.sin(pre_step_vis_angle / 2) * pre_step_distance

        post_step_distance = torch.abs(pre_step_distance + pre_step_distance_dot * timestep)
        post_step_offset_angle = (pre_step_offset_angle + pre_step_angle_dot * timestep + torch.pi) % (2 * torch.pi) - torch.pi

        # transform global target velocity to robot rotation through staying constant in cartesian space
        pre_step_global_cartesian_rtf_vel = torch.tensor([pre_step_radial_object_vel, pre_step_distance * pre_step_angular_object_vel])
        post_step_global_cartesian_rtf_vel = rotate_vector_2d(post_step_offset_angle - pre_step_offset_angle, pre_step_global_cartesian_rtf_vel)
        post_step_global_phi_dot = post_step_global_cartesian_rtf_vel[1] / post_step_distance

        # update target distance dot
        post_step_rtf_vel = rotate_vector_2d(-post_step_offset_angle, u[1:3])
        post_step_distance_dot = - post_step_rtf_vel[0] + post_step_global_cartesian_rtf_vel[0]
        post_step_angle_dot_robot_component = - post_step_rtf_vel[1]/post_step_distance - u[3]

        # update visual angle and visual angle dot
        post_step_vis_angle = 2 * torch.asin(estimated_target_radius / post_step_distance)
        post_step_vis_angle_dot = -2 / post_step_offset_angle * torch.tan(post_step_vis_angle/2) * (post_step_distance_dot)

        ret_mean = torch.empty_like(x_mean)
        ret_mean[0] = post_step_distance
        ret_mean[1] = post_step_offset_angle
        ret_mean[2] = post_step_distance_dot
        ret_mean[3] = post_step_angle_dot_robot_component + post_step_global_phi_dot
        ret_mean[4] = post_step_global_cartesian_rtf_vel[0]
        ret_mean[5] = post_step_global_phi_dot
        ret_mean[6] = post_step_vis_angle
        ret_mean[7] = post_step_vis_angle_dot

        return ret_mean, cov
