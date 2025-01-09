import torch

def get_robot_target_frame_vel(offset_angle, robot_vel):
    if not torch.is_tensor(offset_angle):
        offset_angle = torch.tensor(offset_angle)
    robot_target_frame_rotation_matrix = torch.stack([
        torch.stack([torch.cos(-offset_angle), -torch.sin(-offset_angle)]),
        torch.stack([torch.sin(-offset_angle), torch.cos(-offset_angle)]),
    ]).squeeze()
    return torch.matmul(robot_target_frame_rotation_matrix, robot_vel)