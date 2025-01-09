import torch

def rotate_vector_2d(rotation_angle, vector):
    if not torch.is_tensor(rotation_angle):
        rotation_angle = torch.tensor(rotation_angle)
    rotation_matrix = torch.stack([
        torch.stack([torch.cos(-rotation_angle), -torch.sin(-rotation_angle)]),
        torch.stack([torch.sin(-rotation_angle), torch.cos(-rotation_angle)]),
    ]).squeeze()
    return torch.matmul(rotation_matrix, vector)