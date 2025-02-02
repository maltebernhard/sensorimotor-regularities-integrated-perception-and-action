import numpy as np
import torch

def rotate_vector_2d(rotation_angle, vector):
    if type(vector) == torch.Tensor:
        if not torch.is_tensor(rotation_angle):
            rotation_angle = torch.tensor(rotation_angle)
        rotation_matrix = torch.stack([
            torch.stack([torch.cos(rotation_angle), -torch.sin(rotation_angle)]),
            torch.stack([torch.sin(rotation_angle), torch.cos(rotation_angle)]),
        ]).squeeze()
        return torch.matmul(rotation_matrix, vector)
    elif type(vector) == np.ndarray:
        rotation_matrix = np.array([
            [np.cos(rotation_angle), -np.sin(rotation_angle)],
            [np.sin(rotation_angle), np.cos(rotation_angle)],
        ])
        return np.matmul(rotation_matrix, vector)