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
    
def transform_vector_to_rtf(v, phi, theta):
    # Ensure input is a numpy array or torch tensor
    if isinstance(v, np.ndarray):
        # Compute the direction vector for the new frame
        direction = np.array([
            np.sin(theta) * np.cos(phi),
            np.sin(theta) * np.sin(phi),
            np.cos(theta)
        ])
        # Compute w[1] orthogonal to direction in the xy-plane
        w1 = np.array([-np.sin(phi), np.cos(phi), 0])
        # Compute w[2] orthogonal to both direction and w[1]
        w2 = np.cross(direction, w1)
        # Transform the vector to the new frame
        w = np.array([np.dot(v, direction), np.dot(v, w1), np.dot(v, w2)])
        return w
    elif isinstance(v, torch.Tensor):
        # Compute the direction vector for the new frame
        direction = torch.tensor([
            torch.sin(theta) * torch.cos(phi),
            torch.sin(theta) * torch.sin(phi),
            torch.cos(theta)
        ], dtype=v.dtype, device=v.device)
        # Compute w[1] orthogonal to direction in the xy-plane
        w1 = torch.tensor([-torch.sin(phi), torch.cos(phi), 0], dtype=v.dtype, device=v.device)
        # Compute w[2] orthogonal to both direction and w[1]
        w2 = torch.cross(direction, w1, dim=0)
        # Transform the vector to the new frame
        w = torch.stack([torch.dot(v, direction), torch.dot(v, w1), torch.dot(v, w2)])
        return w
    else:
        raise TypeError("Input vector must be a numpy array or torch tensor")