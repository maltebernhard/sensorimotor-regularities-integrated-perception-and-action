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
    
def spherical_to_cartesian_direction(phi, theta):
    """Convert spherical angles to a unit direction vector in Cartesian coordinates."""
    if not torch.is_tensor(phi):
        phi = torch.tensor(phi)
    if not torch.is_tensor(theta):
        theta = torch.tensor(theta)
    x = torch.sin(theta) * torch.cos(phi)
    y = torch.sin(theta) * torch.sin(phi)
    z = torch.cos(theta)
    return torch.stack([x, y, z])

def build_rotation_matrix(phi, theta):
    """Build rotation matrix from local frame to world frame."""
    # Local x-axis (forward direction)
    x_axis = spherical_to_cartesian_direction(phi, theta)
    # Local y-axis: perpendicular to x and horizontal (world z-up)
    horizontal = torch.stack([x_axis[0], x_axis[1], torch.tensor(0.0)])
    if torch.linalg.norm(horizontal) < 1e-8:
        # x-axis is vertical; choose arbitrary horizontal y-axis
        y_axis = torch.tensor([0.0, 1.0, 0.0])
    else:
        y_axis = torch.cross(torch.tensor([0.0, 0.0, 1.0]), x_axis)
        y_axis = y_axis / torch.linalg.norm(y_axis)
    
    # Local z-axis via right-hand rule
    z_axis = torch.cross(x_axis, y_axis)
    
    # Stack axes as columns: [x | y | z]
    R = torch.stack((x_axis, y_axis, z_axis), dim=1)
    return R

def rtf_to_world(v_rtf, phi, theta):
    """Transform a velocity vector from the local to the world frame."""
    R = build_rotation_matrix(phi, theta)
    return torch.matmul(R, v_rtf) if torch.is_tensor(v_rtf) else R @ v_rtf

def world_to_rtf(v_world, phi, theta):
    """Transform a velocity vector from the world to the local frame."""
    R = build_rotation_matrix(phi, theta)
    return torch.matmul(R.T, v_world) if torch.is_tensor(v_world) else R.T @ v_world  # Transpose is the inverse for rotation matrices

def world_to_rtf_numpy(v_world, phi, theta):
    """Transform a velocity vector from the world to the local frame (numpy version)."""
    R = build_rotation_matrix(phi, theta).cpu().numpy()
    return R.T @ v_world if isinstance(v_world, np.ndarray) else R.T @ v_world