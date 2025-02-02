import torch

def smooth_abs(x, margin=1.0):
    abs_x = torch.abs(x)
    smooth_part = 0.5 * (x**2) / margin
    linear_part = abs_x - 0.5 * margin
    return torch.where(abs_x <= margin, smooth_part, linear_part)

def get_foveal_noise(angle, index, foveal_vision_noise, sensor_angle):
    return (abs(angle) * foveal_vision_noise[index]/(sensor_angle/2))