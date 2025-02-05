import torch

def smooth_abs(x, margin=1.0):
    abs_x = torch.abs(x)
    smooth_part = 0.5 * (x**2) / margin
    linear_part = abs_x - 0.5 * margin
    return torch.where(abs_x <= margin, smooth_part, linear_part)

def get_foveal_noise(angle, index, fv_noise, sensor_angle):
    if index == 0:
        return (abs(angle) * fv_noise["target_offset_angle"]/(sensor_angle/2))
    else:
        return (abs(angle) * fv_noise["target_offset_angle_dot"]/(sensor_angle/2))