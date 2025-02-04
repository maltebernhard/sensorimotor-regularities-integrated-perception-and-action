# --------------------- sensor noise configs -------------------------

from typing import Dict


general_small_noise = {
    "target_offset_angle":      0.02,
    "target_offset_angle_dot":  0.02,
    "target_visual_angle":      0.01,
    "target_visual_angle_dot":  0.01,
    "vel_frontal":              0.02,
    "vel_lateral":              0.02,
    "vel_rot":                  0.01,
    "target_distance":          0.02,
}
general_large_noise = {
    "target_offset_angle":      0.1,
    "target_offset_angle_dot":  0.1,
    "target_visual_angle":      0.05,
    "target_visual_angle_dot":  0.05,
    "vel_frontal":              0.2,
    "vel_lateral":              0.2,
    "vel_rot":                  0.1,
    "target_distance":          0.2,
}
large_triang_noise = {
    "target_offset_angle":      0.1,
    "target_offset_angle_dot":  0.1,
    "target_visual_angle":      0.01,
    "target_visual_angle_dot":  0.01,
    "vel_frontal":              0.02,
    "vel_lateral":              0.02,
    "vel_rot":                  0.01,
    "target_distance":          0.2,
}
large_divergence_noise = {
    "target_offset_angle":      0.02,
    "target_offset_angle_dot":  0.02,
    "target_visual_angle":      0.05,
    "target_visual_angle_dot":  0.05,
    "vel_frontal":              0.02,
    "vel_lateral":              0.02,
    "vel_rot":                  0.01,
    "target_distance":          0.2,
}
large_distance_noise = {
    "target_offset_angle":      0.02,
    "target_offset_angle_dot":  0.02,
    "target_visual_angle":      0.01,
    "target_visual_angle_dot":  0.01,
    "vel_frontal":              0.02,
    "vel_lateral":              0.02,
    "vel_rot":                  0.01,
    "target_distance":          0.2,
}
huge_distance_noise = {
    "target_offset_angle":      0.02,
    "target_offset_angle_dot":  0.02,
    "target_visual_angle":      0.01,
    "target_visual_angle_dot":  0.01,
    "vel_frontal":              0.02,
    "vel_lateral":              0.02,
    "vel_rot":                  0.01,
    "target_distance":          0.5,
}
noise_dict = {
    "SmallNoise":    general_small_noise,
    "LargeNoise":    general_large_noise,
    "TriNoise":      large_triang_noise,
    "DivNoise":      large_divergence_noise,
    "DistNoise":     large_distance_noise,
    "HugeDistNoise": huge_distance_noise,
}

# --------------------- observation loss configs ---------------------

# for each observation key, deactivate sensor readings in given ranges of seconds
double_loss = {
    "target_offset_angle": [
        (5.0, 6.0),
        (10.0, 11.0),
    ],
    "target_offset_angle_dot": [
        (5.0, 6.0),
        (10.0, 11.0),
    ],
},

# --------------------- foveal vision noise configs ---------------------

fv_noise = {
    "target_offset_angle":      0.3,
    "target_offset_angle_dot":  0.3,
    "target_visual_angle":      0.3,
    "target_visual_angle_dot":  0.3,
    "target_distance":          0.3,
}

# ---------------------- collection with keys -----------------------

config_dicts: Dict[str,dict] = {
    "smcs": {
        "None": [],
        "Both": ["Divergence", "Triangulation"],
        "Tri":  ["Triangulation"],
        "Div":  ["Divergence"],
    },
    "control": {
        "AICON": False,
        "CONTROL": True,
    },
    "sensor_noise": {
        "SmallNoise":    general_small_noise,
        "LargeNoise":    general_large_noise,
        "TriNoise":      large_triang_noise,
        "DivNoise":      large_divergence_noise,
        "DistNoise":     large_distance_noise,
        "HugeDistNoise": huge_distance_noise,
    },
    "foveal_vision_noise": {
        "FVNoise": fv_noise,
        "NoFVNoise": {},
    },
    "moving_target": {
        "stationary": "false",
        "linear":     "linear",
        "sine":       "sine",
        "flight":     "flight",
    },
    "observation_loss": {
        "NoObsLoss": {},
    },
}