# --------------------- sensor noise configs -------------------------

general_small_noise = {
    "target_offset_angle":      0.02,
    "target_offset_angle_dot":  0.02,
    "target_visual_angle":      0.02,
    "target_visual_angle_dot":  0.02,
    "vel_frontal":              0.02,
    "vel_lateral":              0.02,
    "vel_rot":                  0.02,
    "target_distance":          0.02,
    "target_distance_dot":      0.02,
}
general_large_noise = {
    "target_offset_angle":      0.1,
    "target_offset_angle_dot":  0.1,
    "target_visual_angle":      0.1,
    "target_visual_angle_dot":  0.1,
    "vel_frontal":              0.1,
    "vel_lateral":              0.1,
    "vel_rot":                  0.1,
    "target_distance":          0.1,
    "target_distance_dot":      0.1,
}
large_triang_noise = {
    "target_offset_angle":      0.1,
    "target_offset_angle_dot":  0.1,
    "target_visual_angle":      0.02,
    "target_visual_angle_dot":  0.02,
    "vel_frontal":              0.02,
    "vel_lateral":              0.02,
    "vel_rot":                  0.02,
    "target_distance":          0.02,
    "target_distance_dot":      0.02,
}
large_divergence_noise = {
    "target_offset_angle":      0.02,
    "target_offset_angle_dot":  0.02,
    "target_visual_angle":      0.1,
    "target_visual_angle_dot":  0.1,
    "vel_frontal":              0.02,
    "vel_lateral":              0.02,
    "vel_rot":                  0.02,
    "target_distance":          0.02,
    "target_distance_dot":      0.02,
}
large_distance_noise = {
    "target_offset_angle":      0.02,
    "target_offset_angle_dot":  0.02,
    "target_visual_angle":      0.02,
    "target_visual_angle_dot":  0.02,
    "vel_frontal":              0.02,
    "vel_lateral":              0.02,
    "vel_rot":                  0.02,
    "target_distance":          0.1,
    "target_distance_dot":      0.1,
}
huge_distance_noise = {
    "target_offset_angle":      0.02,
    "target_offset_angle_dot":  0.02,
    "target_visual_angle":      0.02,
    "target_visual_angle_dot":  0.02,
    "vel_frontal":              0.02,
    "vel_lateral":              0.02,
    "vel_rot":                  0.02,
    "target_distance":          0.2,
    "target_distance_dot":      0.2,
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
tri_loss = {
    "target_offset_angle": [
        (100, 200),
    ],
    "target_offset_angle_dot": [
        (100, 200),
    ],
}
div_loss = {
    "target_visual_angle": [
        (100, 200),
    ],
    "target_visual_angle_dot": [
        (100, 200),
    ],
}
dist_loss = {
    "target_distance": [
        (100, 200),
    ],
    "target_distance_dot": [
        (100, 200),
    ],
}
no_obs_loss = {}

# --------------------- foveal vision noise configs ---------------------

fv_noise = {
    "target_offset_angle":      0.3,
    "target_offset_angle_dot":  0.3,
    "target_visual_angle":      0.3,
    "target_visual_angle_dot":  0.3,
    "target_distance":          0.3,
}

# ---------------------- collection with keys -----------------------

class SMCConfig:
    nosmcs = []
    both = ["Divergence", "Triangulation"]
    tri = ["Triangulation"]
    div = ["Divergence"]

class ControlConfig:
    aicon   = "aicon"
    manual = "manual"

class DistanceSensorConfig:
    dist_sensor = "distsensor"
    no_dist_sensor = "nodistsensor"

class SensorNoiseConfig:
    small_noise = general_small_noise
    large_noise = general_large_noise
    tri_noise = large_triang_noise
    div_noise = large_divergence_noise
    dist_noise = large_distance_noise
    huge_dist_noise = huge_distance_noise

class FovealVisionNoiseConfig:
    fv_noise = fv_noise
    no_fv_noise = {}

class MovingTargetConfig:
    stationary_target = "stationary"
    linear_target = "linear"
    sine_target = "sine"
    flight_target = "flight"

class ObservationLossConfig:
    no_obs_loss = no_obs_loss
    tri_loss = tri_loss
    div_loss = div_loss
    dist_loss = dist_loss

class ExperimentConfig:
    keys = ["smcs", "control", "distance_sensor", "sensor_noise", "fv_noise", "moving_target", "observation_loss"]
    smcs = SMCConfig
    control = ControlConfig
    distance_sensor = DistanceSensorConfig
    sensor_noise = SensorNoiseConfig
    control = ControlConfig
    fv_noise = FovealVisionNoiseConfig
    moving_target = MovingTargetConfig
    observation_loss = ObservationLossConfig
