# --------------------- sensor noise configs -------------------------

general_small_noise = {
    "offset_angle":      (0.0, 0.02),
    "offset_angle_dot":  (0.0, 0.02),
    "visual_angle":      (0.0, 0.02),
    "visual_angle_dot":  (0.0, 0.02),
    "vel_frontal":       (0.0, 0.02),
    "vel_lateral":       (0.0, 0.02),
    "vel_rot":           (0.0, 0.02),
    "distance":          (0.0, 0.02),
    "distance_dot":      (0.0, 0.02),
}
general_large_noise = {
    "offset_angle":      (0.0, 0.1),
    "offset_angle_dot":  (0.0, 0.1),
    "visual_angle":      (0.0, 0.1),
    "visual_angle_dot":  (0.0, 0.1),
    "vel_frontal":       (0.0, 0.1),
    "vel_lateral":       (0.0, 0.1),
    "vel_rot":           (0.0, 0.1),
    "distance":          (0.0, 0.1),
    "distance_dot":      (0.0, 0.1),
}
large_triang_noise = {
    "offset_angle":      (0.0, 0.1),
    "offset_angle_dot":  (0.0, 0.1),
    "visual_angle":      (0.0, 0.02),
    "visual_angle_dot":  (0.0, 0.02),
    "vel_frontal":       (0.0, 0.02),
    "vel_lateral":       (0.0, 0.02),
    "vel_rot":           (0.0, 0.02),
    "distance":          (0.0, 0.02),
    "distance_dot":      (0.0, 0.02),
}
large_divergence_noise = {
    "offset_angle":      (0.0, 0.02),
    "offset_angle_dot":  (0.0, 0.02),
    "visual_angle":      (0.0, 0.1),
    "visual_angle_dot":  (0.0, 0.1),
    "vel_frontal":       (0.0, 0.02),
    "vel_lateral":       (0.0, 0.02),
    "vel_rot":           (0.0, 0.02),
    "distance":          (0.0, 0.02),
    "distance_dot":      (0.0, 0.02),
}
large_distance_noise = {
    "offset_angle":      (0.0, 0.02),
    "offset_angle_dot":  (0.0, 0.02),
    "visual_angle":      (0.0, 0.02),
    "visual_angle_dot":  (0.0, 0.02),
    "vel_frontal":       (0.0, 0.02),
    "vel_lateral":       (0.0, 0.02),
    "vel_rot":           (0.0, 0.02),
    "distance":          (0.0, 0.1),
    "distance_dot":      (0.0, 0.1),
}
neg_distance_offset_noise = {
    "offset_angle":      (0.0, 0.02),
    "offset_angle_dot":  (0.0, 0.02),
    "visual_angle":      (0.0, 0.02),
    "visual_angle_dot":  (0.0, 0.02),
    "vel_frontal":       (0.0, 0.02),
    "vel_lateral":       (0.0, 0.02),
    "vel_rot":           (0.0, 0.02),
    "distance":          (-0.3, 0.1),
    "distance_dot":      (-0.3, 0.1),
}
huge_distance_noise = {
    "offset_angle":      (0.0, 0.02),
    "offset_angle_dot":  (0.0, 0.02),
    "visual_angle":      (0.0, 0.02),
    "visual_angle_dot":  (0.0, 0.02),
    "vel_frontal":       (0.0, 0.02),
    "vel_lateral":       (0.0, 0.02),
    "vel_rot":           (0.0, 0.02),
    "distance":          (0.0, 0.3),
    "distance_dot":      (0.0, 0.3),
}
pos_distance_offset_noise = {
    "offset_angle":      (0.0, 0.02),
    "offset_angle_dot":  (0.0, 0.02),
    "visual_angle":      (0.0, 0.02),
    "visual_angle_dot":  (0.0, 0.02),
    "vel_frontal":       (0.0, 0.02),
    "vel_lateral":       (0.0, 0.02),
    "vel_rot":           (0.0, 0.02),
    "distance":          (0.3, 0.1),
    "distance_dot":      (0.3, 0.1),
}

# --------------------- observation loss configs ---------------------

# for each observation key, deactivate sensor readings in given ranges of seconds
tri_loss = {
    "offset_angle": [
        (100, 200),
    ],
    "offset_angle_dot": [
        (100, 200),
    ],
}
div_loss = {
    "visual_angle": [
        (100, 200),
    ],
    "visual_angle_dot": [
        (100, 200),
    ],
}
dist_loss = {
    "distance": [
        (100, 200),
    ],
    "distance_dot": [
        (100, 200),
    ],
}
no_obs_loss = {}

# --------------------- foveal vision noise configs ---------------------

fv_noise = {
    "offset_angle":      (0.0, 0.5),
    "offset_angle_dot":  (0.0, 0.5),
    "visual_angle":      (0.0, 0.5),
    "visual_angle_dot":  (0.0, 0.5),
    "distance":          (0.0, 0.5),
}

# ---------------------- collection with keys -----------------------

class SMCConfig:
    nosmcs = []
    both   = ["Divergence", "Triangulation"]
    tri    = ["Triangulation"]
    div    = ["Divergence"]

class ControllerConfig:
    aicon  = "aicon"
    trc = "trc"
    task   = "task"

class DistanceSensorConfig:
    dist_sensor     = "distsensor"
    dist_dot_sensor = "distdotsensor"
    no_dist_sensor  = "nodistsensor"

class SensorNoiseConfig:
    small_noise       = general_small_noise
    large_noise       = general_large_noise
    tri_noise         = large_triang_noise
    div_noise         = large_divergence_noise
    dist_noise        = large_distance_noise
    neg_dist_o_noise  = neg_distance_offset_noise
    huge_dist_noise   = huge_distance_noise
    pos_dist_o_noise  = pos_distance_offset_noise

class FovealVisionNoiseConfig:
    fv_noise    = fv_noise
    no_fv_noise = {}

class TargetConfig:
    """
    tuple: (target_motion, target_vel (relative to max robot vel), target_radius)
    """
    stationary = ("stationary", 0.0, 1.0)
    linear     = ("linear",     0.5, 1.0)
    sine       = ("sine",       0.5, 1.0)
    flight     = ("flight",     0.5, 1.0)
    chase      = ("chase",      0.5, 1.0)

class ObstacleConfig:
    """
    list of tuples: (obstacle_motion, obstacle_vel (relative to max robot vel), obstacle_radius)
    """
    no_obstacles         = []
    stationary_obstacle  = [("stationary",  0.0, 3.0)]
    chase_obstacle       = [("chase",       0.5, 3.0)]
    rapid_chase_obstacle = [("chase",       0.8, 3.0)]

class ObservationLossConfig:
    no_obs_loss = no_obs_loss
    tri_loss    = tri_loss
    div_loss    = div_loss
    dist_loss   = dist_loss

class WindConfig:
    no_wind     = (0.0, 0.0)
    light_wind  = (0.2, 0.2)
    strong_wind = (0.5, 0.5)

class ControlConfig:
    acc = "acc"
    vel = "vel"

# ---------------------- collection with keys --------------------------------

class ExperimentConfig:
    keys = ["smcs", "control", "distance_sensor", "sensor_noise", "fv_noise", "target_config", "observation_loss", "controller"]
    smcs             = SMCConfig
    controller       = ControllerConfig
    distance_sensor  = DistanceSensorConfig
    sensor_noise     = SensorNoiseConfig
    fv_noise         = FovealVisionNoiseConfig
    target_config    = TargetConfig
    obstacles        = ObstacleConfig
    observation_loss = ObservationLossConfig
    wind             = WindConfig
    control          = ControlConfig
