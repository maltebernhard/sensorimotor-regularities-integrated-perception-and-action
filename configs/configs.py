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
distance_offset_noise = {
    "offset_angle":      (0.0, 0.02),
    "offset_angle_dot":  (0.0, 0.02),
    "visual_angle":      (0.0, 0.02),
    "visual_angle_dot":  (0.0, 0.02),
    "vel_frontal":       (0.0, 0.02),
    "vel_lateral":       (0.0, 0.02),
    "vel_rot":           (0.0, 0.02),
    "distance":          (-0.2, 0.1),
    "distance_dot":      (-0.2, 0.1),
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
huge_distance_offset_noise = {
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
    manual = "manual"

class DistanceSensorConfig:
    dist_sensor    = "distsensor"
    no_dist_sensor = "nodistsensor"

class SensorNoiseConfig:
    small_noise       = general_small_noise
    large_noise       = general_large_noise
    tri_noise         = large_triang_noise
    div_noise         = large_divergence_noise
    dist_noise        = large_distance_noise
    dist_o_noise      = distance_offset_noise
    huge_dist_noise   = huge_distance_noise
    huge_dist_o_noise = huge_distance_offset_noise

class FovealVisionNoiseConfig:
    fv_noise    = fv_noise
    no_fv_noise = {}

class MovingTargetConfig:
    """
    first tuple value defines movement direction pattern, second defines fraction of robot max vel
    """
    stationary_target = ("stationary", 0.0)
    linear_target     = ("linear",     0.5)
    sine_target       = ("sine",       0.5)
    flight_target     = ("flight",     0.5)
    chase_target      = ("chase",      0.5)

class MovingObstacleConfig:
    stationary_obstacle  = ("stationary",  0.0)
    chase_obstacle       = ("chase",       0.5)
    rapid_chase_obstacle = ("chase",       0.8)

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
    keys = ["smcs", "control", "distance_sensor", "sensor_noise", "fv_noise", "moving_target", "observation_loss", "controller"]
    smcs             = SMCConfig
    controller       = ControllerConfig
    distance_sensor  = DistanceSensorConfig
    sensor_noise     = SensorNoiseConfig
    fv_noise         = FovealVisionNoiseConfig
    moving_target    = MovingTargetConfig
    moving_obstacles = MovingObstacleConfig
    observation_loss = ObservationLossConfig
    wind             = WindConfig
    control          = ControlConfig
