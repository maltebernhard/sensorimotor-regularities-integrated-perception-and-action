from components.analysis import Analysis

# ========================================================================================================

base_env_config = {
    "vel_control":          True,
    "sensor_angle_deg":     360,
    "num_obstacles":        0,
    "timestep":             0.05,
}

base_run_config = {
    "num_steps":        300,
    "initial_action":   [0.0, 0.0, 0.0],
    "seed":             1,
}

# --------------------- aicon types -------------------------

aicon_types = {
    # ------ SANDBOX ------
    0: "Base",
    1: "GeneralTest",
    # ------ BEHAVIOR ------
    2: "Control",
    3: "Goal",
    4: "FovealVision",
    5: "Interconnection",
    6: "Estimator",
    # ---- ACTUAL SMCs ----
    7: "Divergence",
    8: "Visibility",
}

# --------------------- moving target types -------------------------

# moving_target variations:
# "false"  - target is stationary
# "linear" - target moves linearly
# "sine"   - target moves in a sine wave
# "flight" - target moves in a flight pattern

# --------------------- observation noise configs ---------------------

small_noise = {
    "target_offset_angle":      0.001,
    "target_offset_angle_dot":  0.001,
    "target_visual_angle":      0.01,
    "target_visual_angle_dot":  0.01,
    "vel_frontal":              0.02,
    "vel_lateral":              0.02,
    "vel_rot":                  0.01
}
large_noise = {
    "target_offset_angle":      0.01,
    "target_offset_angle_dot":  0.01,
    "target_visual_angle":      0.03,
    "target_visual_angle_dot":  0.03,
    "vel_frontal":              0.1,
    "vel_lateral":              0.1,
    "vel_rot":                  0.05
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

# ==================================================================================

if __name__ == "__main__":
    # --------------------- config ---------------------

    #aicon_type_config = [2, 3, 4, 5]
    aicon_type_config = [2]
    observation_noise_config = [small_noise, large_noise]
    #observation_noise_config = [small_noise]
    moving_target_config = ["false"]
    observation_loss_config = [{}]

    use_moving_target = False
    use_observation_noise = True
    use_observation_loss = False

    experiment_config = {
        "num_runs":                2,
        "num_steps":               10,
        "initial_action":          [0.0, 0.0, 0.0],
        "base_env_config":         base_env_config,
        "base_run_config":         base_run_config,
        "aicon_type_config":       [aicon_types[type_id] for type_id in aicon_type_config],
        "moving_target_config":    moving_target_config if use_moving_target else ["false"],
        "sensor_noise_config":     observation_noise_config if use_observation_noise else [{}],
        "observation_loss_config": observation_loss_config if use_observation_loss else [{}],
    }

    plotting_config = {
        "PolarTargetPos": [[0,1,2,3], ["Distance","Angle","DistanceDot","AngleDot"]],
    }

    # --------------------- run ---------------------

    analysis = Analysis(experiment_config)
    analysis.run_analysis()
    analysis.plot_states(plotting_config, save=True, show=False)
    analysis.plot_goal_losses(save=True, show=False)