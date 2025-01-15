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

    aicon_type_config = ["Control", "Goal", "FovealVision", "Interconnection"]
    #observation_noise_config = [small_noise, large_noise]
    observation_noise_config = [small_noise]
    moving_target_config = ["false"]
    observation_loss_config = [{}]

    use_moving_target = False
    use_observation_noise = True
    use_observation_loss = False

    experiment_config = {
        "num_runs":                10,
        "initial_action":          [0.0, 0.0, 0.0],
        "base_env_config":         base_env_config,
        "base_run_config":         base_run_config,
        "aicon_type_config":       aicon_type_config,
        "moving_target_config":    moving_target_config if use_moving_target else ["false"],
        "sensor_noise_config":     observation_noise_config if use_observation_noise else [{}],
        "observation_loss_config": observation_loss_config if use_observation_loss else [{}],
    }

    plotting_config = {
        "name": "all_aicon_types",
        "states": {
            "PolarTargetPos": {
                "indices": [0,1,2,3],
                "labels" : ["Distance","Angle","DistanceDot","AngleDot"],
                "ybounds": [
                    [(-5, 20), (-2, 2),     (-4, 4),     (-2, 2)],
                    [(0, 10), (-0.1, 0.1), (-0.1, 0.1), (-0.05, 0.05)],
                    [(0, 4),  (0, 0.1),    (-0.0, 0.5), (0, 0.05)],
                ]
            }
        },
        "axes": {
            "Control": {
                "aicon_type": aicon_type_config[0],
                "target_movement": moving_target_config[0],
                "sensor_noise": observation_noise_config[0],
                "observation_loss": observation_loss_config[0],
            },
            "Goal": {
                "aicon_type": aicon_type_config[1],
                "target_movement": moving_target_config[0],
                "sensor_noise": observation_noise_config[0],
                "observation_loss": observation_loss_config[0],
            },
            "FovealVision": {
                "aicon_type": aicon_type_config[2],
                "target_movement": moving_target_config[0],
                "sensor_noise": observation_noise_config[0],
                "observation_loss": observation_loss_config[0],
            },
            "Interconnection": {
                "aicon_type": aicon_type_config[3],
                "target_movement": moving_target_config[0],
                "sensor_noise": observation_noise_config[0],
                "observation_loss": observation_loss_config[0],
            },
        }
    }

    # --------------------- run ---------------------

    analysis = Analysis(experiment_config)
    analysis.run_analysis()
    analysis.plot_states(plotting_config, save=True, show=False)
    #analysis.plot_goal_losses(save=True, show=False)