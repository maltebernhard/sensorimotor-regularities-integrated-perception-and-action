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
    0: "Experimental",
    # ------ BEHAVIOR ------
    1: "Control",
    2: "Goal",
    3: "FovealVision",
    4: "Interconnection",
    5: "Estimator",
    # ---- ACTUAL SMCs ----
    6: "Divergence",
    7: "Visibility",
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

# --------------------- foveal vision noise configs ---------------------

small_fv_noise = {
    "target_offset_angle":      0.1,
    "target_offset_angle_dot":  0.1,
}
large_fv_noise = {
    "target_offset_angle":      0.5,
    "target_offset_angle_dot":  0.5,
}

# ==================================================================================

if __name__ == "__main__":
    # --------------------- config ---------------------
    num_runs_per_config = 10
    model_type = "Experiment1"
    aicon_type_config = ["Baseline", "Control", "Goal", "FovealVision", "Interconnection"]
    #observation_noise_config = [small_noise, large_noise]
    observation_noise_config = [small_noise]
    moving_target_config = ["false"]
    observation_loss_config = [{}]
    foveal_vision_noise_config = [large_fv_noise]

    use_moving_target = False
    use_observation_noise = True
    use_observation_loss = False
    use_foveal_vision_noise = True

    # ------------------- plotting config -----------------------

    plotting_config = {
        "name": "all_aicon_types",
        "states": {
            "PolarTargetPos": {
                "indices": [0,1,2,3],
                "labels" : ["Distance","Angle","DistanceDot","AngleDot"],
                "ybounds": [
                    [(-5, 20), (-2, 2),     (-4, 4),     (-2, 2)],
                    [(0, 10),  (-0.1, 0.1), (-0.1, 0.1), (-0.05, 0.05)],
                    [(0, 4),   (0, 0.1),    (-0.0, 0.5), (0, 0.05)],
                ]
            },
        },
        "goals": {
            "PolarGoToTarget": {
                "ybounds": (0, 300)
            },
        },
        "axes": {
            "Baseline": {
                "aicon_type":           "Baseline",
                "target_movement":      moving_target_config[0],
                "sensor_noise":         observation_noise_config[0],
                "observation_loss":     observation_loss_config[0],
                "foveal_vision_noise":  foveal_vision_noise_config[0],
            },
            "Control": {
                "aicon_type":           "Control",
                "target_movement":      moving_target_config[0],
                "sensor_noise":         observation_noise_config[0],
                "observation_loss":     observation_loss_config[0],
                "foveal_vision_noise":  foveal_vision_noise_config[0],
            },
            "Goal": {
                "aicon_type":           "Goal",
                "target_movement":      moving_target_config[0],
                "sensor_noise":         observation_noise_config[0],
                "observation_loss":     observation_loss_config[0],
                "foveal_vision_noise":  foveal_vision_noise_config[0],
            },
            "FovealVision": {
                "aicon_type":           "FovealVision",
                "target_movement":      moving_target_config[0],
                "sensor_noise":         observation_noise_config[0],
                "observation_loss":     observation_loss_config[0],
                "foveal_vision_noise":  foveal_vision_noise_config[0],
            },
            "Interconnection": {
                "aicon_type":           "Interconnection",
                "target_movement":      moving_target_config[0],
                "sensor_noise":         observation_noise_config[0],
                "observation_loss":     observation_loss_config[0],
                "foveal_vision_noise":  foveal_vision_noise_config[0],
            },
        }
    }

    # --------------------- run ---------------------

    analysis = Analysis({
         "num_runs":                  num_runs_per_config,
        "model_type":                 model_type,
        "base_env_config":            base_env_config,
        "base_run_config":            base_run_config,
        "aicon_type_config":          aicon_type_config,
        "moving_target_config":       moving_target_config if use_moving_target else ["false"],
        "sensor_noise_config":        observation_noise_config if use_observation_noise else [{}],
        "observation_loss_config":    observation_loss_config if use_observation_loss else [{}],
        "foveal_vision_noise_config": foveal_vision_noise_config if use_foveal_vision_noise else [{}],
    })
    analysis.run_analysis()

    # analysis = Analysis.load("records/2025_01_21_11_44")
    
    analysis.plot_states(plotting_config, save=True, show=False)
    analysis.plot_state_runs(plotting_config, "Control", save=True, show=False)
    analysis.plot_goal_losses(plotting_config, save=True, show=False)
    
    # analysis.run_demo(
    #     "Control",
    #     observation_noise_config[0],
    #     moving_target_config[0],
    #     observation_loss_config[0],
    #     foveal_vision_noise_config[0],
    #     run_number = 6
    # )