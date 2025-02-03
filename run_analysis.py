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

# --------------------- moving target types -------------------------

# moving_target variations:
# "false"  - target is stationary
# "linear" - target moves linearly
# "sine"   - target moves in a sine wave
# "flight" - target moves in a flight pattern

# --------------------- observation noise configs ---------------------

general_small_noise = {
    "target_offset_angle":      0.01,
    "target_offset_angle_dot":  0.01,
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
    "target_visual_angle":      0.1,
    "target_visual_angle_dot":  0.1,
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
    "target_offset_angle":      0.01,
    "target_offset_angle_dot":  0.01,
    "target_visual_angle":      0.1,
    "target_visual_angle_dot":  0.1,
    "vel_frontal":              0.02,
    "vel_lateral":              0.02,
    "vel_rot":                  0.01,
    "target_distance":          0.2,
}
large_distance_noise = {
    "target_offset_angle":      0.01,
    "target_offset_angle_dot":  0.01,
    "target_visual_angle":      0.01,
    "target_visual_angle_dot":  0.01,
    "vel_frontal":              0.02,
    "vel_lateral":              0.02,
    "vel_rot":                  0.01,
    "target_distance":          0.2,
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
    "target_visual_angle":      0.1,
    "target_visual_angle_dot":  0.1,
    "target_distance":          0.1,
}
fv_noise = {
    "target_offset_angle":      0.3,
    "target_offset_angle_dot":  0.3,
    "target_visual_angle":      0.3,
    "target_visual_angle_dot":  0.3,
    "target_distance":          0.3,
}

large_fv_noise = {
    "target_offset_angle":      0.5,
    "target_offset_angle_dot":  0.5,
    "target_visual_angle":      0.5,
    "target_visual_angle_dot":  0.5,
    "target_distance":          0.5,
}

# ==================================================================================

if __name__ == "__main__":
    aicon_type_smcs = [[], ["Divergence"], ["Triangulation"], ["Divergence", "Triangulation"]]
    aicon_type_controls = [True, False]
    aicon_type_distance_sensors = [True, False]

    exp1_aicon_type_config = []
    exp2_aicon_type_config = []
    for smcs in aicon_type_smcs:
        for control in aicon_type_controls:
            for distance_sensor in aicon_type_distance_sensors:
                if not distance_sensor and not len(smcs)==0:
                    exp1_aicon_type_config.append({
                        "SMCs":           smcs,
                        "Control":        control,
                        "DistanceSensor": distance_sensor,
                    })
                if distance_sensor and not control:
                    exp2_aicon_type_config.append({
                        "SMCs":           smcs,
                        "Control":        control,
                        "DistanceSensor": distance_sensor,
                    })

    # --------------------- config ---------------------
    num_runs_per_config = 20
    model_type = "SMC"

    exp1_observation_noise_config = [general_small_noise, general_large_noise, large_triang_noise, large_divergence_noise]
    exp2_observation_noise_config = [general_small_noise, general_large_noise, large_triang_noise, large_divergence_noise, large_distance_noise]

    moving_target_config = ["false"]
    observation_loss_config = [{}]
    foveal_vision_noise_config = [{}, fv_noise]

    use_moving_target = False
    use_observation_noise = True
    use_observation_loss = True
    use_foveal_vision_noise = True

    # ------------------- plotting config -----------------------

    # plotting_config = {
    #     "name": "all_aicon_types",
    #     "states": {
    #         "PolarTargetPos": {
    #             "indices": [0,1],
    #             "labels" : ["Distance","Angle"],
    #             "ybounds": [
    #                 [(-5, 20), (-2, 2)    ],
    #                 [(0, 10),  (-0.1, 0.1)],
    #                 [(0, 4),   (0, 0.1)   ],
    #             ]
    #         },
    #     },
    #     "goals": {
    #         "PolarGoToTarget": {
    #             "ybounds": (0, 300)
    #         },
    #     },
    #     "axes": {
    #         "Baseline": {
    #             "aicon_type":           "Baseline",
    #             "target_movement":      moving_target_config[0],
    #             "sensor_noise":         observation_noise_config[0],
    #             "observation_loss":     observation_loss_config[0],
    #             "foveal_vision_noise":  foveal_vision_noise_config[0],
    #         },
    #         "Control": {
    #             "aicon_type":           "Control",
    #             "target_movement":      moving_target_config[0],
    #             "sensor_noise":         observation_noise_config[0],
    #             "observation_loss":     observation_loss_config[0],
    #             "foveal_vision_noise":  foveal_vision_noise_config[0],
    #         },
    #         "Goal": {
    #             "aicon_type":           "Goal",
    #             "target_movement":      moving_target_config[0],
    #             "sensor_noise":         observation_noise_config[0],
    #             "observation_loss":     observation_loss_config[0],
    #             "foveal_vision_noise":  foveal_vision_noise_config[0],
    #         },
    #         "FovealVision": {
    #             "aicon_type":           "FovealVision",
    #             "target_movement":      moving_target_config[0],
    #             "sensor_noise":         observation_noise_config[0],
    #             "observation_loss":     observation_loss_config[0],
    #             "foveal_vision_noise":  foveal_vision_noise_config[0],
    #         },
    #         "Interconnection": {
    #             "aicon_type":           "Interconnection",
    #             "target_movement":      moving_target_config[0],
    #             "sensor_noise":         observation_noise_config[0],
    #             "observation_loss":     observation_loss_config[0],
    #             "foveal_vision_noise":  foveal_vision_noise_config[0],
    #         },
    #     }
    # }

    # --------------------- run ---------------------

    exp1_analysis = Analysis({
        "name":                       "Experiment1",
        "num_runs":                   num_runs_per_config,
        "model_type":                 model_type,
        "base_env_config":            base_env_config,
        "base_run_config":            base_run_config,
        "aicon_type_config":          exp1_aicon_type_config,
        "moving_target_config":       moving_target_config,
        "sensor_noise_config":        exp1_observation_noise_config,
        "observation_loss_config":    observation_loss_config,
        "foveal_vision_noise_config": foveal_vision_noise_config,
        "record_videos":              False,
    })

    exp2_analysis = Analysis({
        "name":                       "Experiment2",
        "num_runs":                   num_runs_per_config,
        "model_type":                 model_type,
        "base_env_config":            base_env_config,
        "base_run_config":            base_run_config,
        "aicon_type_config":          exp2_aicon_type_config,
        "moving_target_config":       moving_target_config,
        "sensor_noise_config":        exp2_observation_noise_config,
        "observation_loss_config":    observation_loss_config,
        "foveal_vision_noise_config": foveal_vision_noise_config,
        "record_videos":              False,
    })

    exp1_analysis.run_analysis()
    exp2_analysis.run_analysis()
    
    #exp1_analysis.plot_states(plotting_config, save=True, show=False)
    #exp1_analysis.plot_goal_losses(plotting_config, save=True, show=False)
    
    # analysis = Analysis.load("records/2025_01_21_11_44")
    # analysis.plot_states(plotting_config, save=True, show=False)
    # analysis.plot_goal_losses(plotting_config, save=True, show=False)
    # analysis.plot_state_runs(plotting_config, "Control", save=True, show=False)

    # analysis.run_demo(
    #     "Control",
    #     observation_noise_config[0],
    #     moving_target_config[0],
    #     observation_loss_config[0],
    #     foveal_vision_noise_config[0],
    #     run_number = 6,
    #     record_video = True,
    # )