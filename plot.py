from components.analysis import Analysis

# ========================================================================================================

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

    exp1_observation_noise_config = [general_small_noise, general_large_noise, large_triang_noise, large_divergence_noise]
    exp2_observation_noise_config = [general_small_noise, general_large_noise, large_triang_noise, large_divergence_noise, large_distance_noise]
    foveal_vision_noise_config = [{}, fv_noise]

    moving_target_config = ["false"]
    observation_loss_config = [{}]

# ========================================================================================================

plotting_config1 = {
    "name": "without_interconnection",
    "states": {
        "PolarTargetPos": {
            "indices": [0,1],
            "labels" : ["Distance","Angle"],
            "ybounds": [
                [(-1, 20),  (-0.5, 0.5), ],
                [(0, 10),  (-0.1, 0.1), ],
                [(0, 4),   (0, 0.1),    ],
            ]
        },
    },
    "goals": {
        "PolarGoToTarget": {
            "ybounds": (0, 20)
        },
    },
    "runs": [7, 17],
    "axes": {                                                       # ====================================== SMCs vs. Control ======================================
        "Test": {
            "aicon_type": {
                "SMCs":           ["Divergence", "Triangulation"],  # [["Divergence"], ["Triangulation"], ["Divergence", "Triangulation"]]
                "Control":        False,                            # [True, False]
                "DistanceSensor": False,                            # CONSTANT
            },
            "sensor_noise":         general_small_noise,            # [general_small_noise, general_large_noise, large_triang_noise, large_divergence_noise]
            "foveal_vision_noise":  {},                             # [{}, fv_noise]
            # constant
            "target_movement":      moving_target_config[0],
            "observation_loss":     observation_loss_config[0],
        },
    }
}

analysis1 = Analysis.load("records/2025_02_03_01_00_Experiment1")
analysis1.run_demo(plotting_config1["axes"]["Test"], run_number=17, record_video=False)

# analysis1.plot_states(plotting_config1, save=True, show=False)
# analysis1.plot_goal_losses(plotting_config1, save=True, show=False)
# analysis1.plot_state_runs(plotting_config1, "Test", save=True, show=False)