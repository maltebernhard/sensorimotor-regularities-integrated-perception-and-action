from components.analysis import Runner

# ========================================================================================================

if __name__ == "__main__":

    model_types = {
        # ------ SANDBOX ------
        "GlobalVelEstimation": {},
        "Base": {},
        # ------ Experiments ------
        "Experiment1": {
            1: "Baseline",
            2: "Control",
            3: "Goal",
            4: "FovealVision",
            5: "Interconnection",
        },
        "ExperimentFovealVision": {},
        # 6: "Estimator",
        # # ---- ACTUAL SMCs ----
        # 7: "Divergence",
        # 8: "Visibility",
    }

    model_type = "Base"
    aicon_type = 2

    # --------------------- config ---------------------

    observation_noise = {
        "target_offset_angle":      0.01,
        "target_offset_angle_dot":  0.01,
        "obstacle1_offset_angle":      0.01,
        "obstacle1_offset_angle_dot":  0.01,
        "target_visual_angle":      0.01,
        "target_visual_angle_dot":  0.01,
        "target_distance":          1.0,
        "vel_frontal":              0.01,
        "vel_lateral":              0.01,
        "vel_rot":                  0.001
    }
    use_observation_noise = True

    observation_loss = {
        "target_offset_angle":      (3.0, 5.0),
        "target_offset_angle_dot":  (3.0, 5.0),
    }
    use_observation_loss = False

    foveal_vision_noise = {
        "target_offset_angle":        0.5,
        "target_offset_angle_dot":    0.5,
        "obstacle1_offset_angle":     0.5,
        "obstacle1_offset_angle_dot": 0.5,
    }
    use_foveal_vision_noise = True

    env_config = {
        "vel_control":          True,
        "moving_target":        "false",        #"sine", "linear", "flight", "false"
        "sensor_angle_deg":     360,
        "num_obstacles":        0,
        "timestep":             0.05,
        "observation_noise":    observation_noise if use_observation_noise else {},
        "observation_loss":     observation_loss if use_observation_loss else {},
        "foveal_vision_noise":  foveal_vision_noise if use_foveal_vision_noise else {},
    }

    run_config = {
        "num_steps":        500,
        "initial_action":   [0.0, 0.0, 0.0],
        "seed":             1,
        "render":           True,
        "prints":           1,
        "step_by_step":     True,
    }

    # --------------------- run ---------------------

    runner = Runner(
        model=model_type,
        run_config=run_config,
        env_config=env_config,
        aicon_type=model_types[model_type][aicon_type] if len(model_types[model_type]) > 0 else None,
    )
    runner.run()