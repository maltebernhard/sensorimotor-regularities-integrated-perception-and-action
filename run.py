from components.analysis import Analysis

# ========================================================================================================

if __name__ == "__main__":

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

    aicon_type = 8

    # --------------------- config ---------------------

    observation_noise = {
        "target_offset_angle":      0.01,
        "target_offset_angle_dot":  0.01,
        "target_visual_angle":      0.01,
        "target_visual_angle_dot":  0.01,
        "vel_frontal":              0.1,
        "vel_lateral":              0.1,
        "vel_rot":                  0.01
    }

    use_observation_noise = True

    env_config = {
        "vel_control":          True,
        "moving_target":        True,
        "sensor_angle_deg":     360,
        "num_obstacles":        0,
        "timestep":             0.05,
        "observation_noise":    observation_noise if use_observation_noise else {},
    }

    experiment_config = {
        "num_runs":         1,
        "num_steps":        2000,
        "initial_action":   [0.0, 0.0, 0.0],
        "seed":             1,
        #"seed":             60,
        "render":           True,
        "prints":           1,
        "step_by_step":     False,
        "record_data":      False,
    }

    # --------------------- run ---------------------

    analysis = Analysis(type=aicon_types[aicon_type], env_config=env_config, experiment_config=experiment_config)
    analysis.run()