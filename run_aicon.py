from components.analysis import Runner
from configs import config_dicts

# ========================================================================================================

if __name__ == "__main__":

    model_types = {
        # ------ SANDBOX ------
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
        "SingleEstimator": {},
        "Divergence": {},
        "GlobalVel": {},
        "SMC": {
            1: {
                "smcs":            ["Triangulation"],   # ["Triangulation", "Divergence"],
                "distance_sensor": False,
                "control":         True,
            }
        }
        # 6: "Estimator",
        # # ---- ACTUAL SMCs ----
        # 7: "Divergence",
        # 8: "Visibility",
    }

    model_type = "SMC"
    aicon_type = 1

    # --------------------- config ---------------------

    observation_noise =       config_dicts["sensor_noise"]["SmallNoise"]
    observation_loss =        config_dicts["observation_loss"]["NoObsLoss"]
    foveal_vision_noise =     config_dicts["foveal_vision_noise"]["FVNoise"]
    use_observation_noise =   True
    use_observation_loss =    False
    use_foveal_vision_noise = True

    env_config = {
        "vel_control":          True,
        "moving_target":        "false",        #"false", "sine", "linear", "flight"
        "sensor_angle_deg":     360,
        "num_obstacles":        0,
        "timestep":             0.05,
        "observation_noise":    observation_noise if use_observation_noise else {},
        "observation_loss":     observation_loss if use_observation_loss else {},
        "foveal_vision_noise":  foveal_vision_noise if use_foveal_vision_noise else {},
    }

    run_config = {
        "num_steps":        500,
        "initial_action":   [0.1, 0.0, 0.0],
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