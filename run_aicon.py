from components.analysis import Runner
from configs.configs import ExperimentConfig as config

# ========================================================================================================

if __name__ == "__main__":
    model_type = "SMC"
    aicon_type = 1

    # --------------------- config ---------------------

    variation_config = {
        "smcs":              config.smcs.both,
        "distance_sensor":   config.distance_sensor.dist_sensor,
        "controller":        config.controller.aicon,
        "moving_target":     config.moving_target.sine_target,
        "sensor_noise":      config.sensor_noise.small_noise,
        "observation_loss":  config.observation_loss.no_obs_loss,
        "fv_noise":          config.fv_noise.no_fv_noise,
        "desired_distance":  10,
        "wind":              config.wind.light_wind,
        "control":           config.control.acc,
    }

    base_env_config = {
        "vel_control":          False,
        "sensor_angle_deg":     360,
        "num_obstacles":        0,
        "timestep":             0.05,
    }

    run_config = {
        "num_steps":        500,
        "initial_action":   [0.1, 0.0, 0.0],
        "seed":             10,
        "render":           True,
        "prints":           1,
        "step_by_step":     True,
    }

    # --------------------- run ---------------------

    runner = Runner(
        run_config=run_config,
        base_env_config=base_env_config,
        variation=variation_config,
        variation_id=1
    )
    runner.run()