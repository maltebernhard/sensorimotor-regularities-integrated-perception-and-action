from components.analysis import Runner
from configs import ExperimentConfig as config

# ========================================================================================================

if __name__ == "__main__":
    model_type = "SMC"
    aicon_type = 1

    # --------------------- config ---------------------

    smcs                    = config.smcs.both
    distance_sensor         = config.distance_sensor.no_dist_sensor
    control                 = config.control.aicon
    moving_target           = config.moving_target.stationary_target
    observation_noise       = config.sensor_noise.small_noise
    observation_loss        = config.observation_loss.no_obs_loss
    fv_noise                = config.fv_noise.no_fv_noise

    env_config = {
        "vel_control":          True,
        "moving_target":        moving_target,
        "sensor_angle_deg":     360,
        "num_obstacles":        0,
        "timestep":             0.05,
        "observation_noise":    observation_noise,
        "observation_loss":     observation_loss,
        "fv_noise":             fv_noise,
        "desired_distance":     10,
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
        model="SMC",
        run_config=run_config,
        base_env_config=env_config,
        aicon_type={
            "smcs": smcs,
            "distance_sensor": distance_sensor,
            "control": control,
        },
    )
    runner.run()