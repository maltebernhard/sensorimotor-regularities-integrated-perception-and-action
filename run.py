from components.analysis import Runner
from configs.configs import ExperimentConfig as config

# ========================================================================================================

# --------------------- config ---------------------

variation_config = {
    "smcs":              config.smcs.both,
    "distance_sensor":   config.distance_sensor.no_dist_sensor,
    "controller":        config.controller.trc,
    "target_config":     config.target_config.sine,
    "obstacles":         config.obstacles.no_obstacles,
    "sensor_noise":      config.sensor_noise.small_noise,
    "observation_loss":  config.observation_loss.no_obs_loss,
    "fv_noise":          config.fv_noise.fv_noise,
    "desired_distance":  5,
    "start_distance":    10,
    "wind":              config.wind.no_wind,
    "control":           config.control.vel,
}

base_env_config = {
    "sensor_angle_deg":     360,
    "timestep":             0.05,
}

run_config = {
    "num_steps":        200,
    "initial_action":   [0.1, 0.0, 0.0],
    "seed":             1,
    "render":           True,
    "prints":           1,
    "step_by_step":     False,
}

# --------------------- run ---------------------

if __name__ == "__main__":
    runner = Runner(
        run_config=run_config,
        base_env_config=base_env_config,
        variation=variation_config,
        variation_id=1
    )
    video_path = f"test_vid.mp4"
    runner.video_record_path = video_path
    runner.run()