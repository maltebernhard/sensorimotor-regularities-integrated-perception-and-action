from components.analysis import Runner
from configs.configs import ExperimentConfig as config

# ========================================================================================================

# --------------------- config ---------------------

variation_config = {
    # Sensorimotor Regularities used by the model. Options:
    # - nosmrs: No sensorimotor regularities
    # - tri:    Triangulation
    # - div:    Divergence
    # - both:   Both triangulation and divergence
    "smrs": config.smrs.both,

    # Determines whether the agent has a distance sensor. Options:
    # - no_dist_sensor: No distance sensor
    # - dist_sensor:    Distance sensor enabled
    "distance_sensor": config.distance_sensor.no_dist_sensor,

    # Controller type used by the agent. Options:
    # - aicon: AICON's gradient-based control
    # - task:  Controller aimed only at fulfilling the task
    # - trc:   Controller aimed at fulfilling the task and reducing uncertainty (executes actions related to SMRs proportionally to extimation uncertainty)
    "controller": config.controller.aicon,

    # Configuration of the target's behavior. Options:
    # - stationary: Target remains stationary
    # - linear:     Target moves on a straight line
    # - sine:       Target moves along a sine wave
    # - flight:     Target flees from agent (always moves straight away from agent)
    # - chase:      Target chases the agent (always moves straight towards agent)
    "target_config": config.target_config.sine,

    # Configuration of obstacles in the environment. Options:
    # - no_obstacles:        No obstacles present
    # - stationary_obstacle: Single, stationary obstacle
    # - chase_obstacle:      Single obstacle chasing the agent
    "obstacles": config.obstacles.rapid_chase_obstacle,

    # Noise level in the agent's sensors. Options:
    # - small_noise: Small amount of noise
    # - large_noise: Large amount of noise
    # - neg_dist_o_noise: Negative 20% offset on distance measurement (only has an effect if distance sensor is enabled)
    # - pos_dist_o_noise: Positive 20% offset on distance measurement (only has an effect if distance sensor is enabled)
    "sensor_noise": config.sensor_noise.small_noise,

    # Sensor failure for specified observations and time ranges. Options:
    # - no_obs_loss: No observation loss
    # - dist_loss:   Temporary failure of distance measurements
    "observation_loss": config.observation_loss.no_obs_loss,

    # "Foveal Vision Noise" - increased measurement noise in the agent's peripheral vision.
    # If activated, the AICON model automatically models this into its expected measurement uncertainty. In effect, it turns
    # towards objects it wants to perceive with high certainty. Options:
    # - no_fv_noise: No foveal vision noise
    # - fv_noise: Small amount of foveal vision noise
    "fv_noise": config.fv_noise.no_fv_noise,

    # Desired distance the agent should maintain from the target.
    "desired_distance": 5,

    # Agent's initial offset from the desired distance.
    "start_distance": 10,

    # Systematic drift in the agent's velocity, simulating wind effects. Options:
    # - no_wind:     No wind effect
    # - tiny_wind:   Small wind effect
    # - light_wind:  Light wind effect
    # - strong_wind: Large wind effect
    "wind": config.wind.no_wind,

    # Determines whether the agent outputs velocity or acceleration. NOTE: "Wind" currently only works with acceleration control. Options:
    # - vel: Velocity control
    # - acc: Acceleration control
    "control": config.control.vel,
}

base_env_config = {
    "sensor_angle_deg":     360,          # Agent FOV in degrees. NOTE: The basic model does not support "searching" for targets outside the FOV, but can be extended to do so. For the standard casse, set to 360 degrees.
    "timestep":             0.05,         # Seconds per timestep. NOTE: Control gains need to be adjusted when changing this value.
}

run_config = {
    "num_steps":        500,              # number of timesteps to run
    "initial_action":   [0.0, 0.0, 0.0],  # initial action
    "seed":             1,                # random seed
    "render":           True,             # render the environment
    "prints":           0,                # (for debugging) if > 0: prints estimator values and errors, update steps and gradients every n steps
    "step_by_step":     False,            # (for debugging) wait for user to press enter before computing next step
}

# --------------------- run ---------------------

if __name__ == "__main__":
    runner = Runner(
        run_config=run_config,
        base_env_config=base_env_config,
        variation=variation_config,
        variation_id=1
    )
    video_path = None
    runner.video_record_path = video_path
    runner.run()