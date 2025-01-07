

from components.analysis import Analysis

# ========================================================================================================

if __name__ == "__main__":

    aicon_types = {
        0: "GeneralTest",
        1: "FovealVision",
        2: "Divergence",
        3: "Goal",
        #4: "Control",
        5: "Interconnection",
        6: "Estimator"
    }
    aicon_type = 2

    env_config = {
        "vel_control":      True,
        "moving_target":    True,
        "sensor_angle_deg": 360,
        "num_obstacles":    0,
        "timestep":         0.05
    }

    experiment_config = {
        "num_runs": 1,
        "num_steps": 2000,
        "initial_action": [0.0, 0.0, 0.0],
        "seed": 1,
        "render": True,
        "prints": 1,
        "step_by_step": True,
        "record_data": False,
    }

    analysis = Analysis(type=aicon_types[aicon_type], env_config=env_config, experiment_config=experiment_config)
    analysis.run()
    #analysis.plot()