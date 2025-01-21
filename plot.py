from components.analysis import Analysis

# ========================================================================================================

analysis = Analysis.load("records/2025_01_21_18_09")

plotting_config = {
    "name": "without_interconnection",
    "states": {
        "PolarTargetPos": {
            "indices": [0,1],
            "labels" : ["Distance","Angle"],
            "ybounds": [
                [(-1, 2), (-0.5, 0.5), ],
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
    "axes": {
        "Baseline": {
            "aicon_type":           "Baseline",
            "target_movement":      analysis.experiment_config["moving_target_config"][0],
            "sensor_noise":         analysis.experiment_config["sensor_noise_config"][0],
            "observation_loss":     analysis.experiment_config["observation_loss_config"][0],
            "foveal_vision_noise":  analysis.experiment_config["foveal_vision_noise_config"][0],
        },
        "Control": {
            "aicon_type":           "Control",
            "target_movement":      analysis.experiment_config["moving_target_config"][0],
            "sensor_noise":         analysis.experiment_config["sensor_noise_config"][0],
            "observation_loss":     analysis.experiment_config["observation_loss_config"][0],
            "foveal_vision_noise":  analysis.experiment_config["foveal_vision_noise_config"][0],
        },
        "Goal": {
            "aicon_type":           "Goal",
            "target_movement":      analysis.experiment_config["moving_target_config"][0],
            "sensor_noise":         analysis.experiment_config["sensor_noise_config"][0],
            "observation_loss":     analysis.experiment_config["observation_loss_config"][0],
            "foveal_vision_noise":  analysis.experiment_config["foveal_vision_noise_config"][0],
        },
        "FovealVision": {
            "aicon_type":           "FovealVision",
            "target_movement":      analysis.experiment_config["moving_target_config"][0],
            "sensor_noise":         analysis.experiment_config["sensor_noise_config"][0],
            "observation_loss":     analysis.experiment_config["observation_loss_config"][0],
            "foveal_vision_noise":  analysis.experiment_config["foveal_vision_noise_config"][0],
        },
        # "Interconnection": {
        #     "aicon_type":           "Interconnection",
        #     "target_movement":      analysis.experiment_config["moving_target_config"][0],
        #     "sensor_noise":         analysis.experiment_config["sensor_noise_config"][0],
        #     "observation_loss":     analysis.experiment_config["observation_loss_config"][0],
        #     "foveal_vision_noise":  analysis.experiment_config["foveal_vision_noise_config"][0],
        # },
    }
}

analysis.plot_states(plotting_config, save=True, show=False)
analysis.plot_goal_losses(plotting_config, save=True, show=False)
analysis.plot_state_runs(plotting_config, "Control", save=True, show=False)