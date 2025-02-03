from components.analysis import Analysis

# ========================================================================================================

# --------------------- sensor noise configs -------------------------

general_small_noise = {
    "target_offset_angle":      0.01,
    "target_offset_angle_dot":  0.01,
    "target_visual_angle":      0.002,
    "target_visual_angle_dot":  0.002,
    "vel_frontal":              0.02,
    "vel_lateral":              0.02,
    "vel_rot":                  0.01,
    "target_distance":          0.02,
}
general_large_noise = {
    "target_offset_angle":      0.1,
    "target_offset_angle_dot":  0.1,
    "target_visual_angle":      0.02,
    "target_visual_angle_dot":  0.02,
    "vel_frontal":              0.2,
    "vel_lateral":              0.2,
    "vel_rot":                  0.1,
    "target_distance":          0.2,
}
large_triang_noise = {
    "target_offset_angle":      0.1,
    "target_offset_angle_dot":  0.1,
    "target_visual_angle":      0.002,
    "target_visual_angle_dot":  0.002,
    "vel_frontal":              0.02,
    "vel_lateral":              0.02,
    "vel_rot":                  0.01,
    "target_distance":          0.2,
}
large_divergence_noise = {
    "target_offset_angle":      0.01,
    "target_offset_angle_dot":  0.01,
    "target_visual_angle":      0.02,
    "target_visual_angle_dot":  0.02,
    "vel_frontal":              0.02,
    "vel_lateral":              0.02,
    "vel_rot":                  0.01,
    "target_distance":          0.2,
}
large_distance_noise = {
    "target_offset_angle":      0.01,
    "target_offset_angle_dot":  0.01,
    "target_visual_angle":      0.002,
    "target_visual_angle_dot":  0.002,
    "vel_frontal":              0.02,
    "vel_lateral":              0.02,
    "vel_rot":                  0.01,
    "target_distance":          0.2,
}
noise_dict = {
    "SmallNoise": general_small_noise,
    "LargeNoise": general_large_noise,
    "TriNoise": large_triang_noise,
    "DivNoise": large_divergence_noise,
    "DistNoise": large_distance_noise,
}

# --------------------- observation loss configs ---------------------

# for each observation key, deactivate sensor readings in given ranges of seconds
double_loss = {
    "target_offset_angle": [
        (5.0, 6.0),
        (10.0, 11.0),
    ],
    "target_offset_angle_dot": [
        (5.0, 6.0),
        (10.0, 11.0),
    ],
},

# --------------------- foveal vision noise configs ---------------------

fv_noise = {
    "target_offset_angle":      0.3,
    "target_offset_angle_dot":  0.3,
    "target_visual_angle":      0.3,
    "target_visual_angle_dot":  0.3,
    "target_distance":          0.3,
}

# ==================================================================================

def create_aicon_type_configs(experiment_id):
    #aicon_type_smcs = [[], ["Divergence"], ["Triangulation"], ["Divergence", "Triangulation"]]
    aicon_type_smcs = [["Divergence", "Triangulation"]]
    aicon_type_controls = [True, False]
    aicon_type_distance_sensors = [True, False]
    aicon_type_configs = []
    for smcs in aicon_type_smcs:
        for control in aicon_type_controls:
            for distance_sensor in aicon_type_distance_sensors:
                if experiment_id == 1 and not distance_sensor and not len(smcs)==0:
                    aicon_type_configs.append({
                        "SMCs":           smcs,
                        "Control":        control,
                        "DistanceSensor": distance_sensor,
                    })
                elif experiment_id == 2 and distance_sensor and not control:
                    aicon_type_configs.append({
                        "SMCs":           smcs,
                        "Control":        control,
                        "DistanceSensor": distance_sensor,
                    })
    return aicon_type_configs

def get_relevant_noise_keys(aicon_type_config, experiment_id):
    if experiment_id == 1:
        relevant_noise_keys = ["SmallNoise", "LargeNoise"]
        if aicon_type_config["SMCs"] == "Both":
            relevant_noise_keys += ["TriNoise", "DivNoise"]
    elif experiment_id == 2:
        relevant_noise_keys = ["SmallNoise", "LargeNoise", "DistNoise"]
        if aicon_type_config["SMCs"] == "Both":
            relevant_noise_keys += ["TriNoise", "DivNoise"]
    return relevant_noise_keys

def get_foveal_vision_noise_config(aicon_type_config, experiment_id):
    if experiment_id == 1:
        return [{}, fv_noise]
    elif experiment_id == 2:
        return [{}, fv_noise]

def get_moving_target_config(aicon_type_config, experiment_id):
    # "false"  - target is stationary
    # "linear" - target moves linearly
    # "sine"   - target moves in a sine wave
    # "flight" - target moves in a flight pattern
    if experiment_id == 1:
        return ["false"]
    elif experiment_id == 2:
        return ["false"]
    
def get_observation_loss_config(aicon_type_config, experiment_id):
    if experiment_id == 1:
        return [{}]
    elif experiment_id == 2:
        return [{}]

def create_configs(experiment):
    exp_configs = []
    for aicon_type_config in create_aicon_type_configs(experiment):
        for observation_noise_config in [noise for noise_key, noise in noise_dict.items() if noise_key in get_relevant_noise_keys(aicon_type_config, experiment)]:
            for foveal_vision_noise_config in get_foveal_vision_noise_config(aicon_type_config, experiment):
                for moving_target_config in get_moving_target_config(aicon_type_config, experiment):
                    for observation_loss_config in get_observation_loss_config(aicon_type_config, experiment):
                        exp_configs.append({
                            "aicon_type":          aicon_type_config,
                            "moving_target":       moving_target_config,
                            "sensor_noise":        observation_noise_config,
                            "observation_loss":    observation_loss_config,
                            "foveal_vision_noise": foveal_vision_noise_config,
                        })
    return exp_configs

# ==================================================================================

base_env_config = {
    "vel_control":          True,
    "sensor_angle_deg":     360,
    "num_obstacles":        0,
    "timestep":             0.05,
}

base_run_config = {
    "num_steps":        300,
    "initial_action":   [0.0, 0.0, 0.0],
    "seed":             1,
}

if __name__ == "__main__":
    exp1_analysis = Analysis({
        # general
        "name":                       "Experiment1",
        "num_runs":                   20,
        "model_type":                 "SMC",
        "base_env_config":            base_env_config,
        "base_run_config":            base_run_config,
        "record_videos":              False,
        "variations":                 create_configs(1),
    })

    exp2_analysis = Analysis({
        "name":                       "Experiment2",
        "num_runs":                   1,
        "model_type":                 "SMC",
        "base_env_config":            base_env_config,
        "base_run_config":            base_run_config,
        "record_videos":              False,
        "variations":                 create_configs(2),
    })

    exp1_analysis.run_analysis()
    exp2_analysis.run_analysis()