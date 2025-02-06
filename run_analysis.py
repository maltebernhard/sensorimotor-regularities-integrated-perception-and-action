from components.analysis import Analysis
from configs import ExperimentConfig as config
from plot import plot_states_and_losses

# ==================================================================================

def get_config_values(key) -> dict:
    return [getattr(config, key).__dict__[subkey] for subkey in getattr(config, key).__dict__.keys() if subkey[0] != "_"]

def get_config_keys(key) -> dict:
    return [subkey for subkey in getattr(config, key).__dict__.keys() if subkey[0] != "_"]

def create_aicon_type_configs(experiment_id, smcs_list=None):
    aicon_type_configs = []
    for smcs in get_config_keys("smcs") if smcs_list is None else smcs_list:
        for control in get_config_keys("control"):
            for distance_sensor in get_config_keys("distance_sensor"):
                if experiment_id == 1 and distance_sensor=="no_dist_sensor" and smcs!="nosmcs":
                    aicon_type_configs.append({
                        "smcs":            smcs,
                        "control":         control,
                        "distance_sensor": distance_sensor,
                    })
                elif experiment_id == 2 and distance_sensor=="dist_sensor" and control=="aicon":
                    aicon_type_configs.append({
                        "smcs":            smcs,
                        "control":         control,
                        "distance_sensor": distance_sensor,
                    })
    return aicon_type_configs

def get_sensor_noise_config(aicon_type_config, experiment_id):
    if experiment_id == 1:
        # noise_config = ["small_noise", "large_noise"]
        # if aicon_type_config["smcs"] == "both":
        #     noise_config += ["tri_noise", "div_noise"]
        noise_config = ["small_noise"]
    elif experiment_id == 2:
        noise_config = ["small_noise", "large_noise", "dist_noise", "huge_dist_noise"]
        if aicon_type_config["smcs"] == "both":
            noise_config += ["tri_noise", "div_noise"]
    return noise_config

def get_fv_noise_config(experiment_id):
    if experiment_id == 1:
        return ["no_fv_noise"]
    elif experiment_id == 2:
        return ["no_fv_noise"]

def get_moving_target_config(experiment_id):
    if experiment_id == 1:
        # return ["stationary_target"]
        return ["stationary_target", "sine_target"]
    elif experiment_id == 2:
        return ["stationary_target"]
    
def get_observation_loss_config(experiment_id):
    if experiment_id == 1:
        return ["no_obs_loss"]
    elif experiment_id == 2:
        return ["no_obs_loss"]

def create_variations(experiment_id):
    exp_configs = []
    # aicon_type_configs = create_aicon_type_configs(experiment_id)
    aicon_type_configs = create_aicon_type_configs(experiment_id, ['both'])
    observation_noise_configs = get_sensor_noise_config("both" if "both" in [conf["smcs"] for conf in aicon_type_configs] else "", experiment_id)
    fv_noise_configs = get_fv_noise_config(experiment_id)
    moving_target_configs = get_moving_target_config(experiment_id)
    observation_loss_configs = get_observation_loss_config(experiment_id)
    print(f"Creating experiment {experiment_id} variations for ...")
    print("AICON Types:")
    for aicon_config in aicon_type_configs:
        print(aicon_config)
    print("Sensor Noises:        ", observation_noise_configs)
    print("Foveal Vision Noises: ", fv_noise_configs)
    print("Moving Targets:       ", moving_target_configs)
    print("Observation Losses:   ", observation_loss_configs)
    for aicon_type_config in aicon_type_configs:
        for observation_noise_config in observation_noise_configs:
            for fv_noise_config in fv_noise_configs:
                for moving_target_config in moving_target_configs:
                    for observation_loss_config in observation_loss_configs:
                        exp_configs.append({
                            "smcs":                config.smcs.__dict__[aicon_type_config["smcs"]],
                            "control":             config.control.__dict__[aicon_type_config["control"]],
                            "distance_sensor":     config.distance_sensor.__dict__[aicon_type_config["distance_sensor"]],
                            "moving_target":       config.moving_target.__dict__[moving_target_config],
                            "sensor_noise":        config.sensor_noise.__dict__[observation_noise_config],
                            "observation_loss":    config.observation_loss.__dict__[observation_loss_config],
                            "fv_noise":            config.fv_noise.__dict__[fv_noise_config],
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

runs_per_variation = 20

if __name__ == "__main__":
    experiment_type = 1
    variations = create_variations(experiment_type)

    analysis = Analysis({
        "name":                       f"Experiment{experiment_type}",
        "num_runs":                   runs_per_variation,
        "model_type":                 "SMC",
        "base_env_config":            base_env_config,
        "base_run_config":            base_run_config,
        "record_videos":              False,
        "variations":                 variations,
        "wandb":                      True,
    })
    analysis.run_analysis()


    if experiment_type == 1:
        invariant_config = {
            "smcs": None,
            "fv_noise": None,
        }
    elif experiment_type == 2:
        invariant_config = {
            "sensor_noise": None,
            "fv_noise":     None,
        }
    plot_states_and_losses(analysis, invariant_config)