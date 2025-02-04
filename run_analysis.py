from components.analysis import Analysis
from configs import config_dicts

# ==================================================================================

def create_aicon_type_configs(experiment_id):
    aicon_type_smcs = [[], ["Divergence"], ["Triangulation"], ["Divergence", "Triangulation"]]
    #aicon_type_smcs = [["Divergence", "Triangulation"]]
    aicon_type_controls = [True, False]
    aicon_type_distance_sensors = [True, False]
    aicon_type_configs = []
    for smcs in aicon_type_smcs:
        for control in aicon_type_controls:
            for distance_sensor in aicon_type_distance_sensors:
                if experiment_id == 1 and not distance_sensor and not len(smcs)==0:
                    aicon_type_configs.append({
                        "smcs":            smcs,
                        "control":         control,
                        "distance_sensor": distance_sensor,
                    })
                elif experiment_id == 2 and distance_sensor and not control:
                    aicon_type_configs.append({
                        "smcs":           smcs,
                        "control":        control,
                        "distance_sensor": distance_sensor,
                    })
    return aicon_type_configs

def get_relevant_noise_keys(aicon_type_config, experiment_id):
    if experiment_id == 1:
        relevant_noise_keys = ["SmallNoise", "LargeNoise"]
        if len(aicon_type_config["smcs"]) == 2:
            relevant_noise_keys += ["TriNoise", "DivNoise"]
    elif experiment_id == 2:
        relevant_noise_keys = ["SmallNoise", "LargeNoise", "DistNoise", "HugeDistNoise"]
        if len(aicon_type_config["smcs"]) == 2:
            relevant_noise_keys += ["TriNoise", "DivNoise"]
    return relevant_noise_keys

def get_foveal_vision_noise_config(aicon_type_config, experiment_id):
    if experiment_id == 1:
        #return [config_dicts["foveal_vision_noise"]["NoFVNoise"], config_dicts["foveal_vision_noise"]["FVNoise"]]
        return [config_dicts["foveal_vision_noise"]["NoFVNoise"]]
    elif experiment_id == 2:
        #return [config_dicts["foveal_vision_noise"]["NoFVNoise"], config_dicts["foveal_vision_noise"]["FVNoise"]]
        return [config_dicts["foveal_vision_noise"]["NoFVNoise"]]

def get_moving_target_config(aicon_type_config, experiment_id):
    if experiment_id == 1:
        return [config_dicts["moving_target"]["stationary"]]
    elif experiment_id == 2:
        return [config_dicts["moving_target"]["stationary"]]
    
def get_observation_loss_config(aicon_type_config, experiment_id):
    if experiment_id == 1:
        return [config_dicts["observation_loss"]["NoObsLoss"]]
    elif experiment_id == 2:
        return [config_dicts["observation_loss"]["NoObsLoss"]]

def create_configs(experiment):
    exp_configs = []
    for aicon_type_config in create_aicon_type_configs(experiment):
        for observation_noise_config in [noise for noise_key, noise in config_dicts["sensor_noise"].items() if noise_key in get_relevant_noise_keys(aicon_type_config, experiment)]:
            for foveal_vision_noise_config in get_foveal_vision_noise_config(aicon_type_config, experiment):
                for moving_target_config in get_moving_target_config(aicon_type_config, experiment):
                    for observation_loss_config in get_observation_loss_config(aicon_type_config, experiment):
                        exp_configs.append({
                            "smcs":                aicon_type_config["smcs"],
                            "control":             aicon_type_config["control"],
                            "distance_sensor":     aicon_type_config["distance_sensor"],
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
    "num_steps":        200,
    "initial_action":   [0.0, 0.0, 0.0],
    "seed":             1,
}

if __name__ == "__main__":
    experiment_type = 2
    exp_analysis = Analysis({
        "name":                       f"Experiment{experiment_type}",
        "num_runs":                   20,
        "model_type":                 "SMC",
        "base_env_config":            base_env_config,
        "base_run_config":            base_run_config,
        "record_videos":              False,
        "variations":                 create_configs(experiment_type),
    })
    exp_analysis.run_analysis()
