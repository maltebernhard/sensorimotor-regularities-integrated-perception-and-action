from components.analysis import Analysis
from configs import ExperimentConfig as config
from plot import plot_states_and_losses

# ==================================================================================

def create_aicon_type_configs(experiment_id):
    aicon_type_configs = []
    for smcs in config.smcs.__dict__.values():
        for control in config.control.__dict__.values():
            for distance_sensor in config.control.__dict__.values():
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

def get_sensor_noise_config(aicon_type_config, experiment_id):
    if experiment_id == 1:
        noise_config = [config.sensor_noise.small_noise, config.sensor_noise.large_noise]
    elif experiment_id == 2:
        noise_config = [config.sensor_noise.small_noise, config.sensor_noise.large_noise, config.sensor_noise.dist_noise, config.sensor_noise.huge_dist_noise]
    if len(aicon_type_config["smcs"]) == 2:
        noise_config += [config.sensor_noise.tri_noise, config.sensor_noise.div_noise]
    return noise_config

def get_fv_noise_config(aicon_type_config, experiment_id):
    if experiment_id == 1:
        #return [cd.fv_noise.no_fv_noise, cd.fv_noise.fv_noise]
        return [config.fv_noise.no_fv_noise]
    elif experiment_id == 2:
        #return [cd.fv_noise.no_fv_noise, cd.fv_noise.fv_noise]
        return [config.fv_noise.no_fv_noise]

def get_moving_target_config(aicon_type_config, experiment_id):
    if experiment_id == 1:
        return [config.moving_target.stationary_target]
    elif experiment_id == 2:
        return [config.moving_target.stationary_target]
    
def get_observation_loss_config(aicon_type_config, experiment_id):
    if experiment_id == 1:
        return [config.observation_loss.no_obs_loss]
    elif experiment_id == 2:
        return [config.observation_loss.no_obs_loss]

def create_variations(experiment):
    exp_configs = []
    for aicon_type_config in create_aicon_type_configs(experiment):
        for observation_noise_config in get_sensor_noise_config(aicon_type_config, experiment):
            for fv_noise_config in get_fv_noise_config(aicon_type_config, experiment):
                for moving_target_config in get_moving_target_config(aicon_type_config, experiment):
                    for observation_loss_config in get_observation_loss_config(aicon_type_config, experiment):
                        exp_configs.append({
                            "smcs":                aicon_type_config["smcs"],
                            "control":             aicon_type_config["control"],
                            "distance_sensor":     aicon_type_config["distance_sensor"],
                            "moving_target":       moving_target_config,
                            "sensor_noise":        observation_noise_config,
                            "observation_loss":    observation_loss_config,
                            "fv_noise":            fv_noise_config,
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
    variations = create_variations(experiment_type)

    # custom test case
    variations = [{
        "smcs":                config.smcs.tri,
        "control":             config.control.aicon,
        "distance_sensor":     True,
        "moving_target":       config.moving_target.stationary_target,
        "sensor_noise":        config.sensor_noise.small_noise,
        "observation_loss":    config.observation_loss.no_obs_loss,
        "fv_noise":            config.fv_noise.no_fv_noise,
    }]

    analysis = Analysis({
        "name":                       f"Experiment{experiment_type}",
        "num_runs":                   20,
        "model_type":                 "SMC",
        "base_env_config":            base_env_config,
        "base_run_config":            base_run_config,
        "record_videos":              False,
        "variations":                 variations,
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