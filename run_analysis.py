from components.analysis import Analysis, Runner
from configs import ExperimentConfig as config
from plot import plot_states_and_losses
import sys
from pprint import pprint

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

def get_sensor_noise_config(smcs_config, experiment_id):
    if experiment_id == 1:
        # noise_config = ["small_noise", "large_noise"]
        # if smcs_config == "both":
        #     noise_config += ["tri_noise", "div_noise"]
        noise_config = ["small_noise"]
    elif experiment_id == 2:
        # noise_config = ["small_noise", "large_noise", "dist_noise", "huge_dist_noise"]
        # if smcs_config == "both":
        #     noise_config += ["tri_noise", "div_noise"]
        noise_config = ["small_noise", "huge_dist_noise"]
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
        #return ["stationary_target", "sine_target"]
        return ["sine_target"]
    
def get_observation_loss_config(smcs_config, experiment_id):
    if experiment_id == 1:
        return ["no_obs_loss"]
    elif experiment_id == 2:
        # loss_config =["no_obs_loss", "dist_loss"]
        # if smcs_config == "both" or smcs_config == "div":
        #     loss_config += ["div_loss"]
        # if smcs_config == "both" or smcs_config == "tri":
        #     loss_config += ["tri_loss"]
        # return loss_config
        # TODO: for dist loss, only compute small noise | for no loss, compare small and huge_dist noise
        return ["no_obs_loss"]
        #return ["dist_loss"]

def create_variations(experiment_id):
    exp_configs = []
    exp_config_keys = []
    # TODO: remove for ablations
    #for aicon_type_config in create_aicon_type_configs(experiment_id):
    for aicon_type_config in create_aicon_type_configs(experiment_id, ['both', 'nosmcs']):
        for observation_noise_config in get_sensor_noise_config(aicon_type_config["smcs"], experiment_id):
            for fv_noise_config in get_fv_noise_config(experiment_id):
                for moving_target_config in get_moving_target_config(experiment_id):
                    for observation_loss_config in get_observation_loss_config(aicon_type_config["smcs"], experiment_id):
                        exp_config_keys.append({
                            "smcs":                aicon_type_config["smcs"],
                            "control":             aicon_type_config["control"],
                            "distance_sensor":     aicon_type_config["distance_sensor"],
                            "moving_target":       moving_target_config,
                            "sensor_noise":        observation_noise_config,
                            "observation_loss":    observation_loss_config,
                            "fv_noise":            fv_noise_config,
                        })
                        exp_configs.append({
                            "smcs":                config.smcs.__dict__[aicon_type_config["smcs"]],
                            "control":             config.control.__dict__[aicon_type_config["control"]],
                            "distance_sensor":     config.distance_sensor.__dict__[aicon_type_config["distance_sensor"]],
                            "moving_target":       config.moving_target.__dict__[moving_target_config],
                            "sensor_noise":        config.sensor_noise.__dict__[observation_noise_config],
                            "observation_loss":    config.observation_loss.__dict__[observation_loss_config],
                            "fv_noise":            config.fv_noise.__dict__[fv_noise_config],
                        })
    print(f"================ all variations =================")
    for variation in exp_config_keys:
        pprint(variation)
    print("=================================================")
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

runs_per_variation = 10

if __name__ == "__main__":
    if len(sys.argv) != 2:
        print("Usage: python run_analysis.py <experiment_type>")
        sys.exit(1)
    try:
        experiment_type = int(sys.argv[1])
        analysis = None
    except ValueError:
        if "records/" in sys.argv[1]:
            experiment_type = 1 if "Experiment1" in sys.argv[1] else 2
            analysis = Analysis.load(sys.argv[1])
        else:
            print("Experiment type must be an integer.")
            sys.exit(1)
    
    variations = create_variations(experiment_type)
    if analysis is not None:
        analysis.add_and_run_variations(variations)
    else:
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

    # NOTE: use this to run a single demo with rendering and prints, for a specific variation, e.g. to check bugs
    # analysis.run_demo(variations[0], 1, True)
    # raise ValueError("Demo run complete")

    if experiment_type == 1:
        invariant_config = {
            "smcs": None,
            #"fv_noise": None,
        }
    elif experiment_type == 2:
        invariant_config = {
            #"sensor_noise":     None,
            #"fv_noise":         None,
            "moving_target":    None,
            "observation_loss": None,
        }
    plot_states_and_losses(analysis, invariant_config)