from typing import List

import yaml
from components.analysis import Analysis
from configs.configs import ExperimentConfig as config
from plot import create_axes, create_standard_plotting_states, plot_states_and_losses
import sys
from pprint import pprint
import os

# ==================================================================================

def get_config_values(key) -> dict:
    return [getattr(config, key).__dict__[subkey] for subkey in getattr(config, key).__dict__.keys() if subkey[0] != "_"]

def get_config_keys(key) -> dict:
    return [subkey for subkey in getattr(config, key).__dict__.keys() if subkey[0] != "_"]

def create_aicon_type_configs(experiment_id, smcs_list=None):
    aicon_type_configs = []
    for smcs in get_config_keys("smcs") if smcs_list is None else smcs_list:
        for controller in get_config_keys("controller"):
            for distance_sensor in get_config_keys("distance_sensor"):
                if experiment_id == 1 and distance_sensor=="no_dist_sensor" and smcs!="nosmcs":
                    aicon_type_configs.append({
                        "smcs":            smcs,
                        "controller":      controller,
                        "distance_sensor": distance_sensor,
                    })
                elif experiment_id == 2 and distance_sensor=="dist_sensor" and controller=="aicon":
                    aicon_type_configs.append({
                        "smcs":            smcs,
                        "controller":      controller,
                        "distance_sensor": distance_sensor,
                    })
    return aicon_type_configs

def get_sensor_noise_config(smcs_config, experiment_id):
    if experiment_id == 1:
        noise_config = ["small_noise", "large_noise"]
        # if smcs_config == "both" or smcs_config == "div":
        #     noise_config += ["div_noise"]
        # if smcs_config == "both" or smcs_config == "tri":
        #     noise_config += ["tri_noise"]
    elif experiment_id == 2:
        noise_config = ["small_noise", "large_noise", "huge_dist_noise", "dist_offset_noise"]
        # if smcs_config == "both":
        #     noise_config += ["tri_noise", "div_noise"]
        #noise_config = ["dist_offset_noise"]#, "huge_dist_noise"]
    return noise_config

def get_fv_noise_config(experiment_id):
    if experiment_id == 1:
        return ["no_fv_noise"]
    elif experiment_id == 2:
        return ["no_fv_noise"]

def get_moving_target_config(experiment_id):
    if experiment_id == 1:
        #return ["stationary_target", "sine_target"]
        return ["sine_target"]
    elif experiment_id == 2:
        #return ["stationary_target", "sine_target"]
        return ["sine_target"]
    
def get_observation_loss_config(smcs_config, experiment_id):
    if experiment_id == 1:
        loss_config = ["no_obs_loss"]
    elif experiment_id == 2:
        loss_config = ["no_obs_loss"]#, "dist_loss"]
        # if smcs_config == "both" or smcs_config == "div":
        #     loss_config += ["div_loss"]
        # if smcs_config == "both" or smcs_config == "tri":
        #     loss_config += ["tri_loss"]
        # TODO: for dist loss, only compute small noise | for no loss, compare small and huge_dist noise
    return loss_config

def create_variations(experiment_id):
    exp_configs = []
    exp_config_keys = []
    # TODO: remove for ablations
    for aicon_type_config in create_aicon_type_configs(experiment_id):
    #for aicon_type_config in create_aicon_type_configs(experiment_id, ['both', 'nosmcs']):
        for observation_noise_config in get_sensor_noise_config(aicon_type_config["smcs"], experiment_id):
            for fv_noise_config in get_fv_noise_config(experiment_id):
                for moving_target_config in get_moving_target_config(experiment_id):
                    for observation_loss_config in get_observation_loss_config(aicon_type_config["smcs"], experiment_id):
                        exp_config_keys.append({
                            "smcs":                aicon_type_config["smcs"],
                            "controller":          aicon_type_config["controller"],
                            "distance_sensor":     aicon_type_config["distance_sensor"],
                            "moving_target":       moving_target_config,
                            "sensor_noise":        observation_noise_config,
                            "observation_loss":    observation_loss_config,
                            "fv_noise":            fv_noise_config,
                            "desired_distance":    5,
                            "control":             "vel",
                        })
                        exp_configs.append({
                            "smcs":                config.smcs.__dict__[aicon_type_config["smcs"]],
                            "controller":          config.control.__dict__[aicon_type_config["controller"]],
                            "distance_sensor":     config.distance_sensor.__dict__[aicon_type_config["distance_sensor"]],
                            "moving_target":       config.moving_target.__dict__[moving_target_config],
                            "sensor_noise":        config.sensor_noise.__dict__[observation_noise_config],
                            "observation_loss":    config.observation_loss.__dict__[observation_loss_config],
                            "fv_noise":            config.fv_noise.__dict__[fv_noise_config],
                            "desired_distance":    5,
                            "control":             "vel",
                        })
    print(f"================ all variations =================")
    for variation in exp_config_keys:
        pprint(variation)
    print("=================================================")
    return exp_configs

def create_analysis(name, num_runs, base_env_config, base_run_config, variations):
    return Analysis({
        "name":                       name,
        "num_runs":                   num_runs,
        "model_type":                 "SMC",
        "base_env_config":            base_env_config,
        "base_run_config":            base_run_config,
        "record_videos":              False,
        "variations":                 variations,
        "wandb":                      True,
    })

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

# ==================================================================================

if __name__ == "__main__":
    def print_usage():
        print("Usage: python run_analysis.py requires two arguments: <action> <argument>")
        print("Action: run         | Argument: config path: str")
        print("Action: rerun       | Argument: analysis_path: str -> useful after changing model or environment code")
        print("Action: replot      | Argument: analysis_path: str -> useful after changing plotting config")
        print("Action: auto_run    | Argument: experiment_type: int [1,2]")
        print("Action: auto_replot | Argument: analysis_path: str")

    if len(sys.argv) != 3:
        print_usage()
        sys.exit(1)
    else:
        if sys.argv[1]   == "run":
            config_path = sys.argv[2]
            with open(config_path, "r") as config_file:
                custom_config = yaml.safe_load(config_file)
            custom_name = config_path.split("/")[-1].split(".")[0]
            custom_same_for_all = custom_config["same_for_all"]
            custom_variations: List[dict] = custom_config["variations"]
            for var in custom_variations:
                var.update(custom_same_for_all)
            variations: List[dict] = [{key: (config.__dict__[key].__dict__[val] if key!="desired_distance" else val) for key, val in var.items()} for var in custom_variations]
            custom_plotting_state_config = custom_config["plotting_state_config"]
            custom_plotting_style_config = custom_config["plotting_style_config"]
            analysis = create_analysis(custom_name, runs_per_variation, base_env_config, base_run_config, variations)
            analysis.run_analysis()
            plot_states_and_losses(analysis, plotting_states_config=custom_plotting_state_config, plot_styles=custom_plotting_style_config, plot_ax_runs=True)
        elif sys.argv[1] == "demo":
            try:
                analysis = Analysis.load(sys.argv[2])
            except:
                print("No valid analysis path.")
                sys.exit(1)
            variations = analysis.variations
            axes = create_axes(variations, {key: [variation[key] for variation in variations] for key in config.keys})
            print(f"Available Variations:")
            for i, key in enumerate(axes.keys()):
                print(f"{i+1}: {key}")
            try:
                key = int(input(f"Choose your variation (1-{len(axes.keys())}): "))
                analysis.run_demo(list(axes.values())[key], 1, True)
            except:
                print("Error: Didn't work.")
        elif sys.argv[1] == "rerun":
            try:
                analysis = Analysis.load(sys.argv[2])
            except:
                print("No valid analysis path.")
                sys.exit(1)
            analysis.run_analysis()
            plot_states_and_losses(analysis, plot_ax_runs=True)
        elif sys.argv[1] == "replot":
            config_path = sys.argv[2]
            with open(config_path, "r") as config_file:
                custom_config = yaml.safe_load(config_file)
            custom_name = config_path.split("/")[-1].split(".")[0]
            custom_plotting_state_config = custom_config["plotting_state_config"]
            custom_plotting_style_config = custom_config["plotting_style_config"]
            records_path = "./records"
            found_counter = 0
            for subfolder_name in os.listdir(records_path):
                if '_'.join(subfolder_name.split('_')[1:]) == custom_name:
                    found_counter += 1
                    analysis = Analysis.load(os.path.join(records_path, subfolder_name))
                    plot_states_and_losses(analysis, plotting_states_config=custom_plotting_state_config, plot_styles=custom_plotting_style_config, plot_ax_runs=True)
            else:
                print(f"Replotted {found_counter} analyses.")
        elif sys.argv[1] == "auto_replot":
            try:
                experiment_type = 1 if "Experiment1" in sys.argv[1] else 2
                analysis = Analysis.load(sys.argv[2])
            except:
                print("No valid analysis path.")
                sys.exit(1)
            plotting_states, invariant_config, plot_styles = create_standard_plotting_states(experiment_type)
            plot_states_and_losses(analysis, invariant_config, plotting_states, plot_styles=plot_styles)
        elif sys.argv[1] == "auto_run":
            try:
                experiment_type = int(sys.argv[2])
            except ValueError:
                print("Experiment type must be an integer in [1,2].")
                sys.exit(1)
            variations = create_variations(experiment_type)
            analysis = create_analysis(f"Experiment{experiment_type}", runs_per_variation, base_env_config, base_run_config, variations)
            analysis.run_analysis()
            plotting_states, invariant_config, plot_styles = create_standard_plotting_states(experiment_type)
            plot_states_and_losses(analysis, invariant_config, plotting_states, plot_styles=plot_styles, plot_ax_runs=True)
        else:
            print("Invalid argument.")
            print_usage()