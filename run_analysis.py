from typing import List

import yaml
from components.analysis import Analysis
from configs import configs
from configs.configs import ExperimentConfig as config
import sys
import os

# ==================================================================================

def get_config_values(key) -> dict:
    return [getattr(config, key).__dict__[subkey] for subkey in getattr(config, key).__dict__.keys() if subkey[0] != "_"]

def get_config_keys(key) -> dict:
    return [subkey for subkey in getattr(config, key).__dict__.keys() if subkey[0] != "_"]

def create_analysis(name, num_runs, base_env_config, base_run_config, variations, custom_config):
    return Analysis({
        "name":            name,
        "num_runs":        num_runs,
        "base_env_config": base_env_config,
        "base_run_config": base_run_config,
        "record_videos":   False,
        "variations":      variations,
        "wandb":           True,
        "custom_config":   custom_config,
    })

def create_analysis_from_custom_config(name: str, custom_config: dict):
    custom_same_for_all = custom_config["same_for_all"]
    custom_variations: List[dict] = custom_config["variations"]
    for var in custom_variations:
        var.update(custom_same_for_all)
    variations: List[dict] = [{key: (config.__dict__[key].__dict__[val] if key not in ["desired_distance", "start_distance"] else val) for key, val in var.items()} for var in custom_variations]
    return create_analysis(name, runs_per_variation, base_env_config, base_run_config, variations, custom_config)

def create_analysis_from_sys_arg(arg):
    if os.path.isfile(arg):
        try:
            with open(arg, "r") as config_file:
                custom_config = yaml.safe_load(config_file)
            analysis = create_analysis_from_custom_config(arg.split("/")[-1].split(".")[0], custom_config)
        except:
            print("No valid config path.")
            sys.exit(1)
    elif os.path.isdir(arg):
        try:
            analysis = Analysis.load(arg)
        except:
            print("No valid analysis path.")
            sys.exit(1)
    else:
        print("Invalid path. Please provide a valid file or directory path.")
        sys.exit(1)
    return analysis

# ==================================================================================

base_env_config = {
    "vel_control":          True,
    "sensor_angle_deg":     360,
    "num_obstacles":        0,
    "timestep":             0.05,
}
base_run_config = {
    "num_steps":        300,
    "seed":             1,
}
runs_per_variation = 5

# ==================================================================================

if __name__ == "__main__":
    def print_usage():
        print("Usage: python run_analysis.py requires two arguments: <action> <argument>")
        print("Action: run         | Argument: config path: str")
        print("Action: demo        | Argument: config path: str OR analysis_path: str -> useful for quick testing")
        print("Action: rerun       | Argument: analysis_path: str -> useful after changing model or environment code")
        print("Action: replot      | Argument: analysis_path: str -> useful after changing plotting config")

    if len(sys.argv) != 3:
        print_usage()
        sys.exit(1)
    else:
        if sys.argv[1]   == "run":
            with open(sys.argv[2], "r") as config_file:
                custom_config = yaml.safe_load(config_file)
            analysis = create_analysis_from_custom_config(sys.argv[2].split("/")[-1].split(".")[0], custom_config)
            analysis.run_analysis()
            analysis.plot_states_and_losses()
        elif sys.argv[1] == "demo":
            analysis = create_analysis_from_sys_arg(sys.argv[2])
            variations = analysis.variations
            variation_values = {key: [variation[key] for variation in variations] for key in config.keys}
            axes = {
                "_".join([analysis.get_key_from_value(vars(vars(configs)[key]), variation[key]) for key in config.keys if (key not in variation_values.keys() or len(variation_values[key])>1) and analysis.count_variations(variations, key)>1]): {
                    subkey: variation[subkey] for subkey in variation.keys()
                } for variation in analysis.variations if all([variation[key] in variation_values[key] for key in variation_values.keys()])
            }  
            try:
                keys = list(axes.keys())
                print(f"Available Variations:")
                for i, key in enumerate(keys):
                    print(f"{i+1}: {key}")
                key_index = int(input(f"Choose your variation (1-{len(axes.keys())}): ")) - 1
                analysis.run_demo(axes[keys[key_index]], 1, True)
            except:
                print("Error: Didn't work.")
        elif sys.argv[1] == "rerun":
            try:
                analysis = Analysis.load(sys.argv[2])
            except:
                print("No valid analysis path.")
                sys.exit(1)
            analysis.run_analysis()
            analysis.plot_states_and_losses()
        elif sys.argv[1] == "replot":
            config_path = sys.argv[2]
            with open(config_path, "r") as config_file:
                custom_config = yaml.safe_load(config_file)
            custom_name = config_path.split("/")[-1].split(".")[0]
            records_path = "./records"
            found_counter = 0
            for subfolder_name in os.listdir(records_path):
                if '_'.join(subfolder_name.split('_')[1:]) == custom_name:
                    found_counter += 1
                    analysis = Analysis.load(os.path.join(records_path, subfolder_name))
                    analysis.custom_config = custom_config
                    analysis.plot_states_and_losses()
                    analysis.experiment_config["custom_config"] = custom_config
                    analysis.save()
            else:
                print(f"Replotted {found_counter} analyses.")