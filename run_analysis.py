import time
from typing import List

import yaml
from components.analysis import Analysis
from configs.configs import ExperimentConfig as config
import sys
import os

# ==================================================================================

def get_config_values(key) -> dict:
    return [getattr(config, key).__dict__[subkey] for subkey in getattr(config, key).__dict__.keys() if subkey[0] != "_"]

def get_config_keys(key) -> dict:
    return [subkey for subkey in getattr(config, key).__dict__.keys() if subkey[0] != "_"]

def create_analysis(name, base_env_config, variations, custom_config):
    return Analysis({
        "name":            name,
        "base_env_config": base_env_config,
        "record_videos":   False,
        "variations":      variations,
        "wandb":           False,               # set true to log run data to weights and biases
        "variation_config":   custom_config,       # custom config for plotting
    })

def create_analysis_from_experiment_config(name: str, experiment_config: dict):
    experiment_invariants = experiment_config["same_for_all"]
    experiment_variations: List[dict] = experiment_config["variations"]
    for var in experiment_variations:
        var.update(experiment_invariants)
    variations: List[dict] = [{key: (config.__dict__[key].__dict__[val] if key not in ["desired_distance", "start_distance"] else val) for key, val in var.items()} for var in experiment_variations]
    return create_analysis(name, base_env_config, variations, experiment_config)

def create_analysis_from_sys_arg(arg, demo=False):
    if os.path.isfile(arg):
        try:
            with open(arg, "r") as config_file:
                experiment_config: dict = yaml.safe_load(config_file)
            analysis = create_analysis_from_experiment_config(arg.split("/")[-1].split(".")[0], experiment_config)
        except Exception as e:
            print(e)
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
    "sensor_angle_deg":     360,
    "timestep":             0.05,
}

# ==================================================================================

if __name__ == "__main__":
    def print_usage():
        print("Usage: python run_analysis.py requires two arguments: <action> <argument>")
        print("Action: run         | Argument: config path: str")
        print("Action: demo        | Argument: config path: str OR analysis_path: str -> useful for quick testing")
        print("Action: rerun       | Argument: analysis_path: str -> useful after changing model or environment code")
        print("Action: replot      | Argument: analysis_path: str -> useful after changing plotting config")
        print("Action: run_all     | Argument: None -> runs all analyses in ./configs")
        print("Action: replot_all  | Argument: None -> replots all analyses in ./records")

    if len(sys.argv) != 3 and len(sys.argv) != 2:
        print_usage()
        sys.exit(1)
    else:
        if sys.argv[1]   == "run":
            with open(sys.argv[2], "r") as config_file:
                custom_config = yaml.safe_load(config_file)
            analysis = create_analysis_from_experiment_config(sys.argv[2].split("/")[-1].split(".")[0], custom_config)
            analysis.run_analysis()
            analysis.plot_states_and_losses()
        elif sys.argv[1] == "demo":
            analysis = create_analysis_from_sys_arg(sys.argv[2])
            variations = analysis.variations
            variation_values = {key: [variation[key] for variation in variations] for key in config.keys}
            axes = {
                "_".join([analysis.get_key_from_value(vars(vars(config)[key]), variation[key]) for key in config.keys if (key not in variation_values.keys() or len(variation_values[key])>1) and analysis.count_variations(variations, key)>1]): {
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
                if '_'.join(subfolder_name.split('_')[:-1]) == custom_name:
                    found_counter += 1
                    analysis = Analysis.load(os.path.join(records_path, subfolder_name))
                    analysis.variant_config = custom_config
                    analysis.plot_states_and_losses()
                    analysis.experiment_config["variation_config"].update(custom_config)
                    analysis.save()
            else:
                print(f"Replotted {found_counter} analyses.")
        elif sys.argv[1] == "run_all":
            analysis_configs: list[str] = [
                "exp1_extended.yaml",
                "exp2_distloss_extended.yaml",
                "exp2_non0mean_extended.yaml",
                "exp2_lightwind_extended.yaml",
            ]
            num_runs:  int = 10
            num_steps: int = 200

            for filename in analysis_configs:
                with open("./configs/" + filename, "r") as config_file:
                    experiment_config = yaml.safe_load(config_file)

                experiment_config.update({
                    "num_runs": num_runs,
                    "num_steps": min(num_steps, experiment_config["plotting_config"]["xbounds"][1]),
                })

                analysis = create_analysis_from_experiment_config(filename.split("/")[-1].split(".")[0], experiment_config)
                time.sleep(1)
                analysis.run_analysis()

            for filename in analysis_configs:
                config_path = f"./configs/{filename}"
                with open(config_path, "r") as config_file:
                    experiment_config = yaml.safe_load(config_file)
                custom_name = config_path.split("/")[-1].split(".")[0]
                records_path = "./records"
                found_counter = 0
                for subfolder_name in os.listdir(records_path):
                    if '_'.join(subfolder_name.split('_')[:-1]) == custom_name:
                        found_counter += 1
                        dirname = os.path.join(records_path, subfolder_name)
                        analysis = Analysis.load(dirname)
                        analysis.variant_config = experiment_config
                        analysis.plot_states_and_losses()
        elif sys.argv[1] == "replot_all":
            def delete_pdfs(dirname: str, namestring):
                if os.path.exists(dirname):
                    for filename in os.listdir(dirname):
                        if filename.endswith(".pdf") and namestring in filename:
                            os.remove(os.path.join(dirname, filename))
                            print(f"Deleted {dirname} {filename}")

            def delete_old_plots(dirname: str, which_plots):
                for plot in which_plots:
                    if plot == "time":
                        delete_pdfs(dirname+"/records/time/", "")
                    elif plot == "boxplots":
                        delete_pdfs(dirname+"/records/box/", "")
                    elif plot == "losses":
                        delete_pdfs(dirname+"/records/loss/", "main")
                    elif plot == "gradients":
                        delete_pdfs(dirname+"/records/loss/", "gradient_")
                    elif plot == "runs":
                        delete_pdfs(dirname+"/records/runs/", "")
                    elif plot == "collisions":
                        delete_pdfs(dirname+"/records/", "collisions")

            analysis_configs: list[str] = [
                "exp1_extended.yaml",
                "exp2_distloss_extended.yaml",
                "exp2_non0mean_extended.yaml",
                "exp2_lightwind_extended.yaml",
            ]

            which_plots = [
                "time",
                "boxplots",
                "losses",
                "gradients",
                "runs",
                "collisions",
            ]

            for filename in analysis_configs:
                config_path = f"./configs/{filename}"
                with open(config_path, "r") as config_file:
                    experiment_config = yaml.safe_load(config_file)
                experiment_name = config_path.split("/")[-1].split(".")[0]
                records_path = "./records"
                found_counter = 0
                for subfolder_name in os.listdir(records_path):
                    if '_'.join(subfolder_name.split('_')[:-1]) == experiment_name:
                        print("Replotting ", subfolder_name)
                        found_counter += 1
                        dirname = os.path.join(records_path, subfolder_name)
                        delete_old_plots(dirname, which_plots)
                        analysis = Analysis.load(dirname)
                        analysis.variant_config = experiment_config
                        plots = {
                            "time":       False,
                            "boxplots":   False,
                            "losses":     False,
                            "gradients":  False,
                            "runs":       False,
                            "collisions": False,
                        }
                        plots.update({key: True for key in which_plots})
                        analysis.plot_states_and_losses(**plots)
                        analysis.experiment_config["variation_config"].update(experiment_config)
                        analysis.save()