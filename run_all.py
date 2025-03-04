import os
import time
import yaml
from components.analysis import Analysis
from run_analysis import create_analysis_from_custom_config

# ============================================================

analysis_configs: list[str] = [
    # -------- AICON vs. control --------
    # "exp1_sine.yaml",
    # "exp1_extended.yaml",
    # "exp1_non0mean.yaml",
    # ---------- Disturbances ----------
    # Dist Sensor Failure
    # "exp2_distloss_sine.yaml",
    # "exp2_distloss_extended.yaml",
    # Dist Noise
    "exp2_non0mean_extended.yaml",
    # "exp2_non0mean_stat.yaml",
    # "exp2_non0mean_sine.yaml",
    # Action Disturbance
    "exp2_wind_extended.yaml",
    # "exp2_wind_stat_nodist.yaml",
    # "exp2_wind_sine_nodist.yaml",
    # "exp2_wind_sine_distdot.yaml",
    # "exp2_wind_stat_distdot.yaml",
]

num_runs:  int = 100
num_steps: int = 500

if __name__ == "__main__":
    for filename in analysis_configs:
        with open("./configs/" + filename, "r") as config_file:
            custom_config = yaml.safe_load(config_file)

        custom_config.update({
            "num_runs": num_runs,
            "num_steps": min(num_steps, custom_config["plotting_config"]["xbounds"][1]),
        })

        analysis = create_analysis_from_custom_config(filename.split("/")[-1].split(".")[0], custom_config)
        time.sleep(1)
        analysis.run_analysis()
        #analysis.plot_states_and_losses()

    for filename in analysis_configs:
        config_path = f"./configs/{filename}"
        with open(config_path, "r") as config_file:
            custom_config = yaml.safe_load(config_file)
        custom_name = config_path.split("/")[-1].split(".")[0]
        records_path = "./records"
        found_counter = 0
        for subfolder_name in os.listdir(records_path):
            if '_'.join(subfolder_name.split('_')[:-1]) == custom_name:
                found_counter += 1
                dirname = os.path.join(records_path, subfolder_name)
                analysis = Analysis.load(dirname)
                analysis.custom_config = custom_config
                analysis.plot_states_and_losses()