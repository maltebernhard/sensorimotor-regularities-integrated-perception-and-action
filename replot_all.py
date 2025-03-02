import os
import yaml
from components.analysis import Analysis

plots = {
    "time":       False,
    "boxplots":   False,
    "losses":     False,
    "gradients":  False,
    "runs":       False,
    "collisions": False,
}


analysis_configs: list[str] = [
    # -------- AICON vs. control --------
    "exp1_sine.yaml",
    "exp1_extended.yaml",
    # # ---------- Disturbances ----------
    # # Dist Sensor Failure
    "exp2_distloss_sine.yaml",
    # # Dist Noise
    "exp2_non0mean_stat.yaml",
    "exp2_non0mean_sine.yaml",
    # Action Disturbance
    "exp2_wind_stat_nodist.yaml",
    "exp2_wind_sine_nodist.yaml",
    "exp2_wind_stat_dist.yaml",
    "exp2_wind_sine_dist.yaml",
    "exp2_wind_sine_distdot.yaml",
    "exp2_wind_stat_distdot.yaml",
]

which_plots = [
    # "time",
    # "boxplots",
    # "losses",
    # "gradients",
    # "runs",
    "collisions",
]

if __name__ == "__main__":

    # NOTE: example conditionals
    for filename in analysis_configs:
        foldername = f"./records/{filename.replace(".yaml", "")}_01"
        if os.path.exists(os.path.join(foldername, "records", "collisions_main.pdf")):
            os.remove(os.path.join(foldername, "records", "collisions_main.pdf"))
    
    
    
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
                analysis = Analysis.load(os.path.join(records_path, subfolder_name))
                analysis.custom_config = custom_config
                plots.update({key: True for key in which_plots})
                analysis.plot_states_and_losses(**plots)
                analysis.experiment_config["custom_config"].update(custom_config)
                analysis.save()