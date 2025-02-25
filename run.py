import time
import yaml
from run_analysis import create_analysis_from_custom_config

# ============================================================

analysis_configs: list[str] = [
    # -------- AICON vs. control --------
    "classic_exp1.yaml",
    "classic_exp1_targetflightnchase.yaml",

    # ---------- Disturbances ----------
    # Dist Sensor Failure
    "classic_exp2_sine.yaml",
    # Dist Noise
    "non0mean_stationary_smcs.yaml",
    "non0mean_sine_smcs.yaml",
    # Action Disturbance
    "acc_wind_stationary_smc-comparison.yaml",
    "acc_wind_sine_smc-comparison.yaml",
    "acc_wind_stationary_distsensor_smc-comparison.yaml",
]

num_runs:  int = 10
num_steps: int = 500

if __name__ == "__main__":
    for filename in analysis_configs:
        with open("./configs/" + filename, "r") as config_file:
            custom_config = yaml.safe_load(config_file)

        custom_config.update({
            "num_runs": num_runs,
            "num_steps": num_steps,
        })

        analysis = create_analysis_from_custom_config(filename.split("/")[-1].split(".")[0], custom_config)
        time.sleep(1)
        analysis.run_analysis()
        analysis.plot_states_and_losses()