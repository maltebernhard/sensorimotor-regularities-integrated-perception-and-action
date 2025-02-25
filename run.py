import time
import yaml
from run_analysis import create_analysis_from_custom_config

# ============================================================

analyses: list[str] = [
    "configs/non0mean_stationary_smcs.yaml",
    "configs/non0mean_sine_smcs.yaml",
    "configs/classic_exp2_sine.yaml",
    "configs/acc_wind_stationary_smc-comparison.yaml",
    "configs/acc_wind_sine_smc-comparison.yaml",
    "configs/acc_wind_stationary_distsensor_smc-comparison.yaml",
]

if __name__ == "__main__":
    for filepath in analyses:
        with open(filepath, "r") as config_file:
            custom_config = yaml.safe_load(config_file)
        analysis = create_analysis_from_custom_config(filepath.split("/")[-1].split(".")[0], custom_config)
        time.sleep(1)
        analysis.run_analysis()
        analysis.plot_states_and_losses()