import time
import yaml
from run_analysis import create_analysis_from_custom_config

# ============================================================

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

num_runs:  int = 20
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