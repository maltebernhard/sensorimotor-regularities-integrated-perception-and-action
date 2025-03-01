import subprocess

analysis_configs: list[str] = [
    # -------- AICON vs. control --------
    "classic_exp1.yaml",
    #"classic_exp1_targetflightnchase.yaml",

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
    "acc_wind_sine_distsensor_smc-comparison.yaml",
    "acc_wind_sine_distdotsensor_smc-comparison.yaml",
    "acc_wind_stationary_distdotsensor_smc-comparison.yaml",
]

if __name__ == "__main__":
    for filename in analysis_configs:
        subprocess.run(["python", "run_analysis.py", "replot", "./configs/" + filename], check=True)