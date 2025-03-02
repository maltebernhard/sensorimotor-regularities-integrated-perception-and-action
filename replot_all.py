import subprocess

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

if __name__ == "__main__":
    for filename in analysis_configs:
        subprocess.run(["python", "run_analysis.py", "replot", "./configs/" + filename], check=True)