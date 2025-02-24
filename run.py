import yaml
from run_analysis import create_analysis_from_custom_config

# ============================================================

analyses: list[str] = []

for filepath in analyses:
    with open(filepath, "r") as config_file:
        custom_config = yaml.safe_load(config_file)
    analysis = create_analysis_from_custom_config(filepath.split("/")[-1].split(".")[0], custom_config)
    analysis.run_analysis()
    analysis.plot_states_and_losses()