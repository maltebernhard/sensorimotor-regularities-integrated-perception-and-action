from components.analysis import Analysis
from run_analysis import config_dicts

# ==================================================================================

def get_key_from_value(d: dict, value):
    for k, v in d.items():
        if v == value:
            return k
    print(f"Value {value} not found in dictionary {d}")
    return None

def count_variations(variations, invariant_key: str):
    seen_variations = []
    for variation in variations:
        if variation[invariant_key] not in seen_variations:
            seen_variations.append(variation[invariant_key])
    return len(seen_variations)


def create_axes(experiment_variations: list, invariants: dict):
    axes = {
        "_".join([get_key_from_value(config_dicts[key], variation[key]) for key in config_dicts.keys() if key not in invariants.keys() and count_variations(experiment_variations, key)>1]): {
            "smcs":                variation["smcs"],
            "control":             variation["control"],
            "distance_sensor":     variation["distance_sensor"],
            "sensor_noise":        variation["sensor_noise"],
            "moving_target":       variation["moving_target"],
            "observation_loss":    variation["observation_loss"],
            "foveal_vision_noise": variation["foveal_vision_noise"],
        } for variation in experiment_variations if all([invariants[key]==variation[key] for key in invariants.keys()])
    }
    return axes

def create_plotting_config(name: str, state_config: dict, axes_config: dict):
    return {
        "name": name,
        "states": state_config,
        "goals": {
            "PolarGoToTarget": {
                "ybounds": (0, 20)
            },
        },
        "axes": axes_config
    }

def create_names_and_invariants(experiment_variations: list, experiment_id: dict):
    if experiment_id == 1:
        smc_invariants = [smc for smc in config_dicts["smcs"].values() if smc in [variation["smcs"] for variation in experiment_variations]]
        fv_invariants = [fv for fv in config_dicts["foveal_vision_noise"].values() if fv in [variation["foveal_vision_noise"] for variation in experiment_variations]]
        configs = [(f"{get_key_from_value(config_dicts["smcs"], smc)}_{get_key_from_value(config_dicts["foveal_vision_noise"], fv)}", {
            "smcs": smc,
            "foveal_vision_noise": fv,
        }) for smc in smc_invariants for fv in fv_invariants]
    elif experiment_id == 2:
        noise_invariants = [noise for noise in config_dicts["sensor_noise"].values() if noise in [variation["sensor_noise"] for variation in experiment_variations]]
        fv_invariants = [fv for fv in config_dicts["foveal_vision_noise"].values() if fv in [variation["foveal_vision_noise"] for variation in experiment_variations]]
        configs = [(f"{get_key_from_value(config_dicts["sensor_noise"], noise)}_{get_key_from_value(config_dicts["foveal_vision_noise"], fv)}", {
            "sensor_noise": noise,
            "foveal_vision_noise": fv,
        }) for noise in noise_invariants for fv in fv_invariants]
    return configs

def plot_states_and_losses(analysis: Analysis):
    if "Experiment1" in analysis.record_dir:
        exp_id = 1
    elif "Experiment2" in analysis.record_dir:
        exp_id = 2
    else:
        raise ValueError("Unknown experiment ID")
    
    experiment_variations = analysis.variations
    configs = create_names_and_invariants(experiment_variations, exp_id)
    for config in configs:
        axes = create_axes(experiment_variations, config[1])
        plotting_config = create_plotting_config(config[0], plotting_states, axes)
        analysis.plot_states(plotting_config, save=True, show=False)
        analysis.plot_goal_losses(plotting_config, save=True, show=False)

# ==================================================================================

plotting_states = {
    "PolarTargetPos": {
        "indices": [0],
        "labels" : ["Distance"],
        "ybounds": [
            [(-1, 10)],
            [(-1, 2)],
            [(0, 4)],
        ]
    },
}

path = "records/2025_02_04_09_24_Experiment1"

if __name__ == "__main__":
    analysis = Analysis.load(path)
    plot_states_and_losses(analysis)

    # axes = create_axes(analysis.variations, {
    #     "smcs": ["Triangulation"],
    #     "foveal_vision_noise": config_dicts["foveal_vision_noise"]["NoFVNoise"],
    # })
    # plotting_config = create_plotting_config("Tri", plotting_states, axes)

    # ax = "AICON_SmallNoise_FVNoise_stationary_NoObsLoss"
    # analysis.plot_state_runs(plotting_config, ax, runs=[6,16], save=True, show=False)

    # analysis.run_demo(plotting_config["axes"][ax], run_number=6, record_video=False)
    # analysis.plot_state_runs(plotting_config, "Test", save=True, show=False)