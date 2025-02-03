from components.analysis import Analysis
from run_analysis import get_relevant_noise_keys, noise_dict

# ==================================================================================

plotting_controls = {
    "AICON": False,
    "CONTROL": True,
}
plotting_smc_types = {
    "None": [],
    "Both": ["Divergence", "Triangulation"],
    "Tri":  ["Triangulation"],
    "Div":  ["Divergence"],
}

def create_axes(experiment_variations: dict, smc_key: str):
    axes = {f"{smc_key}_{control_key}_{noise_key}" : {
        "aicon_type": {
            "SMCs":    plotting_smc_types[smc_key],
            "Control": control,
            "DistanceSensor": variation["aicon_type"]["DistanceSensor"],
        },
        "sensor_noise": noise,
        "target_movement": variation["moving_target"],
        "observation_loss": variation["observation_loss"],
        "foveal_vision_noise": variation["foveal_vision_noise"],
    } for control_key, control in plotting_controls.items() for noise_key, noise in noise_dict.items() for variation in experiment_variations if \
        plotting_smc_types[smc_key] == variation["aicon_type"]["SMCs"] \
        and control == variation["aicon_type"]["Control"] \
        and noise == variation["sensor_noise"]
    }
    return axes

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

path = "records/2025_02_03_17_12_Experiment1"

if __name__ == "__main__":
    analysis = Analysis.load(path)
    if "Experiment1" in path:
        exp_id = 1
    elif "Experiment2" in path:
        exp_id = 2
    else:
        raise ValueError("Unknown experiment ID")
    
    experiment_variations = analysis.variations

    for smc_key, smc in plotting_smc_types.items():
        if smc in [config["aicon_type"]["SMCs"] for config in experiment_variations]:
            axes = create_axes(experiment_variations, smc_key)
            plotting_config = {
                "name": f"{exp_id}",
                "states": plotting_states,
                "goals": {
                    "PolarGoToTarget": {
                        "ybounds": (0, 20)
                    },
                },
                #"runs": [7, 17], # (used to select the runs to plot)
                "axes": axes
            }
            analysis.plot_states(plotting_config, save=True, show=False)
            analysis.plot_goal_losses(plotting_config, save=True, show=False)

            # analysis.run_demo(plotting_config1["axes"]["Test"], run_number=17, record_video=False)
            # analysis.plot_state_runs(plotting_config1, "Test", save=True, show=False)