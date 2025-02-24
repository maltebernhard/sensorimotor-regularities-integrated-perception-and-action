from typing import Dict
from components.analysis import Analysis
from configs.configs import ExperimentConfig as configs

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


def create_axes(experiment_variations: list[dict], invariants: Dict[str,list]):
    axes = {
        "_".join([get_key_from_value(vars(vars(configs)[key]), variation[key]) for key in configs.keys if (key not in invariants.keys() or len(invariants[key])>1) and count_variations(experiment_variations, key)>1]): {
            subkey: variation[subkey] for subkey in variation.keys()
        } for variation in experiment_variations if all([variation[key] in invariants[key] for key in invariants.keys()])
    }
    return axes

def create_plotting_config(name: str, state_config: dict, axes_config: dict, plot_styles: dict):
    return {
        "name":         name,
        "states":       state_config,
        "goals":        {"PolarGoToTarget": {}},
        "axes":         axes_config,
        "exclude_runs": [],
        "style":        plot_styles
    }

def plot_states_and_losses(analysis: Analysis):
    plotting_states_config=analysis.custom_config["plotting_state_config"]
    plot_styles=analysis.custom_config["plotting_style_config"]
    experiment_variations = analysis.variations
    plot_configs = [("all", {key: [variation[key] for variation in experiment_variations] for key in configs.keys})]
    for plot_name, plot_variation_values in plot_configs:
        axes = create_axes(experiment_variations, plot_variation_values)
        plotting_config = create_plotting_config(plot_name, plotting_states_config, axes, plot_styles=plot_styles)
        print(f"Plot {plot_name} has the following axes:")
        print([key for key in axes.keys()])
        analysis.plot_states(plotting_config)
        analysis.plot_state_bars(plotting_config)
        analysis.plot_loss_and_gradient(plotting_config)
        analysis.plot_goal_losses(plotting_config)
        for ax_key in axes.keys():
            analysis.plot_state_runs(plotting_config, ax_key)
