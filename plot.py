from typing import Dict, List, Tuple
from components.analysis import Analysis
from configs import ExperimentConfig as configs
from itertools import product
import sys

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


def create_axes(experiment_variations: list, invariants: Dict[str,list]):
    axes = {
        "_".join([get_key_from_value(vars(vars(configs)[key]), variation[key]) for key in configs.keys if (key not in invariants.keys() or len(invariants[key])>1) and count_variations(experiment_variations, key)>1]): {
            "smcs":                variation["smcs"],
            "control":             variation["control"],
            "distance_sensor":     variation["distance_sensor"],
            "sensor_noise":        variation["sensor_noise"],
            "moving_target":       variation["moving_target"],
            "observation_loss":    variation["observation_loss"],
            "fv_noise":            variation["fv_noise"],
            "desired_distance":    variation["desired_distance"],
        } for variation in experiment_variations if all([variation[key] in invariants[key] for key in invariants.keys()])
    }
    return axes

def create_plotting_config(name: str, state_config: dict, axes_config: dict, plot_styles: dict = None, exclude_runs=[]):
    if state_config is None:
        state_config = {
            "PolarTargetPos": {
                "indices": [0],
                "labels" : ["Distance"],
                "ybounds": [[(-1, 20)],[(-1, 20)],[(-1, 20)]]
            },
        }
    plotting_config = {
        "name": name,
        "states": state_config,
        "goals": {
            "PolarGoToTarget": {
                "ybounds": (0, 20)
            },
        },
        "axes": axes_config,
        "exclude_runs": exclude_runs
    }
    if plot_styles is not None:
        plotting_config["style"] = plot_styles
    return plotting_config

def create_plot_names_and_invariants(experiment_variations: list, invariance_config = Dict[str,Tuple[str,List[object]]]):
    """
    Creates a list of name-config-tuples specifying plots.
    Args:
        experiment_variations: list of all variations in the experiment
        invariance_config: dictionary specifying which parameters should be invariant for each plot
    """
    invariants: Dict[str,Tuple[str,object]] = {}
    for iv_config_key, iv_config in invariance_config.items():
        if iv_config is None:
            # get all possible configs for invariant key
            invariants[iv_config_key] = [(subconfig_key, [val]) for subconfig_key, val in configs.__dict__[iv_config_key].__dict__.items() if val in [variation[iv_config_key] for variation in experiment_variations]]
        else:
            iv_name, iv_value_keys = iv_config
            invariants[iv_config_key] = [(iv_name, [configs.__dict__[iv_config_key].__dict__[value_key] for value_key in iv_value_keys])]
    plot_configs = []
    for combination in product(*[invariants[key] for key in invariants.keys()]):
        # join names of invariant properties to form plot name
        config_name = "_".join([item[0] for item in combination])
        config_dict = {key: item[1] for key, item in zip(invariants.keys(), combination)}
        plot_configs.append((config_name, config_dict))
    return plot_configs

def plot_states_and_losses(analysis: Analysis, invariant_config, plotting_states_config=None, show=False, plot_styles=None, plot_ax_runs:str=None, run_demo:int=None, exclude_runs=[]):
    experiment_variations = analysis.variations
    plot_configs = create_plot_names_and_invariants(experiment_variations, invariant_config)
    for config in plot_configs:
        axes = create_axes(experiment_variations, config[1])
        plotting_config = create_plotting_config(config[0], plotting_states_config, axes, plot_styles=plot_styles, exclude_runs=exclude_runs)
        print(f"Plot {config[0]} has the following axes:")
        print([key for key in axes.keys()])
        analysis.plot_states(plotting_config, save=True, show=show)
        analysis.plot_goal_losses(plotting_config, save=True, show=show)
    if plot_ax_runs is not None:
        if type(plot_ax_runs) is str:
            plot_ax_runs = (plot_ax_runs, None)
        analysis.plot_state_runs(plotting_config, plot_ax_runs[0], plot_ax_runs[1], save=True, show=False)
        if run_demo is not None:
            analysis.run_demo(axes[plot_ax_runs[0]], run_number=run_demo, step_by_step=True, record_video=False)

def create_standard_plotting_states(exp_id: int):
    if exp_id == 1:
        """
        Explanation of the invariant_config dictionary:
        - The key is the name of the configuration parameter that should be invariant for one plot
        - The value is either
            - None: All possible values for this parameter will be plotted
            - A tuple:
                - The first element will be in the filename, the second element is a list of the values that should be plotted together, excluding all other values
        """
        invariant_config = {
            "smcs": None,
            "moving_target": None,
            "control": None,
            #"fv_noise": None,
        }
        plot_styles = {
            # 'aicon_stationary_target': {
            #     'label': 'AICON control | stationary target',
            #     'color': 'blue', # red, green, blue, cyan, magenta, yellow, black, white
            #     'linestyle': 'solid', # dotted, dashed, dashdot
            #     'linewidth': 2
            # },
            # 'aicon_sine_target': {
            #     'label': 'AICON control | moving target',
            #     'color': 'blue',
            #     'linestyle': 'dashed',
            #     'linewidth': 2
            # },
            # 'manual_stationary_target': {
            #     'label': 'designed control | stationary target',
            #     'color': 'red',
            #     'linestyle': 'solid',
            #     'linewidth': 2
            # },
            # 'manual_sine_target': {
            #     'label': 'designed control | moving target',
            #     'color': 'red',
            #     'linestyle': 'dashed',
            #     'linewidth': 2
            # }
            'small_noise': {
                'label': 'small noise',
                'color': 'blue',
                'linestyle': 'solid',
                'linewidth': 2
            },
            'large_noise': {
                'label': 'large noise',
                'color': 'green',
                'linestyle': 'solid',
                'linewidth': 2
            },
            'div_noise': {
                'label': 'divergence noise',
                'color': 'orange',
                'linestyle': 'solid',
                'linewidth': 2
            },
            'tri_noise': {
                'label': 'triangulation noise',
                'color': 'red',
                'linestyle': 'solid',
                'linewidth': 2
            },
        }
        plotting_states = {
            "PolarTargetPos": {
                "indices": [0],
                "labels" : ["Distance"],
                "ybounds": [
                    # Distance State
                    [(-1, 20)],
                    # Distance Estimation Error
                    [(-6, 10)],
                    # Distance Estimation Uncertainty
                    [(0, 4)],
                ]
            },
        }

    elif exp_id == 2:
        invariant_config = {
            #"smcs":             ("BothVSNone", ["nosmcs", "both"]),
            "smcs":             None,
            #"sensor_noise":     None,
            #"fv_noise":         None,
            "moving_target":    None,
            #"observation_loss": None,
        }
        plot_styles = {
            # 'nosmcs': {
            #     'label': 'No SMCs',
            #     'color': 'red', # red, green, blue, cyan, magenta, yellow, black, white
            #     'linestyle': 'solid', # dotted, dashed, dashdot
            #     'linewidth': 2
            # },
            # 'tri': {
            #     'label': 'Triangulation',
            #     'color': 'green',
            #     'linestyle': 'solid',
            #     'linewidth': 2
            # },
            # 'div': {
            #     'label': 'Divergence',
            #     'color': 'orange',
            #     'linestyle': 'solid',
            #     'linewidth': 2
            # },
            # 'both': {
            #     'label': 'Triangulation + Divergence',
            #     'color': 'blue',
            #     'linestyle': 'solid',
            #     'linewidth': 2
            # },
            'no_obs_loss': {
                'label': 'No observation loss',
                'color': 'red',
                'linestyle': 'solid',
                'linewidth': 2
            },
            'tri_loss': {
                'label': 'Triangulation loss',
                'color': 'green',
                'linestyle': 'solid',
                'linewidth': 2
            },
            'div_loss': {
                'label': 'Divergence loss',
                'color': 'orange',
                'linestyle': 'solid',
                'linewidth': 2
            },
        }
        plotting_states = {
            "PolarTargetPos": {
                "indices": [0],
                "labels" : ["Distance"],
                "ybounds": [
                    # Distance State
                    [(-1, 20)],
                    # Distance Estimation Error
                    [(-6, 10)],
                    # Distance Estimation Uncertainty
                    [(-1, 15)],
                ]
            },
        }
    return plotting_states, invariant_config, plot_styles

# ==================================================================================

if __name__ == "__main__":
    if len(sys.argv) < 2:
        print("Usage: python plot.py <path_to_analysis>")
        sys.exit(1)
    path = sys.argv[1]
    if "Experiment1" in path:
        exp_id = 1
    elif "Experiment2" in path:
        exp_id = 2

    analysis = Analysis.load(path)
    plotting_states, invariant_config, plot_styles = create_standard_plotting_states(exp_id)
    plot_states_and_losses(analysis, invariant_config, plotting_states, plot_styles=plot_styles, plot_ax_runs=("aicon_sine_target",None), run_demo=9)#, show=False, print_ax_keys=True, plot_ax_runs=("nosmcs",None), run_demo=1)#, plot_ax_runs=("nosmcs",None), exclude_runs=[9])
