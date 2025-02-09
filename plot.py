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
    for iv_config_key, iv_config_val in invariance_config.items():
        if iv_config_val is None:
            # get all possible configs for invariant key
            invariants[iv_config_key] = [(subconfig_key, [val]) for subconfig_key, val in configs.__dict__[iv_config_key].__dict__.items() if val in [variation[iv_config_key] for variation in experiment_variations]]
        else:
            iv_name, iv_values = iv_config_val
            invariants[iv_config_key] = [(iv_name, iv_values)]
    plot_configs = []
    for combination in product(*[invariants[key] for key in invariants.keys()]):
        # join names of invariant properties to form plot name
        config_name = "_".join([item[0] for item in combination])
        config_dict = {key: item[1] for key, item in zip(invariants.keys(), combination)}
        plot_configs.append((config_name, config_dict))
    return plot_configs

def plot_states_and_losses(analysis: Analysis, invariant_config, plotting_states_config=None, show=False, ax_display_config=None, plot_ax_runs:str=None, run_demo:int=None, exclude_runs=[]):
    experiment_variations = analysis.variations
    plot_configs = create_plot_names_and_invariants(experiment_variations, invariant_config)
    for config in plot_configs:
        axes = create_axes(experiment_variations, config[1])
        plotting_config = create_plotting_config(config[0], plotting_states_config, axes, plot_styles=ax_display_config, exclude_runs=exclude_runs)
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

# ==================================================================================

if __name__ == "__main__":
    if len(sys.argv) < 2:
        print("Usage: python plot.py <path_to_analysis>")
        sys.exit(1)
    path = sys.argv[1]

    analysis = Analysis.load(path)

    if "Experiment1" in path:
        plotting_states = {
            "PolarTargetPos": {
                "indices": [0],
                "labels" : ["Distance"],
                "ybounds": [
                    # Distance State
                    [(-1, 20)],
                    # Distance Estimation Error
                    [(-1, 20)],
                    # Distance Estimation Uncertainty
                    [(-1, 20)],
                ]
            },
        }

        """
        Explanation of the invariant_config dictionary:
        - The key is the name of the configuration parameter that should be invariant for one plot
        - The value is either
            - None: All possible values for this parameter will be plotted
            - A tuple:
                - The first element will be in the filename, the second element is a list of the values that should be plotted together, excluding all other values
        """
        invariant_config = {
            "smcs":          None,
            #"fv_noise":      None,
            #"moving_target": ("TargetMovementComparison", [configs.moving_target.sine_target, configs.moving_target.stationary_target]),
        }
        # invariant_config = {
        #     "smcs":         ("Tri",           [cd.smcs.tri]),
        #     "fv_noise":     ("NoFVNoise",     [cd.fv_noise.no_fv_noise]),
        #     "sensor_noise": ("2Noises",       [cd.sensor_noise.small_noise, cd.sensor_noise.large_noise]),
        # }

        plot_styles = {
            'aicon_stationary_target': {
                'label': 'AICON control | stationary target',
                'color': 'blue', # red, green, blue, cyan, magenta, yellow, black, white
                'linestyle': 'solid', # dotted, dashed, dashdot
                'linewidth': 2
            },
            'aicon_sine_target': {
                'label': 'AICON control | moving target',
                'color': 'blue',
                'linestyle': 'dashed',
                'linewidth': 2
            },
            'manual_stationary_target': {
                'label': 'designed control | stationary target',
                'color': 'red',
                'linestyle': 'solid',
                'linewidth': 2
            },
            'manual_sine_target': {
                'label': 'designed control | moving target',
                'color': 'red',
                'linestyle': 'dashed',
                'linewidth': 2
            }
        }
        plot_states_and_losses(analysis, invariant_config, plotting_states, show=False, ax_display_config=plot_styles)#, plot_ax_runs=("aicon_sine_target",None), run_demo=2)#, exclude_runs=[1])


    elif "Experiment2" in path:
        plotting_states = {
            "PolarTargetPos": {
                "indices": [0],
                "labels" : ["Distance"],
                "ybounds": [
                    # Distance State
                    [(-10, 20)],
                    # Distance Estimation Error
                    [(-10, 20)],
                    # Distance Estimation Uncertainty
                    [(-10, 20)],
                ]
            },
        }

        invariant_config = {
            #"sensor_noise":     None,
            "moving_target":    None,
            "observation_loss": None,
        }
        # invariant_config = {
        #     "sensor_noise": ("HugeDistNoise",    [configs.sensor_noise.huge_dist_noise]),
        #     #"smcs":         ("TriBoth",        [configs.smcs.both, configs.smcs.tri]),
        #     #"sensor_noise": ("SmallNoise",    [configs.sensor_noise.small_noise]),
        #     #"fv_noise":     ("NoFVNoise",     [configs.fv_noise.no_fv_noise]),
        #     #"moving_target": ("Sine", [configs.moving_target.sine_target]),
        #     "moving_target": ("Sine", [configs.moving_target.sine_target]),
        #     #"observation_loss": ("TriDistLoss", [configs.observation_loss.dist_loss, configs.observation_loss.tri_loss]),
        #     "observation_loss": ("NoObsLoss", [configs.observation_loss.no_obs_loss]),
        # }
        plot_states_and_losses(analysis, invariant_config, plotting_states)#, plot_ax_runs=("nosmcs",None), run_demo=8)#, show=False, print_ax_keys=True, plot_ax_runs=("nosmcs",None), run_demo=1)#, plot_ax_runs=("nosmcs",None), exclude_runs=[9])
