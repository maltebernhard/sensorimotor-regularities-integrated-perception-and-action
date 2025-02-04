from typing import Dict, List, Tuple
from components.analysis import Analysis
from run_analysis import config_dicts
from itertools import product

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
        "_".join([get_key_from_value(config_dicts[key], variation[key]) for key in config_dicts.keys() if (key not in invariants.keys() or len(invariants[key])>1) and count_variations(experiment_variations, key)>1]): {
            "smcs":                variation["smcs"],
            "control":             variation["control"],
            "distance_sensor":     variation["distance_sensor"],
            "sensor_noise":        variation["sensor_noise"],
            "moving_target":       variation["moving_target"],
            "observation_loss":    variation["observation_loss"],
            "foveal_vision_noise": variation["foveal_vision_noise"],
        } for variation in experiment_variations if all([variation[key] in invariants[key] for key in invariants.keys()])
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

def create_names_and_invariants(experiment_variations: list, invariance_config = Dict[str,Tuple[str,List[object]]]):
    # list containing invariant names and values
    invariants: Dict[str,Tuple[str,object]] = {}
    for config_key in invariance_config:
        if invariance_config[config_key] is None:
            invariants[config_key] = [(get_key_from_value(config_dicts[config_key], val), [val]) for val in config_dicts[config_key].values() if val in [variation[config_key] for variation in experiment_variations]]
        else:
            invariants[config_key] = [(invariance_config[config_key][0], invariance_config[config_key][1])]
    configs = []
    for combination in product(*[invariants[key] for key in invariants.keys()]):
        config_name = "_".join([item[0] for item in combination])
        config_dict = {key: item[1] for key, item in zip(invariants.keys(), combination)}
        configs.append((config_name, config_dict))
    return configs

def plot_states_and_losses(analysis: Analysis, invariant_config):
    if "Experiment1" in analysis.record_dir:
        exp_id = 1
    elif "Experiment2" in analysis.record_dir:
        exp_id = 2
    else:
        raise ValueError("Unknown experiment ID")
    
    experiment_variations = analysis.variations
    configs = create_names_and_invariants(experiment_variations, invariant_config)
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
            [(-1, 4)],
            [(-1, 2)],
            [(0.5, 3)],
        ]
    },
}

#path = "records/2025_02_04_17_59_Experiment1"
path = "records/2025_02_04_17_49_Experiment2"

if __name__ == "__main__":
    analysis = Analysis.load(path)

    if "Experiment1" in path:
        """
        Explanation of the invariant_config dictionary:
        - The key is the name of the configuration parameter that should be invariant for one plot
        - The value is either
            - None: All possible values for this parameter will be plotted
            - A tuple:
                - The first element will be in the filename, the second element is a list of the values that should be plotted together
        """
        # invariant_config = {
        #     "smcs": None,
        #     "foveal_vision_noise": None,
        # }
        # invariant_config = {
        #     "smcs":                 ("Both",          [config_dicts["smcs"]["Both"]]),
        #     "foveal_vision_noise":  ("NoFVNoise",     [config_dicts["foveal_vision_noise"]["NoFVNoise"]]),
        #     "sensor_noise":         ("OnlyTwoNoises", [config_dicts["sensor_noise"]["SmallNoise"], config_dicts["sensor_noise"]["LargeNoise"]]),
        # }
        invariant_config = {
            "smcs":                 ("Both",           [config_dicts["smcs"]["Both"]]),
            "foveal_vision_noise":  ("NoFVNoise",     [config_dicts["foveal_vision_noise"]["NoFVNoise"]]),
            "sensor_noise":         ("2Noises",    [config_dicts["sensor_noise"]["SmallNoise"], config_dicts["sensor_noise"]["LargeNoise"]]),
        }
    elif "Experiment2" in path:
        invariant_config = {
            "sensor_noise": None,
            "foveal_vision_noise": None,
        }
        # invariant_config = {
        #     "sensor_noise":         ("HugeDistNoise", [config_dicts["sensor_noise"]["HugeDistNoise"]]),
        #     "foveal_vision_noise":  ("NoFVNoise",     [config_dicts["foveal_vision_noise"]["NoFVNoise"]]),
        # }

    plot_states_and_losses(analysis, invariant_config)

    # configs = create_names_and_invariants(analysis.variations, invariant_config)
    # for config in configs:
    #     axes = create_axes(analysis.variations, config[1])
    #     plotting_config = create_plotting_config(config[0], plotting_states, axes)
    #     print(axes.keys())
    #     analysis.plot_state_runs(plotting_config, 'AICON', save=True, show=False)
    #     #analysis.run_demo(axes['AICON'], run_number=1, step_by_step=True, record_video=False)