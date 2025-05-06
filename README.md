# Sensorimotor Regularities in AICON

## todo

- test run all configs and plots
- comment all components


## Requirements

The code was tested on:
- Python 3.13.0
- Numpy 2.2.0
- Torch 2.5.1
- Weights & Biases (wandb) 0.19.6
- Pygame 2.6.1
- Pyyaml 6.0.2

Notes:
- If a Latex Error is thrown during plotting, it can likely be fixed by running
    ```bash
    sudo apt install cm-super
    ```

## Usage
### Demo
To run a demo of agent and environment, use
```bash
python run_demo.py
```
Within the file, the agent and environment can be easily configured to run demos for different scenarios (e.g. different sensorimotor regularities, obstacles, target motion patterns, etc.).

### Experiments
To run an analysis recording experimental data, use
```bash
python run_analysis.py <option>
```
where `<option>` is either of:
- `run_all`: records and plots all experiments used in this thesis
- `run <filename>`: records and plots an experiment specified by a config .yaml file (found in ./configs/)

## Project Structure

Besides the two run files for demo and analysis, the workspace contains 3 folders:
- `components/`: Contains the AICON and environment classes of our model.
- `model/`: Contains the model configuration for our experimental evaluation, including the SMRs "triangulation" and "divergence", target position and robot motion estimators, and the goal function.
- `configs/`: Contains the file `configs.py` specifying different configurable parameters of agent and environment. Additional configs can be added. YAML files specify different experiments, providing variations and plotting configs.