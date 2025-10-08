# Dynamically Leveraging Sensorimotor Regularities: Implementation Repository

This repository contains the complete implementation of the simulation environment and the coupled perception / behavior generation model described in my master's thesis "Dynamically Leveraging Sensorimotor Regularities by Integrating Perception and Behavior Generation".

**📖 For the complete theoretical background and detailed analysis, see the thesis document: [github.com/maltebernhard/master-thesis](https://github.com/maltebernhard/master-thesis)**

Currently, I am working on publishing a paper based on this thesis, which I will reference here once it has been accepted.

## Abstract

This implementation demonstrates a novel method for efficiently leveraging sensorimotor regularities (SMRs) within the Active InterCONnect (AICON) framework. The code provides:

- **AICON Framework**: A system based on a network interconnected recursive estimators, coupling state estimation and gradient-descent-based action generation. The resulting behavior robustly adapts to environment variations and disturbances.
- **Sensorimotor Regularities**: Implementation of "triangulation" and "divergence" SMRs that encode how motor commands modulate sensory input, allowing the agent to leverage active perception strategies in its behavior.
- **Environments**: Implementation of the 2d drone environment the thesis is based on, as well as a simple 3d environment using mujoco sim.
- **Experimental Suite**: Comprehensive experimental analysis (for the drone env only) demonstrating the effectiveness of the approach.

## Requirements

### System Requirements
- **Python**: 3.13 or higher
- **Operating System**: Windows, Linux, or macOS
- **Hardware**: CPU sufficient (GPU acceleration available via PyTorch)

### Python Dependencies
Install all dependencies using:
```bash
pip install -r requirements.txt
```

### Additional Notes
- If LaTeX errors occur during analysis plotting on Linux, install: `sudo apt install cm-super`
- For Windows users, ensure you have a recent version of Visual C++ redistributable

## Quick Start

### 1. Interactive Demo
Launch an interactive demonstration of the AICON agent:

```bash
python run_demo.py
```

The demo allows you to observe the agent's behavior in real-time. Configure different environment and model paramenters by editing the `variation_config` in `run_demo.py`.

### 2. Run Experiments
Execute the complete experimental suite from the thesis:

```bash
# Run all experiments (may take several hours)
python run_analysis.py run_all

# Run a specific experiment - example
python run_analysis.py run exp1_extended
```

Results are automatically saved with plots and analysis data.

### Experiment 1: Effective Behavior Composition
Demonstrates how AICON, as compared to simple, hand-coded control strategies, adapts to disturbances to maintain low perception and task errors.

### Experiment 2: Robustness Analysis
Tests AICON's performance and individual SMRs' contributions under:
- Sensor failures and noise
- Environmental disturbances
- Partial observability
- Dynamic obstacles

## License

This code is released under the same terms as the associated thesis. Please respect academic integrity guidelines when using or referencing this work.