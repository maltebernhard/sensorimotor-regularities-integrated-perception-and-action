import torch
from experiment_divergence.aicon import DivergenceAICON

# ===========================================================================================

#seed = 10
#seed = 50
seed = 70

if __name__ == "__main__":
    aicon = DivergenceAICON()

    aicon.run(2000, seed, render=True, initial_action=torch.tensor([0.0, 0.0, 0.0], device=aicon.device), prints=1, step_by_step=True, record_data=False)