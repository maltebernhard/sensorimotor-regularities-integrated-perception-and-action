import torch
from experiment_divergence.aicon import DivergenceAICON
import time

# ===========================================================================================

#seed = 10
#seed = 50
seed = 70

if __name__ == "__main__":
    aicon = DivergenceAICON()

    aicon.run(2000, seed, render=True, initial_action=torch.tensor([0.0, 0.0, 0.0], device=aicon.device), prints=1, step_by_step=True, record_data=False)

    # for i in range(10):
    #     aicon.run(2000, seed+i, render=False, initial_action=torch.tensor([0.0, 0.0, 0.0], device=aicon.device), prints=100, step_by_step=False, record_data=True)
    
    # aicon.load("records/2024_12_19_21_57_Divergence/data.yaml")
    # aicon.plot()