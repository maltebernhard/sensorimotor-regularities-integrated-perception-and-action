import torch
from experiment_contingent_interconnection.aicon import ContingentInterconnectionAICON

# ===========================================================================================

seed = 10

if __name__ == "__main__":
    print("Running General AICON Test")
    aicon = ContingentInterconnectionAICON(vel_control=True)
    aicon.run(200, seed, render=True, initial_action=torch.tensor([0.0, 0.0, 0.0], device=aicon.device), prints=1, step_by_step=False, record_video=False)