import torch
from experiment_contingent_estimator.aicon import ContingentEstimatorAICON

# ===========================================================================================

seed = 10

if __name__ == "__main__":
    print("Running General AICON Test")
    aicon = ContingentEstimatorAICON(vel_control=False)
    aicon.run(2500, seed, render=True, initial_action=torch.tensor([0.0, 0.0, 0.0], device=aicon.device), prints=1, step_by_step=True, record_video=False)