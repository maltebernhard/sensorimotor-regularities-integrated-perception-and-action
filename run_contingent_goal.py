import torch
from experiment_contingent_goal.aicon import ContingentGoalAICON

# ===========================================================================================

#seed = 10
seed = 50
#seed = 70

if __name__ == "__main__":
    print("Running General AICON Test")
    aicon = ContingentGoalAICON()

    aicon.run(400, seed, render=True, initial_action=torch.tensor([0.0, 0.0, 0.0], device=aicon.device), prints=1, step_by_step=True, record_dir="test")