import torch
from experiment_contingent_goal.aicon import ContingentGoalAICON

# ===========================================================================================

seed = 10
#seed = 20

if __name__ == "__main__":
    print("Running General AICON Test")
    aicon = ContingentGoalAICON()

    # aicon.env.reset(seed)
    # aicon.update_observations()
    # aicon.print_state("target_distance")

    aicon.run(2500, seed, render=True, initial_action=torch.tensor([0.0, 0.0, 0.0], device=aicon.device), prints=1, step_by_step=False, record_video=False)