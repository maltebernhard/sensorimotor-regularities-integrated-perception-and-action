import torch
from experiment_contingent_interconnection.aicon import ContingentInterconnectionAICON

# ===========================================================================================

seed = 10
#seed = 23
#seed = 50
#seed = 70

# NOTE: interesting behavior (angular vel roughly according to gaze fixation constraint) when close to target, otherwise completely chaotic
if __name__ == "__main__":
    aicon = ContingentInterconnectionAICON(moving_target=False)
    aicon.run(200, seed, render=True, initial_action=torch.tensor([0.0, 0.0, 0.0], device=aicon.device), prints=1, step_by_step=False, record_data=True)