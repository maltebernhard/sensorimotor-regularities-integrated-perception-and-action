import torch
from experiment_contingent_interconnection.aicon import ContingentInterconnectionAICON
import os

# ===========================================================================================

#seed = 10
#seed = 23
#seed = 50
seed = 70

# NOTE: interesting behavior (angular vel roughly according to gaze fixation constraint) when close to target, otherwise completely chaotic
if __name__ == "__main__":
    aicon = ContingentInterconnectionAICON(moving_target=False)
    
    # for i in range(3):
    #     aicon.run(50, seed+i, render=False, initial_action=torch.tensor([0.0, 0.0, 0.0], device=aicon.device), prints=1, step_by_step=False, record_data=True)
    #datafile = "records/2024_12_18_18_43_ContingentInterconnection/data.yaml"
    #aicon.load(datafile)
    # aicon.logger.plot_state("RobotVel", save_path="test")

    aicon.run(1000, seed, render=True, initial_action=torch.tensor([0.0, 0.0, 0.0], device=aicon.device), prints=1, step_by_step=True, record_data=False)

