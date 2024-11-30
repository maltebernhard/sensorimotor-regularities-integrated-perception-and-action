import torch
from experiment_general.aicon import GeneralTestAICON

# ===========================================================================================

seed = 10

if __name__ == "__main__":
    print("Running General AICON Test")
    # TODO: INTERNAL VEL IS BROKEN BECAUSE OF SUPPRESSED MEAS UPDATES
    aicon = GeneralTestAICON(num_obstacles=0, internal_vel=False, vel_control=False)
    aicon.run(2500, seed, initial_action=torch.tensor([0.1, 0.0, 0.0], device=aicon.device), render=True, prints=1, step_by_step=True, record_video=False)
