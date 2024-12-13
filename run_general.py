import torch
from experiment_general.aicon import GeneralTestAICON

# ===========================================================================================

seed = 10
# Great seed for 90 degree config failing without gaze fixation | good 2-obstacle seed
#seed = 50

if __name__ == "__main__":
    print("Running General AICON Test")
    aicon = GeneralTestAICON(moving_target=False, sensor_angle_deg=130)

    aicon.run(200, seed, initial_action=torch.tensor([0.0, 0.0, 0.0], device=aicon.device), render=True, prints=1, step_by_step=True,)# record_dir="test")