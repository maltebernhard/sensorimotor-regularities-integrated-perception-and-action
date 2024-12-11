import torch
from experiment_general.aicon import GeneralTestAICON

# ===========================================================================================

seed = 10
# Great seed for 90 degree config failing without gaze fixation | good 2-obstacle seed
#seed = 50

# TODO: acceleration control with active interconnection becomes unstable

if __name__ == "__main__":
    print("Running General AICON Test")
    aicon = GeneralTestAICON(num_obstacles=0, vel_control=True, moving_target=False, sensor_angle_deg=170)

    aicon.run(2500, seed, initial_action=torch.tensor([0.0, 0.0, 0.0], device=aicon.device), render=True, prints=1, step_by_step=True, record_video=False)