import torch
from experiment_general.aicon import GeneralTestAICON

# ===========================================================================================

seed = 10
# Great seed for 90 degree config failing without gaze fixation | good 2-obstacle seed
#seed = 50

if __name__ == "__main__":
    aicon = GeneralTestAICON(moving_target=False,)# sensor_angle_deg=130)

    #aicon.load("./test_recording/data.yaml")

    aicon.run(200, seed, initial_action=torch.tensor([0.0, 0.0, 0.0], device=aicon.device), render=True, prints=1, step_by_step=True)# record_dir="test")
    
    
    # for i in range(10):
    #     aicon.run(50, seed+i, initial_action=torch.tensor([0.0, 0.0, 0.0], device=aicon.device), render=True, prints=200, step_by_step=False, record_data=True)
    # aicon.logger.plot_estimation_error("PolarTargetPos", value_keys={0:"Target Distance", 1:"Target Offset Angle"}, save_path=aicon.record_dir)