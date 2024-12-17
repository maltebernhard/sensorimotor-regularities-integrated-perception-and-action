import torch
from experiment_foveal_vision.aicon import FovealVisionAICON

# ===========================================================================================

#seed = 10
#seed = 50
seed = 70

if __name__ == "__main__":
    aicon = FovealVisionAICON()#sensor_angle_deg=90, moving_target=True)

    aicon.run(2000, seed, render=True, initial_action=torch.tensor([0.0, 0.0, 0.0], device=aicon.device), prints=1, step_by_step=False, record_data=False)