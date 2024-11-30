import torch
from experiment_general.aicon import GeneralTestAICON

# ===========================================================================================

seed = 10

if __name__ == "__main__":
    print("Running General AICON Test")
    # TODO: ACC CONTROL WITH NO INTERNAL VEL IS BROKEN
    aicon = GeneralTestAICON(num_obstacles=0, internal_vel=True, vel_control=True)

    grad, val = aicon.compute_estimator_action_gradient("PolarTargetPos", torch.tensor([0.0, 0.0, -1.0], device=aicon.device))
    print("Value:", val["state_mean"])
    print("Gradient:", grad["state_mean"])

    aicon.run(2500, seed, initial_action=torch.tensor([0.0, 0.0, 0.0], device=aicon.device), render=True, prints=1, step_by_step=True, record_video=False)
