import torch
from experiment_goal.aicon import ContingentGoalAICON

# ===========================================================================================

#seed = 10
#seed = 50
seed = 70

if __name__ == "__main__":
    aicon = ContingentGoalAICON(timestep=0.01)

    aicon.run(350, seed, render=True, initial_action=torch.tensor([0.0, 0.0, 0.0], device=aicon.device), prints=1, step_by_step=False, record_data=False)



    # for i in range(10):
    #     aicon.run(150, seed+i, initial_action=torch.tensor([0.0, 0.0, 0.0], device=aicon.device), render=True, prints=0, step_by_step=False, record_data=True)
    # aicon.logger.plot_estimation_error("PolarTargetPos", value_keys={0:"Target Distance", 1:"Target Offset Angle"}, save_path=aicon.record_dir)
    # aicon.logger.plot_goal_error("PolarTargetPos", save_path=aicon.record_dir)