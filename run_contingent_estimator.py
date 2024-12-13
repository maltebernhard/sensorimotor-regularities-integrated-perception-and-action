import numpy as np
import torch
from experiment_contingent_estimator.aicon import ContingentEstimatorAICON

# ===========================================================================================

def test_acc_control_pos_estimate_forward(aicon: ContingentEstimatorAICON):
    aicon.reset()
    print("Initial State: ", aicon.REs["PolarTargetPos"].state_mean)

    aicon.render()
    input("Press Enter to continue...")

    for _ in range(100):
        #action = torch.tensor([1.0, -1.0, 1.0], device=aicon.device)
        action = torch.tensor([0.0, 1.0, 0.0], device=aicon.device)

        print(f"================================ {[f'{x:.2f}' for x in action.tolist()]} ==================================")

        buffer_dict = {key: estimator.get_buffer_dict() for key, estimator in list(aicon.REs.items()) + list(aicon.obs.items())}
        u = aicon.get_control_input(action)
        aicon.REs["PolarTargetPos"].call_predict(u, buffer_dict)
        aicon.REs["RobotState"].call_predict(u, buffer_dict)
        print("Forward Estimate: ", buffer_dict['PolarTargetPos']['state_mean'])

        estimate = aicon.eval_step(action)
        print("Interconnected Estimate: ", estimate["PolarTargetPos"]["state_mean"])

        aicon.env.step(np.array(action.cpu()))
        # for key, buffer in estimate.items():
        #     aicon.REs[key].load_state_dict(buffer) if key in aicon.REs.keys() else None
        for key, buffer in buffer_dict.items():
            aicon.REs[key].load_state_dict(buffer) if key in aicon.REs.keys() else None
        aicon.render()
        input("Press Enter to continue...")

def test_acc_control_vel_estimate_forward(aicon: ContingentEstimatorAICON):
    aicon.reset()
    for _ in range(100):
        action = 2 * torch.rand(3) - 1
        #print(f"Random action: {action}")
        print(f"================================ {[f'{x:.2f}' for x in action.tolist()]} ==================================")
        estimate = aicon.compute_estimator_action_gradient("RobotState", action)[1]["state_mean"]
        
        aicon.step(action)
        if not torch.allclose(aicon.REs["RobotState"].state_mean, estimate, atol=1e-10):
            print("Estimate: ", estimate)
            aicon.print_state("RobotState")
            actual_vel = [aicon.env.robot.vel[0], aicon.env.robot.vel[1], aicon.env.robot.vel_rot] 
            print("Actual Vel: ", actual_vel)
            print("Offset: ", aicon.REs["RobotState"].state_mean - estimate)

# ===========================================================================================

seed = 10

if __name__ == "__main__":
    print("Running General AICON Test")
    aicon = ContingentEstimatorAICON()

    #test_acc_control_pos_estimate_forward(aicon)

    aicon.run(200, seed, render=True, initial_action=torch.tensor([0.0, 0.0, 0.0], device=aicon.device), prints=1, step_by_step=False, record_dir="test")

    # print("Goal Grad: ", aicon.compute_goal_action_gradient(aicon.goals["GoToTarget"]))
    # print("Estimator Grad: ", aicon.compute_estimator_action_gradient("PolarTargetPos", aicon.last_action)[0]["state_mean"])