import numpy as np
import torch

import yaml

from components.aicon import AICON
from components.active_interconnection import ActiveInterconnection as AI
from components.estimator import RecursiveEstimator, Robot_Vel_Estimator, Target_Distance_Estimator, Target_Pos_Estimator, Target_Pos_Estimator_No_Forward
from components.goal import Goal
from components.measurement_model import ImplicitMeasurementModel, Robot_Vel_MM, Target_Dist_MM, Pos_MM
from environment.gaze_fix_env import GazeFixEnv
from run_env import pick_action

# ==================================================================================================

render_polar = False

# ===============================================================================

# TODO: why is cartesian cov so big?
def polar_to_cartesian_state(polar_mean, polar_cov):
    # Transform target_dist_estimator state mean and covariance from polar to cartesian coordinates
    theta = polar_mean[0]
    r = polar_mean[1]
    x_mean = r * torch.cos(theta)

    print("COS: ", torch.cos(theta))

    y_mean = r * torch.sin(theta)
    J = torch.tensor([
        [torch.cos(theta+np.pi/2), -r * torch.sin(theta+np.pi/2)],
        [torch.sin(theta+np.pi/2), r * torch.cos(theta+np.pi/2)]
    ], dtype=torch.float32).to(DEVICE)
    cartesian_cov = J @ polar_cov[:2, :2] @ J.T
    cartesian_mean = torch.tensor([x_mean, y_mean], dtype=torch.float32).to(DEVICE)
    return cartesian_mean, cartesian_cov

# class PosToDistMM(ImplicitMeasurementModel):
#     def __init__(self, device) -> None:
#         super().__init__(4, {'Target-Dist-Estimator': 4}, device)
#     def implicit_measurement_model(self, x, meas_dict):
#         return torch.stack([
#             meas_dict["Target-Dist-Estimator"][0] - torch.atan2(x[0], x[1]),
#             meas_dict["Target-Dist-Estimator"][1] - torch.norm(x[:2])
#         ]).squeeze()
    
# class DistToPosMM(ImplicitMeasurementModel):
#     def __init__(self, device) -> None:
#         super().__init__(4, {'Target-Pos-Estimator': 4}, device)
#     def implicit_measurement_model(self, x, meas_dict):
#         return torch.stack([
#             meas_dict["Target-Pos-Estimator"][0] - torch.cos(x[0])*x[1],
#             meas_dict["Target-Pos-Estimator"][1] - torch.sin(x[0])*x[1]
#         ]).squeeze()

# class AI1(AI):
#     """
#     Active Interconnection between Target Cartesian Position Estimator and Target Polar Position Estimator
#     """
#     def __init__(self):
#         super().__init__("AI1")
#         self.add_estimator(target_pos_estimator, PosToDistMM(DEVICE))
#         self.add_estimator(target_dist_estimator, DistToPosMM(DEVICE))
        

# class GoToTargetGoal(Goal):
#     def __init__(self, estimator: RecursiveEstimator):
#         super().__init__(estimator, torch.zeros_like(estimator.state_mean))

#     def loss_function(self):
#         return (self.current_state - self.desired_state).pow(2).sum()


# ==================================================================================================

if __name__ == "__main__":

    DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")


    print("================ Running AICON test ======================")
    print("Creating environment ...")

    with open('config/env_config.yaml') as file:
        env_config = yaml.load(file, Loader=yaml.FullLoader)
    env = GazeFixEnv(env_config)

    print("Creating AICON object ...")

    #aicon = AICON(env)

    print("Adding estimators ...")

    robot_vel_estimator = Robot_Vel_Estimator(DEVICE)
    robot_vel_measurement_model = Robot_Vel_MM(DEVICE)
    # --------------------------------------------------
    target_pos_estimator = Target_Pos_Estimator(DEVICE)
    target_pos_estimator.set_state(
        mean = torch.tensor([20.0, 0.0], dtype=torch.float32).to(DEVICE),
        cov = 1000 * torch.eye(2, dtype=torch.float32).to(DEVICE)
    )
    target_pos_measurement_model = Pos_MM(DEVICE)
    # --------------------------------------------------
    target_pos_estimator_no_forward = Target_Pos_Estimator_No_Forward(DEVICE)
    target_pos_estimator_no_forward.set_state(
        mean = torch.tensor([20.0, 0.0], dtype=torch.float32).to(DEVICE),
        cov = 1000 * torch.eye(2, dtype=torch.float32).to(DEVICE)
    )
    # --------------------------------------------------
    target_dist_estimator = Target_Distance_Estimator(DEVICE)
    target_dist_estimator.set_state(
        mean = torch.tensor([0.0, 20.0], dtype=torch.float32).to(DEVICE),
        cov = 1000 * torch.eye(2, dtype=torch.float32).to(DEVICE)
    )
    target_dist_measurement_model = Target_Dist_MM(DEVICE)
    
    # aicon.add_estimator(robot_vel_estimator)
    # aicon.add_estimator(target_pos_estimator)
    # aicon.add_estimator(target_dist_estimator)

    # print("Adding active interconnections ...")

    # ai1 = AI1()
    # aicon.add_interconnection(ai1)

    # print("Adding goals ...")

    # goal = GoToTargetGoal(target_dist_estimator)

    print("Running AICON ...")

    obs, info = env.reset(5)
    done = False
    step = 0
    env.render(1.0, np.array(target_pos_estimator.state_mean.cpu()), np.array(target_pos_estimator.state_cov.cpu()))
    input("Press Enter to continue...\n")

    while (not done) and step < 1000:
        action = pick_action(6, step, obs, target_pos_estimator)
        #action = goal.pick_action()
        #print(f"Action: {action}")

        obs, reward, done, truncated, info = env.step(action)

        # ------------------------------------------------------------------------------------------------------------------------------
        
        u_robot_vel = torch.tensor([
            action[0]*env.robot.max_acc,
            action[1]*env.robot.max_acc,
            action[2]*env.robot.max_acc_rot,
            env.timestep,
        ], dtype=torch.float32).to(DEVICE)
        u_target_pos1 = torch.stack([
            robot_vel_estimator.state_mean[0],
            robot_vel_estimator.state_mean[1],
            robot_vel_estimator.state_mean[2],
            torch.tensor(env.timestep, dtype=torch.float32, device=DEVICE),
        ]).squeeze()
        u_target_pos2 = torch.stack([
            torch.tensor(env.robot.vel[0], dtype=torch.float32, device=DEVICE),
            torch.tensor(env.robot.vel[1], dtype=torch.float32, device=DEVICE),
            torch.tensor(env.robot.vel_rot, dtype=torch.float32, device=DEVICE),
            torch.tensor(env.timestep, dtype=torch.float32, device=DEVICE),
        ]).squeeze()

        if torch.equal(u_target_pos1, u_target_pos2):
            print("EQUAL")
        else:
            print("NOT EQUAL")
            print(u_target_pos1)
            print(u_target_pos2)

        u_target_dist = torch.stack([
            torch.tensor(env.robot.vel[0], dtype=torch.float32, device=DEVICE),
            torch.tensor(env.robot.vel[1], dtype=torch.float32, device=DEVICE),
            torch.tensor(env.robot.vel_rot, dtype=torch.float32, device=DEVICE),
            torch.tensor(env.timestep, dtype=torch.float32, device=DEVICE),
        ]).squeeze()

        robot_vel_estimator.call_predict(u_robot_vel)
        target_pos_estimator.call_predict(u_target_pos1)
        target_dist_estimator.call_predict(u_target_dist)
        target_pos_estimator_no_forward.call_predict(u_target_pos1)

        target_pos_estimator.load_state_dict(target_pos_estimator.buffer_dict)
        target_dist_estimator.load_state_dict(target_dist_estimator.buffer_dict)

        if render_polar:
            rendering_mean, rendering_cov = polar_to_cartesian_state(target_dist_estimator.state_mean, target_dist_estimator.state_cov)
            env.render(1.0, np.array(rendering_mean.cpu()), np.array(rendering_cov.cpu()))
        else:
            env.render(1.0, np.array(target_pos_estimator.state_mean.cpu()), np.array(target_pos_estimator.state_cov.cpu()))
        input("Pre Measurement update")

        # ------------------------------------------------------------------------------------------------------------------------------
        
        target_offset_angle = torch.tensor([obs[1]], dtype=torch.float32).to(DEVICE)
        del_target_offset_angle = torch.tensor([obs[2]], dtype=torch.float32).to(DEVICE)
        robot_vel_env = torch.tensor([obs[4]*env.robot.max_vel, obs[5]*env.robot.max_vel, obs[3]*env.robot.max_vel_rot], dtype=torch.float32).to(DEVICE)
        robot_vel_estimator.load_state_dict(robot_vel_estimator.buffer_dict)
        robot_vel = robot_vel_estimator.state_mean

        robot_vel_estimator.call_update_with_specific_meas(robot_vel_measurement_model, {'robot_vel': robot_vel_env})
        robot_vel_estimator.load_state_dict(robot_vel_estimator.buffer_dict)
        target_pos_estimator.call_update_with_specific_meas(target_pos_measurement_model, {'target_offset_angle': target_offset_angle})
        target_pos_estimator.load_state_dict(target_pos_estimator.buffer_dict)
        target_pos_estimator_no_forward.call_update_with_specific_meas(target_pos_measurement_model, {'target_offset_angle': target_offset_angle})
        target_pos_estimator_no_forward.load_state_dict(target_pos_estimator_no_forward.buffer_dict)
        target_dist_estimator.call_update_with_specific_meas(target_dist_measurement_model, {'robot_vel': robot_vel, 'target_offset_angle': target_offset_angle, 'del_target_offset_angle': del_target_offset_angle})
        target_dist_estimator.load_state_dict(target_dist_estimator.buffer_dict)

        #ai1.update_estimator(target_pos_estimator.id)
        #ai1.update_estimator(target_dist_estimator.id)
        #target_pos_estimator.set_buffer_dict()
        #target_dist_estimator.set_buffer_dict()

        print(f"======================================= Step {step} ==============================================")
        print(f"Robot Vel Estimator State:       {robot_vel_estimator.state_mean.tolist()}")
        print(f"Actual Robot Vel:                {robot_vel_env.tolist()}")
        #print(f"Robot Vel Estimate Covariance:\n{robot_vel_estimator.state_cov}")
        # print("----------------------------------------------------------------------------------------------")
        # print(f"Target Polar Estimator State:    {target_dist_estimator.state_mean.tolist()}")
        # print(f"Actual Target Coordinates:       {[obs[1], obs[6]]}")
        # print(f"Target Polar Estimate Covariance:\n{target_dist_estimator.state_cov}")
        print("----------------------------------------------------------------------------------------------")
        print(f"Target Pos Estimator State:      {target_pos_estimator.state_mean.tolist()}")
        print(f"Actual Target Pos:               {env.rotation_matrix(-env.robot.orientation) @ (env.target.pos - env.robot.pos)}")
        print(f"Target Pos Estimate Covariance:\n{target_pos_estimator.state_cov}")
        # print("----------------------------------------------------------------------------------------------")
        # print(f"Target Pos Estimator No Forward State: {target_pos_estimator_no_forward.state_mean.tolist()}")
        # print(f"Actual Target Pos:               {env.rotation_matrix(-env.robot.orientation) @ (env.target.pos - env.robot.pos)}")
        # print(f"Target Pos Estimate Covariance:\n{target_pos_estimator_no_forward.state_cov}")

        # ------------------------------------------------------------------------------------------------------------------------------

        step += 1

        if render_polar:
            rendering_mean, rendering_cov = polar_to_cartesian_state(target_dist_estimator.state_mean, target_dist_estimator.state_cov)
            print(f"Rendering Mean: {rendering_mean.tolist()}")
            print(f"Rendering Cov: {rendering_cov.tolist()}")
            env.render(1.0, np.array(rendering_mean.cpu()), np.array(rendering_cov.cpu()))
        else:
            env.render(1.0, np.array(target_pos_estimator.state_mean.cpu()), np.array(target_pos_estimator.state_cov.cpu()))

        # for step by step debugging
        input("Press Enter to continue...\n")
    env.close()
    obs, info = env.reset()


    print("================ Finished AICON test ======================")