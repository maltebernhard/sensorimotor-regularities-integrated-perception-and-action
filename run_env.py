import numpy as np
import torch
import yaml

from environment.gaze_fix_env import GazeFixEnv

from components.estimator import Robot_Vel_Estimator, Pos_Estimator_Internal_Vel
from components.measurement_model import Robot_Vel_MM, Pos_MM

# ==================================================================================================

def pick_action(version, step, obs, target_distance_estimator):
    if version == 1:
        action = np.array([1.0, 0.0, 0.0])
    elif version == 2:
        action = np.array([1.0, 0.0, 0.1])
    elif version == 3:
        if step < 50:
            action = np.array([1.0, 0.0, 0.0])
        else:
            action = np.array([1.0, 0.0, 0.2])
    elif version == 4:
        rot = obs[1] * 1.0
        action = np.array([1.0, 0.0, rot])
    elif version == 5:
        if step < 50:
            action = np.array([1.0, 0.0, 0.0])
        else:
            rot = obs[1] * 1.0
            action = np.array([target_distance_estimator.state_mean[0].cpu(), target_distance_estimator.state_mean[1].cpu(), rot])
    elif version == 6:
        if step < 50:
            action = np.array([1.0, 0.0, 0.0])
        rot = ((obs[1] * 1.0) - obs[3]) * 1.0
        action = np.array([1.0, -1.0, rot])
    else:
        raise ValueError(f"Invalid version: {version}")
    if np.linalg.norm(action) > 1.0:
        action = action / np.linalg.norm(action)
    return action

# ==================================================================================================

def run_env(version):
    # ------------------------------------------------------------------
    DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    # DEVICE = torch.device("cpu") # Force CPU
    # ------------------------------------------------------------------

    with open('config/env_config.yaml') as file:
        env_config = yaml.load(file, Loader=yaml.FullLoader)
    env = GazeFixEnv(env_config)

    target_pos_estimator = Pos_Estimator_Internal_Vel().to(DEVICE)
    target_pos_estimator.set_state(torch.tensor([20.0, 0.0], dtype=torch.float64).to(DEVICE), 1000 * torch.eye(2, dtype=torch.float64).to(DEVICE))
    target_pos_estimator.set_static_motion_noise(0.01 * torch.eye(2, dtype=torch.float64).to(DEVICE))
    target_pos_measurement_model = Pos_MM().to(DEVICE)

    robot_vel_estimator = Robot_Vel_Estimator().to(DEVICE)
    robot_vel_measurement_model= Robot_Vel_MM().to(DEVICE)

    obs, info = env.reset(5)
    done = False
    step = 0

    env.render(1.0, np.array(target_pos_estimator.state_mean.cpu()), np.array(target_pos_estimator.state_cov.cpu()))
    input("Press Enter to continue...\n")

    while (not done) and step<1000:
        action = pick_action(version, step, obs, target_pos_estimator)
        #print(f"Action: {action}")

        # ------------------------------------------------------------------------------------------------------------------------------
        u_robot_vel = torch.tensor([
            action[0]*env.robot.max_acc,
            action[1]*env.robot.max_acc,
            action[2]*env.robot.max_acc_rot,
            env.timestep,
            env.robot.max_vel,
            env.robot.max_vel_rot
        ], dtype=torch.float64).to(DEVICE)
        u_target_pos = torch.tensor([
            obs[4]*env.robot.max_vel,
            obs[5]*env.robot.max_vel,
            obs[3]*env.robot.max_vel_rot,
            env.timestep
        ], dtype=torch.float64).to(DEVICE)
        robot_vel_estimator.predict(u_robot_vel)
        target_pos_estimator.predict(u_target_pos)

        # ------------------------------------------------------------------------------------------------------------------------------

        obs, reward, done, truncated, info = env.step(action)

        # ------------------------------------------------------------------------------------------------------------------------------
        target_offset_angle = torch.tensor([obs[1]], dtype=torch.float64).to(DEVICE)
        robot_vel = torch.tensor([obs[4]*env.robot.max_vel, obs[5]*env.robot.max_vel, obs[3]*env.robot.max_vel_rot], dtype=torch.float64).to(DEVICE)
        target_pos_estimator.update_with_specific_meas({'target_offset_angle': target_offset_angle}, target_pos_measurement_model)
        robot_vel_estimator.update_with_specific_meas({'vel_frontal': robot_vel[0], 'vel_lateral': robot_vel[1], 'vel_rot': robot_vel[2]}, robot_vel_measurement_model)
        print(f"Robot Vel Estimator State:       {robot_vel_estimator.state_mean.tolist()}")
        print(f"Actual Robot Vel:                {robot_vel.tolist()}")
        #print(f"Robot Vel Estimate Covariance:\n{robot_vel_estimator.state_cov}")
        print("----------------------------------------------------------")
        print(f"Target Distance Estimator State: {target_pos_estimator.state_mean.tolist()}")
        print(f"Actual Target Distance:          {env.rotation_matrix(-env.robot.orientation) @ (env.target.pos - env.robot.pos)}")
        #print(f"Target Distance Estimate Covariance:\n{target_distance_estimator.state_cov}")
        print(f"================================= Step {step+1} ========================================")
        # ------------------------------------------------------------------------------------------------------------------------------

        step += 1
        env.render(1.0, np.array(target_pos_estimator.state_mean.cpu()), np.array(target_pos_estimator.state_cov.cpu()))
        # for step by step debugging
        input("Press Enter to continue...\n")
    env.close()
    obs, info = env.reset()

    print("END multiple measurements test with MIMEKF")

# ==================================================================================================

if __name__ == '__main__':
    run_env(6)