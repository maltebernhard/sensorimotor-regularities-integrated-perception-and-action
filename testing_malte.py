import numpy as np
import torch
import yaml

from aravind.filters import ImplicitMeasurementModel, MIMEKF
from environment.gaze_fix_env import GazeFixEnv

# ==================================================================================================

# measurement model between robot velocity state and robot_velocity measurement
class Robot_Vel_MM(ImplicitMeasurementModel):
    def __init__(self) -> None:
        super().__init__(3, {"vel_rot" : 1, "vel_frontal" : 1, "vel_lateral" : 1})
    
    def implicit_measurement_model(self, x, meas_dict):
        h_of_x = self.explicit_measurement_model(x)
        meas_tensor = torch.tensor([meas_dict[key] for key in self.measurement_config.keys()])
        return meas_tensor - h_of_x
    
    def explicit_measurement_model(self, x):
        z = x
        return z

# measurement model between target position state and angular offset measurement
class Target_Pos_MM(ImplicitMeasurementModel):
    def __init__(self) -> None:
        super().__init__(2, {'target_offset_angle': 1})
    
    def implicit_measurement_model(self, x, meas_dict):
        h_of_x = self.explicit_measurement_model(x)
        #print(f"Real angle: {meas_dict['target_offset_angle'][0]} | Expected angle: {h_of_x}")
        return meas_dict['target_offset_angle'] - h_of_x
    
    def explicit_measurement_model(self, x):
        z = torch.atan2(x[1],x[0])
        return z

class Target_Distance_MMEKF(MIMEKF):
    def __init__(self):
        super().__init__(2)

    def motion_model(self, x_mean, u):
        ret_mean = torch.empty_like(x_mean)
        #ret_cov = torch.empty_like(x_cov)
        vel_trans = u[:2]
        vel_rot = u[2]
        delta_t = u[3]
        theta = vel_rot * delta_t
        rotation_matrix = torch.tensor([
            [torch.cos(-theta), -torch.sin(-theta)],
            [torch.sin(-theta), torch.cos(-theta)]
        ]).to(x_mean.device)
        ret_mean = torch.matmul(rotation_matrix, x_mean - vel_trans*delta_t)
        #ret_cov = torch.matmul(rotation_matrix, torch.matmul(x_cov, rotation_matrix.T))
        return ret_mean
        #return ret_mean, ret_cov

# ==================================================================================================

def print_state(msg, state):
    if type(state) == tuple:
        print(msg)
        print(state[0])
        print(state[1])
    else:
        print(msg)
        print(state)

def run_test():
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    # DEVICE = torch.device("cpu") # Force CPU
    ###########################################################

    target_pos_measurement_model = Target_Pos_MM().to(device)
    target_distance_estimator = Target_Distance_MMEKF().to(device)

    # for motion model: vel_frontal, vel_lateral, vel_rot, del_t
    us = torch.tensor([
        [1., 1., 1., 1., 1.],
        [1., 1., 1., 1., 1.],
        [0., 0., 0., 0., 0.],
        [0.1, 0.1, 0.1, 0.1, 0.1]
    ]).T.to(device)
    # for measurement model: angular offsets to target
    z1s = torch.tensor([0.0, 0.0, 0.0, 0.0, 0.0]).to(device)

    print_state('Multiple measurements state init:', target_distance_estimator.state)
    print("==============================================")

    for u, z1 in zip(us, z1s):
        print(f"u: {u}\ntype: {u.dtype}")
        target_distance_estimator.predict(u)
        
        print_state("Before Measurement", target_distance_estimator.state)
        target_distance_estimator.update_with_specific_meas({'target_offset_angle': z1}, target_pos_measurement_model)
        print_state("After Measurement", target_distance_estimator.state)
        print("==============================================")

# ==================================================================================================

def run_env(version):

    with open('config/env_config.yaml') as file:
        env_config = yaml.load(file, Loader=yaml.FullLoader)
    env = GazeFixEnv(env_config)

    ###########################################################
    # Example usages works with PyTorch 2.0

    DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    # DEVICE = torch.device("cpu") # Force CPU
    ###########################################################

    target_pos_measurement_model = Target_Pos_MM().to(DEVICE)
    target_distance_estimator = Target_Distance_MMEKF().to(DEVICE)

    obs, info = env.reset(5)
    done = False
    step = 0
    while (not done) and step<1000:
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
        else:
            action = env.action_space.sample()
        #print(f"Action: {action}")
        u = torch.tensor([action[0]*env.robot.max_vel, action[1]*env.robot.max_vel, action[2]*env.robot.max_vel_rot, env.timestep], dtype=torch.float32).to(DEVICE)
        target_distance_estimator.predict(u)
        obs, reward, done, truncated, info = env.step(action)
        z1 = torch.tensor([obs[1]], dtype=torch.float32).to(DEVICE)
        target_distance_estimator.update_with_specific_meas({'target_offset_angle': z1}, target_pos_measurement_model)
        step += 1
        env.render(1.0, np.array(target_distance_estimator.state_mean.cpu()), np.array(target_distance_estimator.state_cov.cpu()))
        # for step by step debugging
        input("Press Enter to continue...")
    env.close()
    obs, info = env.reset()

    print("END multiple measurements test with MIMEKF")

# ==================================================================================================

if __name__ == '__main__':
    run_env(3)