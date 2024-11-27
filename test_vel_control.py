import numpy as np
import torch
import yaml
from components.active_interconnection import Angle_Meas_AI, Pos_Angle_AI, Triangulation_AI
from components.aicon import AICON
from components.estimator import Polar_Pos_Estimator_External_Vel, Pos_Estimator_External_Vel, Robot_Vel_Estimator
from components.goal import GazeFixationGoal, GoToTargetGoal, PolarGoToTargetGoal
from environment.gaze_fix_env import GazeFixEnv

# ==================================================================================

class TestAICON(AICON):
    def __init__(self):
        super().__init__()

        config = 'config/env_config.yaml'
        with open(config) as file:
            env_config = yaml.load(file, Loader=yaml.FullLoader)
            env_config["action_mode"] = 3
        self.set_env(GazeFixEnv(env_config))

        estimators = {
            "RobotVel": Robot_Vel_Estimator(device=self.device),
            "TargetPos": Pos_Estimator_External_Vel(device=self.device, id='TargetPos'),
            "PolarTargetPos": Polar_Pos_Estimator_External_Vel(device=self.device, id='PolarTargetPos'),
        }
        self.set_estimators(estimators)

        active_interconnections = {
            "CartPos" : Pos_Angle_AI([estimators["TargetPos"], self.obs["target_offset_angle"]], self.device, estimate_vel=False),
            #"PolarNoncart": Polar_Angle_NonCart_AI([self.REs["PolarTargetPos"], self.REs["RobotVel"], self.obs["target_offset_angle"], self.obs["del_target_offset_angle"]], self.device),
            "PolarAngle": Angle_Meas_AI([self.REs["PolarTargetPos"], self.obs["target_offset_angle"]], self.device, estimate_vel=False),
            "PolarDistance": Triangulation_AI([self.REs["PolarTargetPos"], self.REs["RobotVel"], self.obs["del_target_offset_angle"]], self.device, estimate_vel=False),
        }
        self.set_active_interconnections(active_interconnections)

        self.set_goals({
            "GazeFixation": GazeFixationGoal(device=self.device),
            "PolarGoToTarget": PolarGoToTargetGoal(device=self.device),
            "GoToTarget": GoToTargetGoal(device=self.device)
        })

        self.reset()

    def reset(self):
        # self.REs["RobotVel"].set_state(torch.tensor([0.0, 0.0, 0.0]), torch.eye(3) * 0.01)
        # self.REs["RobotVel"].set_static_motion_noise(torch.eye(3) * 0.01)
        self.REs["TargetPos"].set_state(torch.tensor([10.0, 0.0]), torch.eye(2) * 1e3)
        #self.REs["TargetPos"].set_state(torch.tensor([26.126, 0.000]), torch.eye(2) * 0.01)
        self.REs["TargetPos"].set_static_motion_noise(torch.eye(2) * 0.01)
        self.REs["PolarTargetPos"].set_state(torch.tensor([10.0, 0.0]), torch.eye(2) * 1e3)
        #self.REs["PolarTargetPos"].set_state(torch.tensor([26.126, 0.000]), torch.eye(2) * 0.01)
        self.REs["PolarTargetPos"].set_static_motion_noise(torch.eye(2) * 0.01)

    def eval_predict(self, action, buffer_dict):
        vel = action * torch.tensor([self.env.robot.max_vel, self.env.robot.max_vel, self.env.robot.max_vel_rot])
        timestep = torch.tensor([0.05], device=self.device)
        self.REs["PolarTargetPos"].call_predict(torch.concat([vel, timestep]), buffer_dict)
        self.REs["TargetPos"].call_predict(torch.concat([vel, timestep]), buffer_dict)
        buffer_dict["RobotVel"]["state_mean"] = vel
        buffer_dict["RobotVel"]["state_cov"] = torch.eye(3) * 1e-3
        return buffer_dict

    def eval_step(self, action, new_step = False):
        self.update_observations()
        buffer_dict = {key: estimator.set_buffer_dict() for key, estimator in list(self.REs.items()) + list(self.obs.items())}

        if not new_step:
            return self.eval_predict(action, buffer_dict)

        vel = action * torch.tensor([self.env.robot.max_vel, self.env.robot.max_vel, self.env.robot.max_vel_rot])
        timestep = torch.tensor([0.05], device=self.device)

        if new_step:
            print(
                f"------------------- Pre Predict: -------------------\n",
                f"Pos: {buffer_dict['TargetPos']['state_mean']}\n",
                #f"Polar Pos: {buffer_dict['PolarTargetPos']['state_mean']}",
            )

        self.REs["PolarTargetPos"].call_predict(torch.concat([vel, timestep]), buffer_dict)
        self.REs["TargetPos"].call_predict(torch.concat([vel, timestep]), buffer_dict)

        buffer_dict["RobotVel"]["state_mean"] = vel
        buffer_dict["RobotVel"]["state_cov"] = torch.eye(3) * 1e-3

        if new_step:
            print(
                f"------------------- Post Predict: -------------------\n",
                f"Pos: {buffer_dict['TargetPos']['state_mean']}\n",
                #f"Polar Pos: {buffer_dict['PolarTargetPos']['state_mean']}",
            )

        self.REs["TargetPos"].call_update_with_specific_meas(self.AIs["CartPos"], buffer_dict)
        self.REs["PolarTargetPos"].call_update_with_specific_meas(self.AIs["PolarAngle"], buffer_dict)
        self.REs["PolarTargetPos"].call_update_with_specific_meas(self.AIs["PolarDistance"], buffer_dict)

        if new_step:
            print(
                f"------------------ Post Measurement: ------------------\n",
                f"Pos: {buffer_dict['TargetPos']['state_mean']}\n",
                #f"Polar Pos: {buffer_dict['PolarTargetPos']['state_mean']}",
            )

        return buffer_dict
    
    def render(self):
        # estimator_means = {key: np.array(self.REs[key].state_mean.cpu()) for key in ["TargetPos"] + [f"Obstacle{i}Pos" for i in range(1, self.num_obstacles + 1)] if key in self.REs.keys()}
        # estimator_covs = {key: np.array(self.REs[key].state_cov.cpu()) for key in ["TargetPos"] + [f"Obstacle{i}Pos" for i in range(1, self.num_obstacles + 1)] if key in self.REs.keys()}
        estimator_means = {"PolarTargetPos": np.array(torch.stack([
            self.REs["PolarTargetPos"].state_mean[0] * torch.cos(self.REs["PolarTargetPos"].state_mean[1]),
            self.REs["PolarTargetPos"].state_mean[0] * torch.sin(self.REs["PolarTargetPos"].state_mean[1])
        ]).cpu())}
        cart_cov = torch.zeros((2, 2), device=self.device)
        cart_cov[0, 0] = self.REs["PolarTargetPos"].state_cov[0, 0] * torch.cos(self.REs["PolarTargetPos"].state_mean[1])**2 + self.REs["PolarTargetPos"].state_cov[1, 1] * torch.sin(self.REs["PolarTargetPos"].state_mean[1])**2
        cart_cov[0, 1] = (self.REs["PolarTargetPos"].state_cov[0, 0] - self.REs["PolarTargetPos"].state_cov[1, 1]) * torch.cos(self.REs["PolarTargetPos"].state_mean[1]) * torch.sin(self.REs["PolarTargetPos"].state_mean[1])
        cart_cov[1, 0] = cart_cov[0, 1]
        cart_cov[1, 1] = self.REs["PolarTargetPos"].state_cov[0, 0] * torch.sin(self.REs["PolarTargetPos"].state_mean[1])**2 + self.REs["PolarTargetPos"].state_cov[1, 1] * torch.cos(self.REs["PolarTargetPos"].state_mean[1])**2
        estimator_covs = {"PolarTargetPos": np.array(cart_cov.cpu())}
        return self.env.render(1.0, estimator_means, estimator_covs)

    def compute_action(self, gradients):
        """
        CAN be implemented by user. Computes the action based on the gradients
        """
        #return self.last_action - gradients["GoToTarget"]
        #return self.last_action - gradients["PolarGoToTarget"]
        return -gradients["PolarGoToTarget"]
    
    def print_states(self):
        obs = self.env._get_observation()
        print("==========================================")
        self.print_state("PolarTargetPos", False)
        print(f"True PolarTargetPos: [{obs['robot_target_distance']:.3f}, {obs['target_offset_angle']:.3f}]")
        print("------------------------------------------")
        self.print_state("TargetPos", False)
        actual_pos = list(self.env.rotation_matrix(-self.env.robot.orientation) @ (self.env.target.pos - self.env.robot.pos))
        print(f"True Target Pos: {[f'{x:.3f}' for x in actual_pos]}")
        print("==========================================")

# ==================================================================================

if __name__ == "__main__":

    aicon = TestAICON()

    # aicon.env.reset(seed=10)
    # # set robot manually to face target
    # aicon.env.robot.orientation = -1.008

    # aicon.step(torch.tensor([1.0, 0.0, 0.0], device=aicon.device))

    # # aicon.print_states()

    # aicon.env.render()

    # jacobian, step_eval = aicon.compute_estimator_action_gradient("PolarTargetPos", torch.tensor([1.0, 1.0, 1.0], device=aicon.device))
    # print("New State: ", step_eval["state_mean"].tolist())
    # print("Polar Pos Jacobian: ", jacobian["state_mean"].tolist())

    # jacobian = aicon.compute_goal_action_gradient(aicon.goals["PolarGoToTarget"])
    # print("Goal Jacobian: ", jacobian.tolist())

    # print("=================================================================================")

    # jacobian, step_eval = aicon.compute_estimator_action_gradient("TargetPos", torch.tensor([0.0, 0.0, 0.0], device=aicon.device))
    # print("New State: ", step_eval["state_mean"].tolist())
    # print("Pos Jacobian: ", jacobian["state_mean"].tolist())

    # input("press Enter to finish test...")

    aicon.run(2500, 10, render=True, prints=1, step_by_step=True, record_video=False)

