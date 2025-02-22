import numpy as np
import torch
from torch.func import jacrev
from abc import ABC, abstractmethod
from typing import Dict, List, Tuple

from components.estimator import Observation, RecursiveEstimator
from components.logger import VariationLogger
from components.measurement_model import ActiveInterconnection
from components.goal import Goal
from environment.base_env import BaseEnv
from environment.gaze_fix_env import GazeFixEnv

# ========================================================================================

class AICON(ABC):
    def __init__(self):
        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        self.dtype = torch.float64
        torch.set_default_device(self.device)
        torch.set_default_dtype(self.dtype)
        self.env: GazeFixEnv = self.define_env()
        self.REs: Dict[str, RecursiveEstimator] = self.define_estimators()
        self.MMs: Dict[str, Tuple[ActiveInterconnection,List[str]]] = self.define_measurement_models()
        self.AIs: Dict[str, ActiveInterconnection] = self.define_active_interconnections()
        self.obs: Dict[str, Observation] = self.set_observations()
        self.set_static_sensor_noises()
        self.connect_states()
        self.goals: Dict[str, Goal] = self.define_goals()
        self.last_action: torch.Tensor = torch.tensor([0.0, 0.0, 0.0])
        self.prints = 0
        self.current_step = 0

    def run(self, timesteps, env_seed=0, initial_action=None, render=True, prints=0, step_by_step=True, logger:VariationLogger=None, video_path=None):
        """
        Runs AICON on the environment for a given number of timesteps
        Args:
            timesteps:    number of timesteps to run
            env_seed:     seed for the environment
            render:       render the environment
            prints:       print the states every "prints" timesteps
            step_by_step: wait for user input after each step
        """
        assert self.env is not None, "Environment not set"
        assert self.REs is not None, "Estimators not set"
        assert self.MMs is not None, "Measurement Models not set"
        assert self.AIs is not None, "Active Interconnections not set"
        assert self.goals is not None, "Goals not set"
        self.reset(seed=env_seed, video_path=video_path)
        if initial_action is not None:
            self.last_action = initial_action
        self.update_observations()
        buffer_dict = self.get_buffer_dict()
        self.prints = prints
        self.meas_updates(buffer_dict)
        self.eval_interconnections(buffer_dict)
        for key, state_buffer_dict in buffer_dict.items():
            self.REs[key].load_state_dict(state_buffer_dict) if key in self.REs.keys() else None
        if prints > 0:
            print(f"======================= Initial State ===========================")
            self.print_estimators()
        if render:
            self.render()
        for step in range(timesteps):
            self.current_step = step
            if prints > 0 and step % prints == 0:
                print("------------ Computing Action Gradient -------------")
            gradients, action = self.compute_action()
            env_state = self.env.get_state()
            if logger is not None:
                buffer_dict = self.get_buffer_dict()
                logger.log(
                    step = step,
                    time = self.env.time,
                    estimators = {key: val for key, val in buffer_dict.items() if key in self.REs.keys()},
                    env_state = env_state,
                    observation = {key: {"measurement": self.last_observation[0][key], "noise": self.last_observation[1][key]} for key in self.last_observation[0].keys()},
                    goal_loss = {key: goal.loss_function_from_buffer(buffer_dict) for key, goal in self.goals.items()},
                    action = action,
                    gradients = gradients,
                )
            if any([val==1.0 for key, val in env_state.items() if "collision" in key]):
                if prints > 0:
                    print("Collision detected. Terminating run.")
                break
            if step_by_step:
                input("Press Enter to continue...")
            for obs in self.obs.values():
                obs.updated = False
            self.step(action)
            if render:
                self.render()
            step += 1
            if prints > 0 and step % prints == 0:
                print(f"============================ Step {step} ================================")
                self.print_estimators()
        logger.end_wandb_run() if logger is not None else None
        self.reset()

    def step(self, action):
        if action[:2].norm() > 1.0:
            action[:2] = action[:2] / action[:2].norm()
        if torch.abs(action[2]) > 1.0:
            action[2] = action[2]/torch.abs(action[2])
        self.last_action = action
        #print(f"-------- Action: ", end=""), self.print_vector(action, trail=" --------")
        self.env.step(np.array(action.cpu()))
        buffers = self.eval_step(action, new_step=True)
        for key, buffer_dict in buffers.items():
            self.REs[key].load_state_dict(buffer_dict) if key in self.REs.keys() else None

    def eval_step(self, action, new_step = False) -> Dict[str, Dict[str, torch.Tensor]]:
        """
        Evaluates one step of the environment and returns the buffer_dict
        """
        if new_step:
            self.update_observations()
        buffer_dict = self.get_buffer_dict()

        # estimator forward models
        for est_key, estimator in self.REs.items():
            u = self.get_control_input(action, buffer_dict, est_key)
            estimator.call_predict(u, buffer_dict)

        # measurement updates
        if new_step:
            self.meas_updates(buffer_dict)
        else:
            self.contingent_meas_updates(buffer_dict)

        # interconnection updates
        return self.eval_interconnections(buffer_dict)

    def meas_updates(self, buffer_dict):
        if self.prints > 0 and self.current_step % self.prints == 0:
            real_state = self.env.get_state()
            print("Real State: ", [f"{key}: " + (f"{val:.3f}" if val is not None else "{None}") for key, val in real_state.items() if key in self.last_observation[0].keys()])
            print("Measurements: ", [f"{key}: " + (f"{val:.3f}" if val is not None else "{None}") for key, val in self.last_observation[0].items()])
            print("Pre Real Meas Target: "), self.print_estimator("PolarTargetPos", print_cov=2, buffer_dict=buffer_dict)
            print("Pre Real Meas Robot: "), self.print_estimator("RobotVel", print_cov=2, buffer_dict=buffer_dict)
        for mm_key, (meas_model, estimator_keys) in self.MMs.items():
            if meas_model.all_observations_updated():
                for estimator_key in estimator_keys:
                    # if self.prints > 0 and self.current_step % self.prints == 0:
                    #     esti_offsets, esti_stddevs = meas_model.get_cov_dict(buffer_dict, estimator_key)
                    #     real_noise = {obs_key: (mean, stddev) for obs_key, (mean, stddev) in self.last_observation[1].items() if obs_key in meas_model.connected_observations.keys()}
                    #     esti_noise = {obs_key: (mean, esti_stddevs[obs_key]) for obs_key, mean in esti_offsets.items() if obs_key in meas_model.connected_observations.keys()}
                    #     print(f"---------------- {mm_key} {estimator_key} ----------------")
                    #     print(f"Real Noise (mean,stddev): {real_noise}")
                    #     print(f"Estimated Noise (mean,stddev): {esti_noise}")
                    self.REs[estimator_key].call_update_with_active_interconnection(meas_model, buffer_dict)
                    if self.prints > 0 and self.current_step % self.prints == 0:
                        print(f"Post Real {mm_key} {estimator_key}: "), self.print_estimator(estimator_key, print_cov=2, buffer_dict=buffer_dict)
                        if abs(buffer_dict["PolarTargetPos"]["mean"][2].item()) > 3e+4:
                            raise ValueError("Target distance is too large. Something is wrong.")

    def contingent_meas_updates(self, buffer_dict: dict):
        """
        Updates the estimators with the expected measurements AND (important!) expected measurement noise. SHOULD be overwritten by user.
        """
        if self.prints > 0 and self.current_step % self.prints == 0:
            print("Pre Cont Meas Target: "), self.print_estimator("PolarTargetPos", print_cov=2, buffer_dict=buffer_dict)
            #print("Pre Cont Meas Robot: "), self.print_estimator("RobotVel", print_cov=2, buffer_dict=buffer_dict)
        for mm_key, (meas_model, estimator_keys) in self.MMs.items():
            if meas_model.all_observations_updated():
                for estimator_key in estimator_keys:
                    self.REs[estimator_key].call_update_with_smc(meas_model, buffer_dict)
                    if self.prints > 0 and self.current_step % self.prints == 0:
                        print(f"Post Cont {mm_key} Target: "), self.print_estimator("PolarTargetPos", print_cov=2, buffer_dict=buffer_dict)
                        #print("Post Cont Meas Robot: "), self.print_estimator("RobotVel", print_cov=2, buffer_dict=buffer_dict)

    def set_observations(self):
        obs: Dict[str, Observation] = {}
        required_observations = [key for key in [obs for mm in [val[0] for val in self.MMs.values()] for obs in mm.required_observations] + [obs for ai in self.AIs.values() for obs in ai.required_observations] if key in self.env.observations.keys()]
        # TODO: Try not passing information about exact mean and stddev of sensor noise to the model
        #observation_noise = {key: (self.env.observation_noise[key] if key in self.env.observation_noise.keys() else (0.0,0.0)) for key in required_observations}
        self.env.required_observations = required_observations
        for key in required_observations:
            # TODO: Find better way to configure update uncertainty
            #obs[key] = Observation(key, 1, (torch.tensor(observation_noise[key][0]), torch.eye(1)*observation_noise[key][1]), self.get_observation_update_noise(key))
            obs[key] = Observation(key, 1)
        self.last_observation: Tuple[dict,dict] = None
        return obs

    def set_static_sensor_noises(self):
        for obs_key, obs in self.obs.items():
            obs.static_sensor_noise = self.get_static_sensor_noise(obs_key)

    def connect_states(self):
        for connection in list(self.AIs.values()) + [val[0] for val in self.MMs.values()]:
            connection.set_connected_states([self.REs[estimator_id] for estimator_id in connection.required_estimators], [self.obs[obs_id] for obs_id in connection.required_observations])

    def update_observations(self):
        self.last_observation = self.env.get_observation()
        for key, value in self.last_observation[0].items():
            if value is not None:
                self.obs[key].set_observation(
                    obs = torch.tensor([value]),
                    time = self.env.time,
                )

    def _eval_goal_with_aux(self, action: torch.Tensor, goal: Goal):
        buffer_dict = self.eval_step(action, new_step=False)
        loss = goal.loss_function_from_buffer(buffer_dict)
        return loss, loss
    
    def _eval_estimator_with_aux(self, action, estimator_id):
        buffer_dict = self.eval_step(action, new_step=False)
        state = buffer_dict[estimator_id]
        return state, state

    def compute_goal_action_gradient(self, goal) -> Dict[str, torch.Tensor]:
        action = self.last_action
        jacobian, step_eval = jacrev(
            self._eval_goal_with_aux,
            argnums=0,
            has_aux=True)(action, goal)
        return jacobian
    
    def compute_estimator_action_gradient(self, estimator_id, action):
        jacobian, step_eval = jacrev(
            self._eval_estimator_with_aux,
            argnums=0,
            has_aux=True)(action, estimator_id)
        return jacobian, step_eval

    def compute_action(self):
        gradients = self.compute_action_gradients()
        return gradients, self.compute_action_from_gradient({key: gradient["total"] for key, gradient in gradients.items()})

    def compute_action_gradients(self):
        gradients: Dict[str, Dict[str,torch.Tensor]] = {}
        for goal_key, goal in self.goals.items():
            gradients[goal_key] = self.compute_goal_action_gradient(goal)
        for goal_key, gradient in gradients.items():
            gradients[goal_key]["total"] = torch.zeros_like(list(gradient.values())[0])
            for gradkey, grad in gradient.items():
                if gradkey != "total":
                    gradients[goal_key]["total"] += grad
            if self.prints>0 and self.current_step % self.prints == 0:
                print(f"------------ {goal_key} Gradients ------------")
                for gradkey, grad in gradient.items():
                    if not torch.allclose(grad, torch.zeros_like(grad)):
                        print(f"{gradkey}: {[f'{-x:.3f}' for x in grad.tolist()]}")
                print("------------------------------------------------------")
        return gradients

    def get_steepest_gradient(self, gradients: Dict[str, torch.Tensor]):
        steepest_gradient = gradients.values()[0]
        for gradient in gradients.values():
            if gradient.norm() > steepest_gradient.norm():
                steepest_gradient = gradient
        return steepest_gradient

    def print_vector(self, vector: torch.Tensor, name = None, trail = "", use_scientific = False):
        if not use_scientific:
            use_scientific = any((abs(x.item()) < 1e-3 and not x.item() == 0.0) or abs(x.item()) > 1e3 for x in vector)
        if name is not None:
            print(f"{name}: [", end="")
        else:
            print("[", end="")
        for i, x in enumerate(vector):
            if i > 0:
                print(" ", end="")
            if 0 <= x.item():
                print(" ", end="")
            if use_scientific:
                print(f"{x.item():.3e}", end="")
            else:
                print(f"{x.item():.3f}", end="")
            if i < vector.shape[0] - 1:
                print(", ", end="")
        else:
            print("]" + trail)

    def print_matrix(self, matrix: torch.Tensor, name = None):
        use_scientific = any(abs(x.item()) < 1e-3 or abs(x.item()) > 1e3 for x in matrix.flatten())
        for i in range(matrix.shape[0]):
            if i == 0:
                if name is not None:
                    print(f"{name}: [", end="")
                else:
                    print("[", end="")
            else:
                if name is not None:
                    print(" " * (len(name) + 3), end="")
                else:
                    print(" ", end="")
            self.print_vector(matrix[i], trail="," if i < matrix.shape[0] - 1 else "]", use_scientific=use_scientific)

    def print_estimator(self, id, print_mean:bool=True, print_cov:int=0, buffer_dict=None):
        if print_mean:
            if buffer_dict is None:
                mean = self.REs[id].mean if id in self.REs.keys() else self.obs[id].mean
            else:
                mean = buffer_dict[id]['mean']
            self.print_vector(mean, id + " Mean" if print_cov else id)
        if print_cov != 0:
            if buffer_dict is None:
                cov = self.REs[id].cov if id in self.REs.keys() else self.obs[id].cov
            else:
                cov = buffer_dict[id]['cov']
            if print_cov == 1:
                self.print_matrix(cov, f"{id} Cov")
            elif print_cov == 2:
                self.print_vector(torch.sqrt(cov.diag()), id + " UCT")

    def reset(self, seed=None, video_path=None) -> None:
        for estimator in self.REs.values():
            estimator.reset()
        self.prints = 0
        self.current_step = 0
        obs, _ = self.env.reset(seed, video_path)
        self.update_observations()
        self.custom_reset()

    @abstractmethod
    def define_env(self) -> BaseEnv:
        """
        MUST be implemented by user. Returns the environment
        """
        raise NotImplementedError
    
    @abstractmethod
    def define_estimators(self) -> Dict[str, RecursiveEstimator]:
        """
        MUST be implemented by user. Returns the estimators
        """
        raise NotImplementedError
    
    @abstractmethod
    def define_measurement_models(self) -> Dict[str, ActiveInterconnection]:
        """
        MUST be implemented by user. Returns the measurement models
        """
        raise NotImplementedError
    
    @abstractmethod
    def define_active_interconnections(self) -> Dict[str, ActiveInterconnection]:
        """
        MUST be implemented by user. Returns the active interconnections
        """
        raise NotImplementedError
    
    @abstractmethod
    def define_goals(self) -> Dict[str, Goal]:
        """
        MUST be implemented by user. Returns the goals
        """
        raise NotImplementedError

    @abstractmethod
    def eval_interconnections(self, buffer_dict) -> Dict[str, Dict[str, torch.Tensor]]:
        """
        MUST be implemented by user. Evaluates the active interconnections between estimators
        """
        raise NotImplementedError
    
    @abstractmethod
    def get_control_input(self, action, buffer_dict, estimator_key) -> torch.Tensor:
        """
        MUST be implemented by user. Returns the control input for an estimator's forward models given an action of the robot's control space.
        As of now, control input is supposed to be the same for all forward models.
        """
        raise NotImplementedError
    
    def render(self):
        """
        CAN be implemented by user. Renders the environment
        """
        return self.env.render()

    def custom_reset(self):
        """
        CAN be implemented by user. Custom reset function
        """
        pass

    def compute_action_from_gradient(self, gradients):
        """
        CAN be implemented by user. Computes the action based on the gradients
        """
        return self.last_action - 1.0 * self.get_steepest_gradient(gradients)

    def print_estimators(self):
        """
        CAN be implemented by user, only necessary if "prints" option is used in run()
        """
        for estimator in self.REs.values():
            self.print_estimator(estimator.id, print_cov=True)
    
    def get_buffer_dict(self):
        """
        Gets the buffer_dict of all estimators, used for simulating update steps
        on a state detached from the real estimate, for gradient computation.
        """
        return {key: state.get_buffer_dict() for key, state in self.REs.items()}
    
    def get_static_sensor_noise(self, obs_key) -> Tuple[torch.Tensor,torch.Tensor]:
        """
        Returns the uncertainty to be added to inflate observation covariances for updates.
        This is done to increase Kalman Filter stability. SHOULD be overwritten by user.
        """
        return torch.zeros(1), 0.0 * torch.eye(1)

    def adapt_contingent_measurements(self, buffer_dict: dict):
        """
        Adapts the expected measurements and measurement noise based on the current state. SHOULD be overwritten by user and return TRUE.
        """
        raise NotImplementedError
    
class DroneEnvAICON(AICON):
    def __init__(self, env_config: dict):
        self.env_config = env_config
        super().__init__()

    def define_env(self):
        return GazeFixEnv(self.env_config)

    def convert_polar_to_cartesian_state(self, polar_mean, polar_cov):
        polar_cov = polar_cov[:2,:2]
        r = polar_mean[0]
        phi = polar_mean[1]
        mean = torch.stack([
            r * torch.cos(phi),
            r * torch.sin(phi)
        ])
        jac = torch.tensor([
            [torch.cos(phi), -r * torch.sin(phi)],
            [torch.sin(phi),  r * torch.cos(phi)]
        ], device=self.device, dtype=r.dtype)
        cov = jac @ polar_cov @ jac.T
        return mean, cov
    
    def get_control_input(self, action, buffer_dict, estimator_key) -> torch.Tensor:
        env_action = torch.empty_like(action)
        env_action[:2] = (action[:2] / action[:2].norm() if action[:2].norm() > 1.0 else action[:2]) * (self.env.robot.max_vel if self.env_config["action_mode"]==3 else self.env.robot.max_acc)
        env_action[2] = action[2] * (self.env.robot.max_vel_rot if self.env_config["action_mode"]==3 else self.env.robot.max_acc_rot)
        if self.prints > 0 and self.current_step % self.prints == 0:
            print("Action: ", end=""), self.print_vector(env_action)
        return torch.concat([torch.tensor([self.env_config["timestep"]]), env_action])
    
    def render(self):
        target_mean, target_cov = self.convert_polar_to_cartesian_state(self.REs["PolarTargetPos"].mean, self.REs["PolarTargetPos"].cov)
        estimator_means: Dict[str, torch.Tensor] = {"PolarTargetPos": target_mean}
        estimator_covs: Dict[str, torch.Tensor] = {"PolarTargetPos": target_cov}
        return self.env.render(1.0, {key: np.array(mean.cpu()) for key, mean in estimator_means.items()}, {key: np.array(cov.cpu()) for key, cov in estimator_covs.items()})

    def custom_reset(self):
        self.goals["PolarGoToTarget"].desired_distance = self.env.target.distance
        self.goals["PolarGoToTarget"].num_obstacles = self.env.num_obstacles
    