import numpy as np
import torch
from torch.func import jacrev
from abc import ABC, abstractmethod
from typing import Dict, List

from components.active_interconnection import ActiveInterconnection
from components.estimator import RecursiveEstimator, State
from components.goal import Goal
from environment.base_env import BaseEnv
from environment.gaze_fix_env import GazeFixEnv

# ========================================================================================

class AICON(ABC):
    def __init__(self, propagate_meas_uncertainty: bool = True):
        self.device = device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        self.dtype = torch.float64
        torch.set_default_device(self.device)
        torch.set_default_dtype(self.dtype)
        self.env: GazeFixEnv = None
        self.set_env(self.define_env())
        self.REs: Dict[str, RecursiveEstimator] = None
        self.set_estimators(self.define_estimators())
        self.AIs: Dict[str, ActiveInterconnection] = None
        self.set_active_interconnections(self.define_active_interconnections())
        self.goals: Dict[str, Goal] = None
        self.set_goals(self.define_goals())
        self.last_action = torch.tensor([0.0, 0.0, 0.0])
        self.propagate_meas_uncertainty = propagate_meas_uncertainty
        self.reset()

    def set_env(self, env):
        self.env = env
        self.obs: Dict[str, State] = {}
        observations: dict = self.env.get_observation()
        for key, value in observations.items():
            self.obs[key] = State(key, 1, self.device)
            self.obs[key].set_state(torch.tensor([value]), torch.zeros((1,1)))

    def set_estimators(self, REs: Dict[str, RecursiveEstimator]):
        self.REs = REs

    def set_active_interconnections(self, AIs: Dict[str, ActiveInterconnection]):
        self.AIs = AIs

    def set_goals(self, goals: Dict[str, Goal]):
        self.goals = goals

    def update_observations(self):
        observations: dict = self.env.get_observation()
        #print("========== Observations: ===========")
        for key, value in observations.items():
            # TODO: consider measurement noise (possibly on env level)
            self.obs[key].set_state(torch.tensor([value], device=self.device, dtype=self.dtype), torch.zeros((1,1), device=self.device, dtype=self.dtype))
            #self.print_state(key)

    def step(self, action):
        if action[:2].norm() > 1.0:
            action[:2] = action[:2] / action[:2].norm()
        if torch.abs(action[2]) > 1.0:
            action[2] = action[2]/torch.abs(action[2])
        self.last_action = action
        print(f"-------- Action: ", end=""), self.print_vector(action, trail=" --------")
        self.env.step(np.array(action.cpu()))
        buffers = self.eval_step(action, new_step=True)
        for key, buffer_dict in buffers.items():
            self.REs[key].load_state_dict(buffer_dict) if key in self.REs.keys() else self.obs[key].load_state_dict(buffer_dict)

    def _eval_goal_with_aux(self, action, goal):
        buffer_dict = self.eval_step(action, new_step=False)
        loss = goal.loss_function_from_buffer(buffer_dict)
        return loss, loss
    
    def _eval_estimator_with_aux(self, action, estimator_id):
        buffer_dict = self.eval_step(action, new_step=False)
        state = buffer_dict[estimator_id]
        #print(f"{state=}\n============================================================")
        return state, state

    def compute_goal_action_gradient(self, goal):
        #action = torch.tensor([0.0, 0.0, 0.0], device=self.device)
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

    def compute_action_gradients(self):
        gradients: Dict[str, torch.Tensor] = {}
        for key, goal in self.goals.items():
            #print(f"----------- Computing gradients for goal: {key} -----------")
            gradients[key] = self.compute_goal_action_gradient(goal)
        print("Action gradients:")
        for key, gradient in gradients.items():
            print(f"{key}: {[f'{x:.3f}' for x in gradient.tolist()]}")
        return gradients

    def get_steepest_gradient(self, gradients: Dict[str, torch.Tensor]):
        steepest_gradient = gradients.values()[0]
        for gradient in gradients.values():
            if gradient.norm() > steepest_gradient.norm():
                steepest_gradient = gradient
        return steepest_gradient

    def run(self, timesteps, env_seed=0, initial_action=None, render=True, prints=0, step_by_step=True, record_video=False):
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
        assert self.AIs is not None, "Active Interconnections not set"
        assert self.goals is not None, "Goals not set"
        self.reset()
        self.env.reset(seed=env_seed, video_path="test_vid.mp4" if record_video else None)
        print(f"============================ Initial State ================================")
        self.print_states()
        if initial_action is not None:
            self.last_action = initial_action
        if render:
            self.render()
        input("Press Enter to continue...")
        for step in range(timesteps):

            grad, val = self.compute_estimator_action_gradient("PolarTargetPos", self.last_action)
            print("Value:", val["state_mean"])
            print("Gradient:", grad["state_mean"])

            action_gradients = self.compute_action_gradients()
            action = self.compute_action(action_gradients)
            self.step(action)
            if render:
                self.render()
            step += 1
            if prints > 0 and step % prints == 0:
                print(f"============================ Step {step} ================================")
                self.print_states()

            if step_by_step:
                input("Press Enter to continue...")
        self.env.reset()

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

    def print_state(self, id, print_cov: bool = False):
        mean = self.REs[id].state_mean if id in self.REs.keys() else self.obs[id].state_mean
        self.print_vector(mean, id + " Mean" if print_cov else id)
        if print_cov:
            cov = self.REs[id].state_cov if id in self.REs.keys() else self.obs[id].state_cov
            self.print_matrix(cov, f"{id} Cov")

    def reset(self) -> None:
        for estimator in self.REs.values():
            estimator.reset()
        self.env.reset()

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
    def eval_step(self, action, new_step = False) -> Dict[str, Dict[str, torch.Tensor]]:
        """
        MUST be implemented by user. Evaluates one step of the environment and returns the buffer_dict
        """
        raise NotImplementedError

    def compute_action(self, gradients):
        """
        CAN be implemented by user. Computes the action based on the gradients
        """
        return - self.get_steepest_gradient(gradients)

    def print_states(self):
        """
        CAN be implemented by user, only necessary if "prints" option is used in run()
        """
        for estimator in self.REs.values():
            self.print_state(estimator.id, print_cov=True)

    def render(self, estimator_means: List[torch.Tensor] = None, estimator_covs: List[torch.Tensor] = None, playback_speed=1.0):
        """
        CAN be implemented by user, overwrite to include custom estimator means and covariances into rendering
        """
        return self.env.render()