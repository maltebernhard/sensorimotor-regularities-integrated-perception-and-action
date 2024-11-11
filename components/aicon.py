import numpy as np
import torch
from torch.func import jacrev, functional_call
from abc import ABC, abstractmethod
from typing import Dict, List
from components.active_interconnection import ActiveInterconnection
from components.estimator import RecursiveEstimator
from components.goal import Goal

# ========================================================================================

class AICON(ABC):
    def __init__(self, device, env, REs, AIs, goals):
        self.device = device
        self.env = env
        self.REs: Dict[str, RecursiveEstimator] = REs
        self.AIs: Dict[str, ActiveInterconnection] = AIs
        self.goals: List[Goal] = goals
        self.reset()

    def call_predict(self, estimator, u, buffer_dict):
        args_to_be_passed = ('predict',)
        kwargs = {'u': u}
        return functional_call(self.REs[estimator], buffer_dict, args_to_be_passed, kwargs)

    def call_update_with_specific_meas(self, estimator, specific_meas_model, meas_dict: Dict[str, torch.Tensor], buffer_dict):
        args_to_be_passed = ('update_with_specific_meas', specific_meas_model)
        kwargs = {'meas_dict': meas_dict}
        return functional_call(self.REs[estimator], buffer_dict, args_to_be_passed, kwargs)

    def step(self, action):
        gain = 10.0
        env_action = action * gain
        if env_action[:2].norm() > 1.0:
            env_action[:2] = env_action[:2] / env_action[:2].norm()
        if env_action[2] > 1.0:
            env_action[2] = 1.0
        #print(f"Action: {env_action}")
        buffers = self.eval_step(env_action)
        for key, buffer_dict in buffers.items():
            self.REs[key].load_state_dict(buffer_dict)
        self.env.step(np.array(env_action.cpu()))

    def _eval_goal_with_aux(self, action, goal):
        buffer_dict = self.eval_step(action)
        loss = goal.loss_function_from_buffer(buffer_dict)
        return loss, loss

    def compute_goal_action_gradient(self, goal):
        action = torch.tensor([0.0, 0.0, 0.0], device=self.device)
        jacobian, step_eval = jacrev(
            self._eval_goal_with_aux,
            argnums=0, # x and meas_dict (all measurements within the dict!)
            has_aux=True)(action, goal)
        return jacobian
    
    def compute_action_gradients(self):
        gradients = []
        for goal in self.goals:
            gradients.append(self.compute_goal_action_gradient(goal))
        return gradients
    
    def get_steepest_gradient(self, gradients):
        steepest_gradient = gradients[0]
        for gradient in gradients:
            if gradient.norm() > steepest_gradient.norm():
                steepest_gradient = gradient
        return steepest_gradient
    
    def run(self, timesteps, env_seed, render = True, prints = 0, step_by_step = True):
        """
        Runs AICON on the environment for a given number of timesteps
        Args:
            timesteps:    number of timesteps to run
            env_seed:     seed for the environment
            render:       render the environment
            prints:       print the states every "prints" timesteps
            step_by_step: wait for user input after each step
        """
        self.reset()
        self.env.reset(seed=env_seed)

        for step in range(timesteps):
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
                input("Press Enter to continue...\n")
    
    @abstractmethod
    def reset(self) -> None:
        """
        MUST be implemented by user. Sets initial states of all estimators.
        """
        raise NotImplementedError

    @abstractmethod
    def eval_step(self, action):
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
        raise NotImplementedError

    def render(self, estimator_means: List[torch.Tensor], estimator_covs: List[torch.Tensor], playback_speed=1.0):
        """
        CAN be implemented by user, overwrite to include custom estimator means and covariances into rendering
        """
        return self.env.render()