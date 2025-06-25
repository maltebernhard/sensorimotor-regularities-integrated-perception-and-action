import numpy as np
import torch
from torch.func import jacrev, functional_call
from abc import ABC, abstractmethod
from typing import Dict, List, Tuple, Type, Optional
from torch.nn import Module
import wandb

from components.environment import BaseEnv

from components.helpers import rotate_vector_2d

# ========================================================================================

class AICON(ABC):
    """
    Abstract parent class for an AICON network.
    """
    def __init__(self):
        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        self.dtype = torch.float64
        torch.set_default_device(self.device)
        torch.set_default_dtype(self.dtype)
        self.env: BaseEnv = self.define_env()
        self.REs: Dict[str, RecursiveEstimator] = self.define_estimators()
        self.MMs: Dict[str, Tuple[ActiveInterconnection,List[str]]] = self.define_measurement_models()
        self.AIs: Dict[str, ActiveInterconnection] = self.define_active_interconnections()
        self.obs: Dict[str, Observation] = self.set_observations()
        self.set_static_sensor_noises()
        self.connect_states()
        self.goal: Goal = self.define_goal()
        self.last_action: torch.Tensor = torch.tensor([0.0, 0.0, 0.0])
        self.prints = 0
        self.current_step = 0

    def run(self, timesteps, env_seed=0, initial_action=None, render=True, prints=0, step_by_step=True, logger=None, video_path=None):
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
        assert self.goal is not None, "Goal not set"
        self.reset(seed=env_seed, video_path=video_path)
        if initial_action is not None:
            self.last_action = initial_action
        if render:
            self.render()
        if step_by_step:
            input("Press Enter to continue...")
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
        for step in range(timesteps+1):
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
                    goal_loss = self.goal.loss_function_from_buffer(buffer_dict),
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
        """
        Performs one step of the environment with the given action and updates the estimators.
        """
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
        Evaluates one step of the environment (including forward models, measurement and interconnection models of all estimators) and returns the buffer_dict.
        Args:
            action:    the action to be taken
            new_step:  whether to utilize new sensor readings or predicted ones (estimation update vs. gradient computation)
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
        """
        updates the estimators with new measurements, using the expected measurement noise.
        """
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
        Updates the estimators with the expected measurements and expected measurement noise.
        """
        if self.prints > 0 and self.current_step % self.prints == 0:
            print("Pre Cont Meas Target: "), self.print_estimator("PolarTargetPos", print_cov=2, buffer_dict=buffer_dict)
            #print("Pre Cont Meas Robot: "), self.print_estimator("RobotVel", print_cov=2, buffer_dict=buffer_dict)
        for mm_key, (meas_model, estimator_keys) in self.MMs.items():
            if meas_model.all_observations_updated():
                for estimator_key in estimator_keys:
                    self.REs[estimator_key].call_update_with_smr(meas_model, buffer_dict)
                    if self.prints > 0 and self.current_step % self.prints == 0:
                        print(f"Post Cont {mm_key} Target: "), self.print_estimator("PolarTargetPos", print_cov=2, buffer_dict=buffer_dict)
                        #print("Post Cont Meas Robot: "), self.print_estimator("RobotVel", print_cov=2, buffer_dict=buffer_dict)

    def set_observations(self):
        """
        Sets observation objects for all measurements required from the environment.
        """
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
        """
        Sets the expected static sensor noise for all observations.
        """
        for obs_key, obs in self.obs.items():
            obs.static_sensor_noise = self.get_static_sensor_noise(obs_key)

    def connect_states(self):
        """
        Sets all connected measurements and estimators for active interconnections.
        """
        for connection in list(self.AIs.values()) + [val[0] for val in self.MMs.values()]:
            connection.set_connected_states([self.REs[estimator_id] for estimator_id in connection.required_estimators], [self.obs[obs_id] for obs_id in connection.required_observations])

    def update_observations(self):
        """
        Gets the latest observations from the environment and updates the observation objects.
        """
        self.last_observation = self.env.get_observation()
        for key, value in self.last_observation[0].items():
            if value is not None:
                self.obs[key].set_observation(
                    obs = torch.tensor([value]),
                    time = self.env.time,
                )

    def _eval_goal_with_aux(self, action: torch.Tensor, goal):
        """
        Hacky function for simultaneously computing action gradient and predicted value of a goal function using torch autodiff.
        """
        buffer_dict = self.eval_step(action, new_step=False)
        loss = goal.loss_function_from_buffer(buffer_dict)
        return loss, loss
    
    def _eval_estimator_with_aux(self, action, estimator_id):
        """
        Hacky function for simultaneously computing action gradient and predicted value of an estimator using torch autodiff.
        """
        buffer_dict = self.eval_step(action, new_step=False)
        state = buffer_dict[estimator_id]
        return state, state

    def compute_goal_action_gradient(self, goal) -> Dict[str, torch.Tensor]:
        """
        Computes the gradient of the goal function w.r.t. the last action.
        """
        action = self.last_action
        jacobian, step_eval = jacrev(
            self._eval_goal_with_aux,
            argnums=0,
            has_aux=True)(action, goal)
        return jacobian
    
    def compute_estimator_action_gradient(self, estimator_id, action):
        """
        Computes the gradient of an estimator w.r.t. the last action.
        """
        jacobian, step_eval = jacrev(
            self._eval_estimator_with_aux,
            argnums=0,
            has_aux=True)(action, estimator_id)
        return jacobian, step_eval

    def compute_action(self):
        """
        Computes the action based on the goal action gradients.
        """
        gradient = self.compute_action_gradient()
        return gradient, self.compute_action_from_gradient(gradient["total"])

    def compute_action_gradient(self):
        """
        Computes the goal action gradients as a dict of subgoal gradients.
        """
        gradient_dict: Dict[str,torch.Tensor] = {}
        gradient_dict = self.compute_goal_action_gradient(self.goal)
        gradient_dict["total"] = torch.zeros_like(list(gradient_dict.values())[0])
        for gradkey, grad in gradient_dict.items():
            if gradkey != "total":
                gradient_dict["total"] += grad
        if self.prints>0 and self.current_step % self.prints == 0:
            print(f"------------ Gradients ------------")
            for gradkey, grad in gradient_dict.items():
                if not torch.allclose(grad, torch.zeros_like(grad)):
                    print(f"{gradkey}: {[f'{-x:.3f}' for x in grad.tolist()]}")
            print("------------------------------------------------------")
        return gradient_dict

    def get_steepest_gradient(self, gradients: Dict[str, torch.Tensor]):
        """
        Returns the steepest gradient from a dict of gradients (not utilized in latest model).
        """
        steepest_gradient = gradients.values()[0]
        for gradient in gradients.values():
            if gradient.norm() > steepest_gradient.norm():
                steepest_gradient = gradient
        return steepest_gradient

    def print_vector(self, vector: torch.Tensor, name = None, trail = "", use_scientific = False):
        """
        Pretty print for debugging.
        """
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
        """
        Pretty print for debugging.
        """
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
        """
        Pretty print for debugging.
        """
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
        """
        Resets estimators and environment.
        """
        for estimator in self.REs.values():
            estimator.reset()
        self.prints = 0
        self.current_step = 0
        obs, _ = self.env.reset(seed)#, video_path) # TODO: removed video path for mujoco env
        self.update_observations()
        self.custom_reset()

    @abstractmethod
    def define_env(self) -> BaseEnv:
        """
        MUST be implemented by user. Returns the environment of a specific AICON model.
        """
        raise NotImplementedError
    
    @abstractmethod
    def define_estimators(self) -> Dict[str, object]:
        """
        MUST be implemented by user. Returns the estimators of a specific AICON model.
        """
        raise NotImplementedError
    
    @abstractmethod
    def define_measurement_models(self) -> Dict[str, Tuple[object, List[str]]]:
        """
        MUST be implemented by user. Returns the measurement models of a specific AICON model.
        """
        raise NotImplementedError
    
    @abstractmethod
    def define_active_interconnections(self) -> Dict[str, object]:
        """
        MUST be implemented by user. Returns the active interconnections of a specific AICON model.
        """
        raise NotImplementedError
    
    @abstractmethod
    def define_goal(self) -> object:
        """
        MUST be implemented by user. Returns the goal of a specific AICON model.
        """
        raise NotImplementedError

    @abstractmethod
    def eval_interconnections(self, buffer_dict) -> Dict[str, Dict[str, torch.Tensor]]:
        """
        MUST be implemented by user. Evaluates the active interconnections between estimators of a specific AICON model.
        """
        raise NotImplementedError
    
    @abstractmethod
    def get_control_input(self, action, buffer_dict, estimator_key) -> torch.Tensor:
        """
        CAN be implemented by user. Returns the control input for an estimator's forward models given an action.
        """
        return action
    
    def render(self):
        """
        CAN be implemented by user. Renders the environment.
        """
        return self.env.render()

    def custom_reset(self):
        """
        CAN be implemented by user. Custom reset for adjusting model parameters if needed.
        """
        pass

    def compute_action_from_gradient(self, gradient):
        """
        CAN be implemented by user to adjust gains. Computes the next action based on the computed goal gradient.
        """
        return self.last_action - 1.0 * gradient

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
        SHOULD be overwritten by user. Returns the expected static sesnro noise to be used in measurement updates. 
        """
        return torch.zeros(1), 0.0 * torch.eye(1)

# =========================================================================================

class Observation:
    """
    Simple class for holding the latest observation of a sensor, and its associated measurement noise.
    """
    def __init__(self, id, obs_dim):
        self.id = id                # name of the observation
        self.dim = obs_dim          # dimension of the observation
        self.updated = False        # whether the observation has been updated in the last time step
        self.last_updated = -1.0    # time of last update
        self.static_sensor_noise: Tuple[torch.Tensor,torch.Tensor] = None # expected static sensor noise

    def set_observation(self, obs: torch.Tensor, time=None):
        self.last_measurement = obs
        self.updated = True
        self.last_updated = time

# ----------------------------------------------------------------------------------------

class State(Module):
    """
    A state value as represented by a recursive estimator (mean, covariance, additional update uncertainty to increase stability of interconnection updates).
    """
    def __init__(self, id, state_dim, device=None, dtype=None):
        super().__init__()
        self.id: str = id
        self.state_dim = state_dim
        self.device = device if device is not None else torch.get_default_device()
        self.dtype = dtype if dtype is not None else torch.get_default_dtype()

        self.default_state: torch.Tensor = torch.zeros(self.state_dim, dtype=dtype, device=device)

        self.register_buffer('mean', torch.zeros(self.state_dim))
        self.register_buffer('cov', torch.eye(self.state_dim))
        self.register_buffer('update_uncertainty', 0.0 * torch.eye(self.state_dim))

        self.to(device)
        self.get_buffer_dict()

    # -----------------------------------------------------------------------------------------------------

    @property
    def state(self):
        # Just a convenience method to return both mean and cov
        # if required. Otherwise, simply access the buffers directly
        return self.mean, self.cov

    def get_buffer_dict(self):
        self.buffer_dict = dict(self.named_buffers())
        return self.buffer_dict
    
    def set_state(self, mean: torch.Tensor, cov: torch.Tensor=None):
        assert mean.shape == (self.state_dim,), ("Mean shape does not match state_dim")
        if cov is not None:
            assert cov.shape == (self.state_dim, self.state_dim), ("Covariance shape does not match state_dim")
        self.mean = mean
        self.cov = torch.eye(
            self.state_dim, dtype=self.dtype, device=self.mean.device) if cov is None else cov
        self.get_buffer_dict()

# ----------------------------------------------------------------------------------------

class RecursiveEstimator(ABC, State):
    """
    Abstract parent class for a recursive estimator in the form of an extended Kalman filter.
    Measurement updates are adapted to allow multi-input active interconenctions.
    """
    def __init__(self, id, state_dim, device=None, dtype=None):
        super().__init__(id, state_dim, device, dtype)
        self.default_cov: torch.Tensor = torch.eye(self.state_dim, dtype=dtype, device=device)                 # default covariance
        self.default_motion_noise: torch.Tensor = 1e-3 * torch.eye(self.state_dim, dtype=dtype, device=device) # forward noise
        
        # Initialize static process/motion noise to identity
        self.register_buffer('forward_noise', torch.eye(self.state_dim, device=self.device, dtype=self.dtype))
        self.get_buffer_dict()

    # --------------------------- properties, getters and setters ---------------------------------

    def reset(self):
        """
        Resets the estimator to its default state.
        """
        self.set_state(self.default_state.clone(), self.default_cov.clone())
        self.set_static_motion_noise(self.default_motion_noise)

    def set_static_motion_noise(self, noise_cov: torch.Tensor) -> None:
        """
        Sets the static forward noise of the estimator.
        """
        assert noise_cov.shape == (self.state_dim, self.state_dim), (
            f"Motion noise shape {noise_cov.shape} does not match {self.state_dim}")
        setattr(self, 'forward_noise', noise_cov.to(self.dtype))

    # --------------------------- everything related to forward model ------------------------------------

    @abstractmethod
    def forward_model(self, state: torch.Tensor, control_input: Dict[str, torch.Tensor]):
        """
        MUST be implemented by user. Forward model of the estimator.
        """
        raise NotImplementedError

    def forward_model_jac(self, state_mean, state_cov, control_input):
        """
        Autograd jacobian w.r.t state.
        Override to implement own jacobian
        """
        return jacrev(self.forward_model, 0)(state_mean, state_cov, control_input)[0]

    def predict(self, u, custom_motion_noise=None):
        """
        Runs the forward model of the estimator and updates the uncertainty covariance matrix.
        """
        u = torch.atleast_1d(u) # Force scalar to be 1D

        self.mean, self.cov = self.forward_model(self.mean, self.cov, u)
        G_t = self.forward_model_jac(self.mean, self.cov, u)

        forward_noise = self.forward_noise if custom_motion_noise is None else custom_motion_noise
        self.cov = torch.matmul(G_t, torch.matmul(self.cov, G_t.t())) + forward_noise

    def functional_call_to_predict(self, buffers, kwargs):
        """
        Adapted from Aravind Battaje.
        
        Helper function to use functional_call for predict()

        The typical use case is to vmap over a bank of filters, where
        `filter_instance` is one instance of the filter, which will be
        stripped down to a functional form. The values of the filters must
        be stored in `buffers`. And the arguments to the predict() function
        must be passed as `kwargs`.
        
        Refer to forward() for more details.
        """
        args_to_be_passed = ( 'predict' )
        return functional_call(
            self, buffers, args_to_be_passed, kwargs)

    # ----------------------------- everything related to measurement update ----------------------------------

    def update_with_specific_meas(self, meas_dict: dict,
                                  implicit_meas_model,
                                  measurement_noise: Optional[Dict] = None):
        """
        Adapted from Aravind Battaje.
        Updates the estimator with a specific measurement model or active interconnection. Essentially a modified extended Kalman filter measurement update.
        """
        if measurement_noise is not None:
            assert meas_dict.keys() == measurement_noise.keys(), (
                f'All measurements must be provided as mentioned in {measurement_noise}')
            
        # Make sure scalar measurements are treated as 1D tensors
        for key in meas_dict.keys():
            meas_dict[key] = torch.atleast_1d(meas_dict[key])
        
        # The direct way would be to call implicit_measurement_model() and implicit_measurement_model_jac()
        # e.g., H_t, F_t = implicit_meas_model.implicit_measurement_model_jac(self.mean, meas)
        # But to save computation, we can use the auxillary 
        # outputs of jacrev() and both eval and compute Jacobian in one go
        H_t, F_t_dict, residual = implicit_meas_model.implicit_measurement_model_eval_and_jac(self.mean, meas_dict)

        # Below assert is not needed as F_t_dict should be produced from meas_dict
        # Still retaining in case in the future, F_t_dict is produced from some other source
        # assert F_t_dict.keys() == meas_dict.keys()

        K_part_1 = torch.matmul(self.cov, H_t.t())
        K_part_1_2 = torch.matmul(H_t, K_part_1)
        K_part_2 = K_part_1_2 # K_part_2 will get additional terms below

        for key in meas_dict.keys():
            if measurement_noise is None:
                raise Exception("Custom measurement noise must be provided.")
            else:
                # Sometimes it is beneficial to send only the diagonal of the noise covariance
                # Especially when the measurement is very high dimensional (and assumed to be uncorrelated)
                # So, although this is a HACK, it should work for all known cases
                if measurement_noise[key].ndim == 1:
                    # This can be memory intensive for really large measurement dimensions
                    # custom_measurement_noise[key] = torch.diag(custom_measurement_noise[key])
                    # So to mitigate that, this is mathematically equivalent to the else case below
                    # but only holds true for diagonal noise covariance!
                    K_part_2 += torch.matmul(F_t_dict[key] * measurement_noise[key][None, :], F_t_dict[key].t())
                elif measurement_noise[key].ndim > 2 or (
                        measurement_noise[key].ndim == 2 and 
                        measurement_noise[key].shape[0] != measurement_noise[key].shape[1]):
                    # A block-diagonal matrix must be given!
                    # NOTE this is not tested very well, and there no asserts
                    # So use with caution!!
                    block_size = measurement_noise[key].shape[-1]
                    
                    # Convert an n-D tensor to 3D tensor with first dimension being the batch
                    # over the actual 2D block diagonal matrices. This makes it independent of the
                    # outside vmapped/batched dimension
                    block_diagonal_custom_measurement_noise = measurement_noise[key].reshape(
                        -1, block_size, block_size)
                    
                    # Chunk up the Jacobian matrix into blocks
                    F_t_blocked = F_t_dict[key].reshape(
                        block_diagonal_custom_measurement_noise.shape[0], -1, block_size)
                    
                    _propagate_covariance_block = torch.vmap(self._propagate_covariance, in_dims=(0, 0))(
                        F_t_blocked, block_diagonal_custom_measurement_noise)
                    
                    K_part_2 += torch.sum(_propagate_covariance_block, dim=0)
                else:
                    K_part_2 += self._propagate_covariance(F_t_dict[key], measurement_noise[key])

        K_part_2 = torch.atleast_2d(K_part_2) # in case state is 1D 

        if implicit_meas_model.regularize_kalman_gain:
            K_part_2 += 1e-6 * torch.eye(K_part_2.shape[0],
                                         dtype=self.dtype,
                                         device=self.mean.device)

        try:
            K_part_2_inv = torch.inverse(K_part_2)
        except RuntimeError as e:
            #print(f"WARN: {self.id} - Kalman update was not possible.")# as {e}. Filter should (normally) recover soon.")
            # HACK resetting covariances. Maybe a more systematic approach is warranted
            #self.set_state(torch.zeros_like(self.state[0]))
            return

        kalman_gain = torch.matmul(K_part_1, K_part_2_inv)
        kalman_gain = torch.atleast_2d(kalman_gain) # in case state is 1D
        innovation = -residual
        innovation = torch.atleast_1d(innovation)

        if implicit_meas_model.outlier_rejection_enabled:
            innovation_distance = torch.sqrt(
                torch.matmul(innovation, torch.matmul(K_part_2_inv, innovation.t())))

            # Don't update if outlier detected
            # The classic way would be to use control flow like this:
            # if innovation_distance > self.outlier_distance_threshold:
            #     return
            # But this is not vmap friendly. So, use torch.where instead:            
            innovation = torch.where(
                innovation_distance > implicit_meas_model.outlier_distance_threshold,
                torch.zeros_like(innovation), innovation)
            kalman_gain = torch.where(
                innovation_distance > implicit_meas_model.outlier_distance_threshold,
                torch.zeros_like(kalman_gain), kalman_gain)

        self.mean = self.mean + torch.matmul(kalman_gain, innovation)
        self.cov = self.cov - torch.matmul(torch.matmul(kalman_gain, H_t), self.cov)

    def functional_call_to_update_with_specific_meas(self, buffers, implicit_meas_model, kwargs):
        """
        Adapted from Aravind Battaje.
        Helper function to use functional_call for update_with_specific_meas()
        """
        args_to_be_passed = ('update_with_specific_meas', implicit_meas_model)
        return functional_call(
            self, buffers, args_to_be_passed, kwargs)

    @staticmethod
    def _propagate_covariance(jacobian_matrix, covariance_matrix):
        """
        Helper function to propagate covariance.
        """
        return torch.matmul(jacobian_matrix, torch.matmul(covariance_matrix, jacobian_matrix.t()))

    # ------------------- workaround for torch.functional_call() for forward_model and measurement ----------------------------

    def forward(self, type_of_compute, *args, **kwargs):
        """
        Adapted from Aravind Battaje.

        Multiple measurement models constrain how
        KF goes through normal iteration. So predict
        and multiple updates maybe asynchronous.
        So instead of simply doing a predict() and update(),
        allow the user to specify what to do as type_of_compute

        Why not simply call the respective predict() and update() directly?
        Well, functional_calls under torch.func only allow forward() calls
        That's why the user specifies what to do as type_of_compute
        And the parameters that go to _those_ functions need
        to be passed as args and kwargs
        NOTE:
          -Things that are meant to be mapped over vector 
          (e.g., batch dimension) need to be passed as kwargs
          Things that are meant to be single instance 
          (e.g., single instance of measurement model) need to be passed as args
        
        Also NOTE there are no complicated checks/asserts for args and kwargs!
        So use with care!
        """
        if type_of_compute == 'predict':
            self.predict(kwargs['u'], kwargs.get('custom_motion_noise', None))
        elif type_of_compute == 'update_with_specific_meas':
            # NOTE only first argument in args considered
            self.update_with_specific_meas(kwargs['meas_dict'], args[0], kwargs.get('custom_measurement_noise', None))
        # elif type_of_compute == 'update_with_active_interconnection':
        #     self.update_with_specific_meas(kwargs['meas_dict'], args[0], kwargs.get('custom_measurement_noise', None))
        else:
            raise TypeError(f'Unknown type_of_compute {type_of_compute}.'
                            ' NOTE you can also call predict() and '
                            'update_with_specific_meas() directly (if not using functional calls).')
        
        # NOTE this return is not actually required for normal operation
        # as the internal state/buffers are updated 
        # But when vmapping, this is essential as it doesn't track updated buffers
        # The naive way would be to simply return dict(self.named_buffers())
        # But it is better to be explicit about persistent (only required) buffers
        return {
            'mean': self.mean,
            'cov': self.cov
        }
    
    def call_predict(self, u, buffer_dict):
        """
        convenience wrapper for using torch.functional_call() to invoke the estimator's forward model
        """
        args_to_be_passed = ('predict',)
        kwargs = {'u': u}
        return functional_call(self, buffer_dict[self.id], args_to_be_passed, kwargs)

    def call_update_with_active_interconnection(self, active_interconnection, buffer_dict: Dict[str, torch.Tensor]):
        """
        convenience wrapper for using torch.functional_call() to update the estimator with an active interconnection
        """
        args_to_be_passed = ('update_with_specific_meas', active_interconnection)
        state_dict = active_interconnection.get_state_dict(buffer_dict, self.id)
        meas_offsets, covs = active_interconnection.get_cov_dict(buffer_dict, self.id)
        kwargs = {'meas_dict': {meas_key: state_dict[meas_key] - meas_offset for meas_key, meas_offset in meas_offsets.items()}, 'custom_measurement_noise': covs}
        return functional_call(self, buffer_dict[self.id], args_to_be_passed, kwargs)
    
    def call_update_with_smr(self, smr, buffer_dict: Dict[str, torch.Tensor]):
        """
        convenience wrapper for using torch.functional_call() to update the estimator with a sensorimotor regularity and its maximum likelihood predicted measurement
        """
        args_to_be_passed = ('update_with_specific_meas', smr)
        kwargs = {'meas_dict': smr.get_state_dict_with_predicted_meas(buffer_dict, self.id), 'custom_measurement_noise': smr.get_cov_dict(buffer_dict, self.id)[1]}
        return functional_call(self, buffer_dict[self.id], args_to_be_passed, kwargs)

# =========================================================================================

class ImplicitMeasurementModel(Module):
    """
    Code for this class taken and adapted with permission from Battaje, Aravind.
    Abstract interface to define a measurement model using implicit function and
    optionally its Jacobian.

    Notation as in Thrun et al., Probabilistic Robotics
    """
    def __init__(self,
                 id: str,
                 device=None,
                 outlier_rejection_enabled=False,
                 outlier_threshold=1.0,
                 regularize_kalman_gain=False,
                 dtype=None) -> None:
        """
        Args: 
            state_dim: Dimension of the state this measurement model will be attached to
            meas_config: A dict of the form {'meas1': meas1_dim, 'meas2': meas2_dim, ...}
            dtype: Data type of the tensors
        """
        super().__init__()

        self.id = id

        self.device = device if device is not None else torch.get_default_device()
        self.dtype = dtype if dtype is not None else torch.get_default_dtype()

        self.meas_config: Dict[str, int] = None
            
        self.outlier_rejection_enabled = outlier_rejection_enabled
        self.register_buffer(
            'outlier_distance_threshold',
            torch.tensor(outlier_threshold, dtype=dtype),
            persistent=False)
        self.regularize_kalman_gain = regularize_kalman_gain
        self.to(self.device)

    @abstractmethod
    def implicit_measurement_model(self, x: torch.Tensor, meas_dict: torch.Tensor):
        """
        MUST be implemented by the user. Relates state x to observations.
        Args:
            x: State tensor
            meas_dict: a dict of measurements according to self.meas_config
        Returns:
            a float value indicating how close the state relates to the given measurements
        """
        raise NotImplementedError

    def explicit_measurement_model(self, x: torch.Tensor) -> Dict[str, torch.Tensor]:
        """
        CAN be implemented by the user. Transforms state x to observation space.
        Args:
            x: State tensor
        Returns:
            a dict of measurements according to self.meas_config
        """
        raise NotImplementedError
    
    def _implicit_measurement_model_with_aux(self, x, meas_dict):
        """Helper function to use auxillary outputs of jacrev.

        This is so that implicit_measurement_model() doesn't have to be
        called twice."""
        ret = self.implicit_measurement_model(x, meas_dict)

        # While using jacrev() first output will be 
        # differentiated and second will be returned as is
        return ret, ret

    def implicit_measurement_model_eval_and_jac(self, x, meas_dict):
        """
        override to implement own jacobian
        """
        jacobians, implicit_measurement_model_eval = jacrev(
            self._implicit_measurement_model_with_aux,
            argnums=(0, 1),
            has_aux=True)(x, meas_dict)
        
        H_t = jacobians[0]
        F_t_dict = jacobians[1]

        return H_t, F_t_dict, implicit_measurement_model_eval

    def forward(self):
        # Multiple measurement models constrain how
        # KF goes through normal iteration. So predict
        # and multiple updates maybe asynchronous.
        # So, don't allow forward calls!
        raise NotImplementedError
    
# ----------------------------------------------------------------------------------------

class ActiveInterconnection(ABC, ImplicitMeasurementModel):
    """
    Active Interconnection class for multidirectional multi-input EKF measurement updates.
    """
    def __init__(self, id, required_estimators, required_observations=[]):
        super().__init__(id)
        self.required_estimators: List[str] = required_estimators
        self.required_observations: List[str] = required_observations
        self.connected_states: Dict[str, Type[State]] = None
        self.connected_observations: Dict[str, Type[Observation]] = None

    def implicit_measurement_model(self, x, meas_dict):
        """
        Hacky function transforming the implicit measurement model for a specific x into an interconnection model for any connected state.
        """
        missing_key = next(key for key in self.connected_states.keys() if key not in meas_dict)
        assert sum(1 for key in self.connected_states.keys() if key not in meas_dict) == 1, "There should be exactly one missing key"
        meas_dict[missing_key] = x
        return self.implicit_interconnection_model(meas_dict)

    def set_connected_states(self, states: List[State], observations: List[Observation]) -> None:
        """
        connects states and observations to the active interconnection
        """
        assert set(state.id for state in states) == set(self.required_estimators), f"Estimators should be {self.required_estimators}, but are {[state.id for state in states]}"
        assert set(obs.id for obs in observations) == set(self.required_observations), f"Observations should be {self.required_observations}, but are {[obs.id for obs in observations]}"
        self.connected_states: Dict[str, State] = {state.id: state for state in states}
        self.connected_observations: Dict[str, Observation] = {obs.id: obs for obs in observations}
        self.meas_config = {state.id: state.state_dim for state in states}
        self.meas_config.update({obs.id: obs.dim for obs in observations})

    def get_expected_meas_noise(self, buffer_dict: Dict[str,torch.Tensor]) -> Dict[str, Tuple[torch.Tensor,torch.Tensor]]:
        """
        returns the expected sensor noise (tuple of mean and stddev) for each sensory component.
        OVERWRITE for SMRs to include additional effects, such as foveal vision noise.
        """
        return {key: obs.static_sensor_noise for key, obs in self.connected_observations.items()}
    
    def all_observations_updated(self):
        """
        Checks whether all connected measurements have been updated in the last timestep. Allows skipping updates from old inputs.
        """
        return all(obs.updated for obs in self.connected_observations.values())

    def get_state_dict(self, buffer_dict: dict, estimator_id):
        """
        gets the connected state properties of the active interconnection that are relevant for an update
        """
        state_dict = {state_key: buffer_dict[state_key]['mean'] for state_key in list(self.connected_states.keys()) if state_key != estimator_id}
        state_dict.update({obs_key: obs.last_measurement for obs_key, obs in self.connected_observations.items()})
        return state_dict

    def get_cov_dict(self, buffer_dict: dict, estimator_id) -> Tuple[Dict[str,torch.Tensor],Dict[str,torch.Tensor]]:
        """
        gets the connected uncertainty covariances of connected states that are relevant for an update
        """
        # estimator_covs
        meas_offset_dict = {id: torch.zeros_like(buffer_dict[id]['mean']) for id in self.required_estimators if id != estimator_id}
        cov_dict = {id: buffer_dict[id]['cov'] + buffer_dict[id]['update_uncertainty'] for id in self.required_estimators if id != estimator_id}
        # observation_covs
        obs_noise = self.get_expected_meas_noise(buffer_dict)
        for key in self.required_observations:
            meas_offset_dict[key] = obs_noise[key][0]
            cov_dict[key] = obs_noise[key][1].pow(2)
        return meas_offset_dict, cov_dict

    @abstractmethod
    def implicit_interconnection_model(self, meas_dict):
        """
        Defines the comparison between the connected states and observations in innovation space. NEEDS to be defined by user
        """
        pass

# ----------------------------------------------------------------------------------------

class SensorimotorRegularity(ActiveInterconnection):
    """
    Special case of an active interconnection that connects a state and an action to sensory components.
    Is used to predict measurements based on the state and action, or to update state or action based on sensor inputs and one another.
    """
    def __init__(self, id: str, state_component: str, action_component: str, sensory_components: List[str]):
        self.state_component = state_component
        self.action_component = action_component
        super().__init__(id, required_estimators=[state_component, action_component], required_observations=sensory_components)

    def implicit_interconnection_model(self, meas_dict):
        predicted_meas = self.get_predicted_meas(meas_dict[self.state_component], meas_dict[self.action_component])
        return torch.atleast_1d(torch.stack([
            predicted_meas[key] - meas_dict[key] for key in self.required_observations
        ]).squeeze())

    @abstractmethod
    def get_predicted_meas(self, state, action) -> Dict[str,torch.Tensor]:
        """
        Returns predicted maximum likelihood measurements. NEEDS to be overwritten by user
        """
        pass

    # TODO: Make it possible to determine the prediction uncertainty of a maximum likelihood measurement.
    # def _get_predicted_meas_with_aux(self, state, action):
    #     pred_meas_dict = self.get_predicted_meas(state, action)
    #     pred_meas_tensor = torch.stack([pred_meas_dict[key] for key in self.required_observations])
    #     return pred_meas_tensor, pred_meas_tensor

    # def get_predicted_meas_eval_and_jac(self, state, action) -> Dict[str,torch.Tensor]:
    #     (H_t, u_t), implicit_measurement_model_eval = jacrev(
    #         self._get_predicted_meas_with_aux,
    #         argnums=(0, 1), # x and meas_dict (all measurements within the dict!)
    #         has_aux=True)(state, action)
    #     return H_t, implicit_measurement_model_eval

    # def get_meas_likelihood(self, buffer_dict: dict):
    #     H_t, F_t_dict, residual = self.implicit_measurement_model_eval_and_jac(self.mean, meas_dict)

    def get_contingent_noise(self, state: Dict[str,torch.Tensor]) -> Dict[str,Tuple[torch.Tensor,torch.Tensor]]:
        """
        Returns the expected state-dependent noise (mean and stddev) for each measurement, additional to basic sensor noise.
        OVERWRITE for SMRs to include state-dependent uncertainty effects, such as higher noise in peripheral vision.
        """
        return {key: (0.0,0.0) for key in self.required_observations}

    def get_expected_meas_noise(self, buffer_dict: Dict[str,torch.Tensor]) -> Dict[str,Tuple[torch.Tensor,torch.Tensor]]:
        """
        Returns the expected total sensor noise (tuple of mean and stddev) for each sensory component.
        OVERWRITE to include additional effects, such as noise scaling with value (e.g. for robot vel)
        """
        contingent_noise = self.get_contingent_noise(buffer_dict[self.state_component]['mean'])
        noise = {key: (
            obs.static_sensor_noise[0] + contingent_noise[key][0],
            obs.static_sensor_noise[1] + contingent_noise[key][1]
        ) for key, obs in self.connected_observations.items()}
        return noise
    
    def get_state_dict_with_predicted_meas(self, buffer_dict: dict, estimator_id):
        """
        Returns the state dict, but replaces measurements with predicted maximum likelihood observations. Used in prediction-based action generation.
        """
        state_dict = {key: buffer_dict[key]['mean'] for key in self.required_estimators if key != estimator_id}
        predicted_meas = self.get_predicted_meas(buffer_dict[self.state_component]['mean'], buffer_dict[self.action_component]['mean'])
        for key in self.required_observations:
            state_dict[key] = predicted_meas[key]
        return state_dict

# =========================================================================================

class Goal(ABC, Module):
    """
    Simple class for defining a goal based on estimated state properties and their estimation uncertainty.
    """
    def __init__(self, device=None):
        super().__init__()
        self.device = device if device is not None else torch.get_default_device()
    
    @abstractmethod
    def loss_function(self, buffer_dict) -> float:
        raise NotImplementedError
    
    def loss_function_from_buffer(self, buffer_dict):
        return self.loss_function(buffer_dict)

# =========================================================================================

class Logger:
    """
    Handles logging of simulation data, including estimators, observations, 
    goal losses, and actions. Supports local logging and integration with 
    Weights & Biases (wandb) for experiment tracking.
    """
    
    def __init__(self, variation_config:dict, variation_id:int, wandb_config:dict):
        self.data = {}
        self.run_seed = 0
        self.variation_config = variation_config
        self.variation_id = variation_id
        self.wandb_project: str = wandb_config['wandb_project'] if wandb_config is not None else None
        self.wandb_group: str = wandb_config['wandb_group'] if wandb_config is not None else None
        self.wandb_run = None

    def log(self, step: int, time: float, estimators: Dict[str,Dict[str,torch.Tensor]], env_state: Dict[str,float], observation: Dict[str,Dict[str,float]], goal_loss: Dict[str,Dict[str,torch.Tensor]], action: torch.Tensor, gradients: Dict[str,Dict[str,torch.Tensor]]):
        """
        Logs a single simulation step, updating local data and optionally 
        syncing with wandb.
        """
        # for new run, set up logging dict
        if self.run_seed not in self.data:
            self.create_run_dict(estimators, env_state, observation, goal_loss, action, gradients)
            if self.wandb_group is not None:
                self.create_wandb_run()
        # log data
        real_state = self.create_real_state(estimators, env_state)
        step_log = self.create_step_log(step, time, estimators, real_state, observation, goal_loss, action, gradients)

        if self.wandb_run is not None:
            self.log_wandb(step_log)
        self.log_local(step_log)

    def create_run_dict(self, estimators: Dict[str,Dict[str,torch.Tensor]], env_state: Dict[str,float], observation: Dict[str,Dict[str,float]], goal_loss: Dict[str,Dict[str,torch.Tensor]], action: torch.Tensor, gradient: Dict[str,torch.Tensor]):
        """
        Initializes the logging structure for a new simulation run.
        """
        self.data[self.run_seed] = {
                "step":                 np.array([]),     # time step
                "time":                 np.array([]),     # time in seconds
                "estimators": {estimator_key: {
                    'mean':             np.empty((0,) + estimators[estimator_key]["mean"].shape),     # mean of state estimate
                    'cov':              np.empty((0,) + estimators[estimator_key]["cov"].shape),      # covariance of state estimate
                    'uncertainty':      np.empty((0,) + (estimators[estimator_key]["mean"].shape[0],)), # uncertainty of state estimate: sqrt(diag(covariance))
                    'estimation_error': np.empty((0,) + estimators[estimator_key]["mean"].shape),     # estimation error: estimation - real state
                    'env_state':        np.empty((0,) + estimators[estimator_key]["mean"].shape),     # task state: real state with subtracted offsets (like desired target distance)
                } for estimator_key in estimators.keys()},
                "observation": {obs_key: {
                    "measurement":      np.array([]),     # measurement
                    "noise_mean":       np.array([]),     # mean of noise
                    "noise_stddev":     np.array([]),     # stddev of noise
                } for obs_key in observation.keys()},
                "goal_loss": {
                    subgoal_key:        np.array([])     # goal loss function value
                for subgoal_key in goal_loss.keys()},
                "gradient": {
                    subgoal_key:        np.empty((0,) + gradient[subgoal_key].shape)     # gradient of goal loss function
                for subgoal_key in gradient.keys()},
                "rtf_gradient": {
                    subgoal_key:        np.empty((0,) + gradient[subgoal_key].shape)     # gradient of goal loss function
                for subgoal_key in gradient.keys()},
                "action":               np.empty((0,) + action.shape),     # action
                "rtf_action":           np.empty((0,) + action.shape),     # action rotated to target frame
                "desired_distance":     env_state["desired_target_distance"],
                "collision":            np.array([]),     # collision flag
            }

    def create_wandb_run(self):
        """
        Creates a new wandb run for tracking the current simulation.
        """
        if self.wandb_run is not None:
            self.wandb_run.finish()
        config = self.variation_config.copy()
        config.update({"id": self.variation_id})
        self.wandb_run = wandb.init(
            project=self.wandb_project,
            name=f'{self.variation_id}_{self.run_seed}',
            group=self.wandb_group,
            config = config,
            save_code=False,
        )

    def create_real_state(self, estimators: Dict[str,Dict[str,torch.Tensor]], env_state: Dict[str,float]):
        """
        Extracts the real state from the environment state, formatted 
        to match the estimator structure.
        NOTE: Written for our specific experiment, needs to be adapted for different estimators and states.
        """
        # extract real state from env_state, matching to estimator structure
        real_state = {}
        for key in estimators.keys():
            if key[:5] == "Polar" and key[-3:] == "Pos":
                obj = key[5:-3].lower()
                real_state[key] = np.array([
                    env_state[f"{obj}_distance"],
                    env_state[f"{obj}_offset_angle"],
                ])
                if (obj=='target' and self.variation_config["target_config"][0] != "stationary") or ('obstacle' in obj and self.variation_config["obstacles"][int(obj[-1])-1][0] != "stationary"):
                    real_state[key] = np.append(real_state[key], env_state[f"{obj}_distance_dot_global"])
                    real_state[key] = np.append(real_state[key], env_state[f"{obj}_offset_angle_dot_global"])
                real_state[key] = np.append(real_state[key], env_state[f"{obj}_radius"])
            elif key == "RobotVel":
                real_state[key] = np.array([
                    env_state["vel_frontal"],
                    env_state["vel_lateral"],
                    env_state["vel_rot"],
                ])
            elif key[-6:] == "Radius":
                obj = key[:-6].lower()
                real_state[key] = np.array([
                    env_state[f"{obj}_radius"]
                ])
        real_state["collision"] = max([val for key, val in env_state.items() if "collision" in key])
        return real_state

    def create_step_log(self, step: int, time: float, estimators: Dict[str,Dict[str,torch.Tensor]], real_state: Dict[str,Dict[str,torch.Tensor]], observation: Dict[str,Dict[str,float]], goal_loss: Dict[str,Dict[str,torch.Tensor]], action: torch.Tensor, gradients: Dict[str,Dict[str,torch.Tensor]]):
        """
        Constructs a log entry for a single simulation step.
        """
        estimators = {estimator_key: {attribute_key: estimators[estimator_key][attribute_key].cpu().numpy() for attribute_key in estimators[estimator_key].keys()} for estimator_key in estimators.keys()}
        # HACK: remove negative covariances
        # TODO: check why they even exist
        for estimator in estimators.values():
            estimator['cov'][estimator['cov'] < 0] = 0
        return {
            "step":                     step,
            "time":                     time,
            "estimators": {
                estimator_key: {
                    'mean':             estimator['mean'],
                    'cov':              estimator['cov'],
                    'uncertainty':      np.sqrt(np.diag(estimator['cov'])),
                    'estimation_error': real_state[estimator_key] - estimator['mean'],
                    'env_state':        real_state[estimator_key],
                } for estimator_key, estimator in estimators.items() if estimator_key != "RobotVel"
            },
            "collision":                real_state["collision"],
            "observation": {
                obs_key: {
                    "measurement":      observation[obs_key]["measurement"],
                    "noise_mean":       observation[obs_key]["noise"][0],
                    "noise_stddev":     observation[obs_key]["noise"][1],
                } for obs_key in observation.keys()
            },
            "goal_loss":                {subgoal_key: goal_loss[subgoal_key].cpu().numpy() for subgoal_key in goal_loss.keys()},
            "gradient":                 {subgoal_key: gradients[subgoal_key].cpu().numpy() for subgoal_key in gradients.keys()},
            "rtf_gradient":             {subgoal_key: np.append(rotate_vector_2d(-estimators["PolarTargetPos"]['mean'][1],gradients[subgoal_key].cpu().numpy()[:2]),gradients[subgoal_key].cpu().numpy()[2]) for subgoal_key in gradients.keys()},
            "action":                   action.cpu().numpy(),
            "rtf_action":               np.append(rotate_vector_2d(-estimators["PolarTargetPos"]['mean'][1],action.cpu().numpy()[:2]),action.cpu().numpy()[2]),
        }

    def log_wandb(self, step_log: dict):
        """
        Logs the current step data to wandb.
        """
        if self.variation_config["target_config"][0] != "stationary":
            index_keys = {
                "PolarTargetPos": ["distance", "angle", "distance_dot", "angle_dot", "radius"],
                "RobotVel": ["frontal", "lateral", "rot"],
            }
        else:
            index_keys = {
                "PolarTargetPos": ["distance", "angle", "radius"],
                "RobotVel": ["frontal", "lateral", "rot"],
            }
        self.wandb_run.log({
            #"step": step_log["step"],
            #"time": step_log["time"],
            "estimators": {estimator_key: {
                attribute_key: {index_keys[estimator_key][i]: val for i, val in enumerate(estimator[attribute_key])} for attribute_key in estimator.keys()
            } for estimator_key, estimator in step_log["estimators"].items() if estimator_key != "RobotVel"},
            "collision": step_log["collision"],
            #"observation": step_log["observation"],
            "goal_loss": step_log["goal_loss"],
            #"gradient": step_log["gradient"],
            #"action": step_log["action"],
        })

    def log_local(self, step_log: dict):
        """
        Logs the current step data to the local data structure.
        """
        for key in ["step", "time", "collision"]:
            if key != "estimators":
                self.data[self.run_seed][key] = np.append(self.data[self.run_seed][key], step_log[key])
        for estimator_key in step_log["estimators"].keys():
            for attribute_key in step_log["estimators"][estimator_key].keys():
                self.data[self.run_seed]["estimators"][estimator_key][attribute_key] = np.append(self.data[self.run_seed]["estimators"][estimator_key][attribute_key], [step_log["estimators"][estimator_key][attribute_key]], axis=0)
        for obs_key in step_log["observation"].keys():
            for sub_key in step_log["observation"][obs_key].keys():
                self.data[self.run_seed]["observation"][obs_key][sub_key] = np.append(self.data[self.run_seed]["observation"][obs_key][sub_key], [step_log["observation"][obs_key][sub_key]], axis=0)
        for subgoal_key in step_log["goal_loss"].keys():
            self.data[self.run_seed]["goal_loss"][subgoal_key] = np.append(self.data[self.run_seed]["goal_loss"][subgoal_key], [step_log["goal_loss"][subgoal_key]], axis=0)
        for subgoal_key in step_log["gradient"].keys():
            self.data[self.run_seed]["gradient"][subgoal_key] = np.append(self.data[self.run_seed]["gradient"][subgoal_key], [step_log["gradient"][subgoal_key]], axis=0)
            self.data[self.run_seed]["rtf_gradient"][subgoal_key] = np.append(self.data[self.run_seed]["rtf_gradient"][subgoal_key], [step_log["rtf_gradient"][subgoal_key]], axis=0)
        self.data[self.run_seed]["action"] = np.append(self.data[self.run_seed]["action"], [step_log["action"]], axis=0)
        self.data[self.run_seed]["rtf_action"] = np.append(self.data[self.run_seed]["rtf_action"], [step_log["rtf_action"]], axis=0)

    def end_wandb_run(self):
        """
        Ends the current wandb run and cleans up resources.
        """
        if self.wandb_run is not None:
            self.wandb_run.finish()
            del self.wandb_run
            self.wandb_run = None
