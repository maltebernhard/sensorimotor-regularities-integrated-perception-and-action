from abc import ABC, abstractmethod
from typing import Dict, Optional
import torch
from torch.nn import Module
from torch.func import functional_call
from torch.func import jacrev

# =========================================================================================

class State(Module):
    def __init__(self, id, state_dim, device=None, dtype=None):
        super().__init__()
        self.id: str = id
        self.state_dim = state_dim
        self.device = device if device is not None else torch.get_default_device()
        self.dtype = dtype if dtype is not None else torch.get_default_dtype()

        self.register_buffer('state_mean', torch.zeros(self.state_dim))
        self.register_buffer('state_cov', torch.eye(self.state_dim))
        self.register_buffer('update_uncertainty', 0.0 * torch.eye(self.state_dim))

        self.to(device)
        self.get_buffer_dict()

    # -----------------------------------------------------------------------------------------------------

    @property
    def state(self):
        # Just a convenience method to return both mean and cov
        # if required. Otherwise, simply access the buffers directly
        return self.state_mean, self.state_cov

    def get_buffer_dict(self):
        self.buffer_dict = dict(self.named_buffers())
        return self.buffer_dict
    
    def set_state(self, mean: torch.Tensor, cov: torch.Tensor=None):
        assert mean.shape == (self.state_dim,), ("Mean shape does not match state_dim")
        if cov is not None:
            assert cov.shape == (self.state_dim, self.state_dim), ("Covariance shape does not match state_dim")
        self.state_mean = mean
        self.state_cov = torch.eye(
            self.state_dim, dtype=self.dtype, device=self.state_mean.device) if cov is None else cov
        self.get_buffer_dict()

# =========================================================================================

class Observation(State):
    def __init__(self, id, state_dim, sensor_noise, update_uncertainty, device=None, dtype=None):
        super().__init__(id, state_dim, device, dtype)
        self.updated = False
        self.last_updated = -1.0
        self.sensor_noise: torch.Tensor = sensor_noise
        self.update_uncertainty: torch.Tensor = update_uncertainty

    def set_observation(self, obs: torch.Tensor, custom_obs_noise: torch.Tensor=None, time=None):
        self.set_state(obs, custom_obs_noise**2)
        self.updated = True
        self.last_updated = time

# =========================================================================================

class RecursiveEstimator(ABC, State):
    def __init__(self, id, state_dim, device=None, dtype=None):
        super().__init__(id, state_dim, device, dtype)

        self.default_state: torch.Tensor = torch.zeros(self.state_dim, dtype=dtype, device=device)
        self.default_cov: torch.Tensor = torch.eye(self.state_dim, dtype=dtype, device=device)
        self.default_motion_noise: torch.Tensor = 1e-3 * torch.eye(self.state_dim, dtype=dtype, device=device)
        
        # Initialize static process/motion noise to identity
        self.register_buffer('forward_noise', torch.eye(self.state_dim, device=self.device, dtype=self.dtype))
        self.get_buffer_dict()

    # --------------------------- properties, getters and setters ---------------------------------

    def reset(self):
        self.set_state(self.default_state.clone(), self.default_cov.clone())
        self.set_static_motion_noise(self.default_motion_noise)

    def set_static_motion_noise(self, noise_cov: torch.Tensor) -> None:
        assert noise_cov.shape == (self.state_dim, self.state_dim), (
            f"Motion noise shape {noise_cov.shape} does not match {self.state_dim}")
        setattr(self, 'forward_noise', noise_cov.to(self.dtype))

    # --------------------------- everything related to forward model ------------------------------------

    @abstractmethod
    def forward_model(self, state: torch.Tensor, control_input: Dict[str, torch.Tensor]):
        raise NotImplementedError

    def forward_model_jac(self, state_mean, state_cov, control_input):
        """Autograd jacobian w.r.t state.
        Override to implement own jacobian"""
        return jacrev(self.forward_model, 0)(state_mean, state_cov, control_input)[0]

    def predict(self, u, custom_motion_noise=None):
        u = torch.atleast_1d(u) # Force scalar to be 1D

        self.state_mean, self.state_cov = self.forward_model(self.state_mean, self.state_cov, u)
        G_t = self.forward_model_jac(self.state_mean, self.state_cov, u)

        forward_noise = self.forward_noise if custom_motion_noise is None else custom_motion_noise
        self.state_cov = torch.matmul(G_t, torch.matmul(self.state_cov, G_t.t())) + forward_noise

    def functional_call_to_predict(self, buffers, kwargs):
        """Helper function to use functional_call for predict()

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
                                  custom_measurement_noise: Optional[Dict] = None):
        #NOTE: this doesn't work with current ActiveInterconnection implementation
        # assert meas_dict.keys() == implicit_meas_model.meas_config.keys(), (
        #     f'All measurements must be provided as mentioned in {implicit_meas_model} meas_config')
        if custom_measurement_noise is not None:
            assert meas_dict.keys() == custom_measurement_noise.keys(), (
                f'All measurements must be provided as mentioned in {custom_measurement_noise}')
            
        # Make sure scalar measurements are treated as 1D tensors
        for key in meas_dict.keys():
            meas_dict[key] = torch.atleast_1d(meas_dict[key])
        
        # The direct way would be to call implicit_measurement_model() and implicit_measurement_model_jac()
        # e.g., H_t, F_t = implicit_meas_model.implicit_measurement_model_jac(self.state_mean, meas)
        # But to save computation, we can use the auxillary 
        # outputs of jacrev() and both eval and compute Jacobian in one go
        H_t, F_t_dict, residual = implicit_meas_model.implicit_measurement_model_eval_and_jac(self.state_mean, meas_dict)

        # Below assert is not needed as F_t_dict should be produced from meas_dict
        # Still retaining in case in the future, F_t_dict is produced from some other source
        # assert F_t_dict.keys() == meas_dict.keys()

        K_part_1 = torch.matmul(self.state_cov, H_t.t())
        K_part_1_2 = torch.matmul(H_t, K_part_1)
        K_part_2 = K_part_1_2 # K_part_2 will get additional terms below

        for key in meas_dict.keys():
            if custom_measurement_noise is None:
                # NOTE cannot also have (or custom_measurement_noise[key] is None) in the if because
                # vmap doesn't broadcast over None within a dict!
                #print(f"Q: {getattr(implicit_meas_model, f'_Q_{key}')}")
                #print(f"F_t.t: {F_t_dict[key].t()}")
                K_part_2 += torch.matmul(F_t_dict[key], torch.matmul(getattr(implicit_meas_model, f'_Q_{key}'), F_t_dict[key].t()))
            else:
                # Sometimes it is beneficial to send only the diagonal of the noise covariance
                # Especially when the measurement is very high dimensional (and assumed to be uncorrelated)
                # So, although this is a HACK, it should work for all known cases
                if custom_measurement_noise[key].ndim == 1:
                    # This can be memory intensive for really large measurement dimensions
                    # custom_measurement_noise[key] = torch.diag(custom_measurement_noise[key])
                    # So to mitigate that, this is mathematically equivalent to the else case below
                    # but only holds true for diagonal noise covariance!
                    K_part_2 += torch.matmul(F_t_dict[key] * custom_measurement_noise[key][None, :], F_t_dict[key].t())
                elif custom_measurement_noise[key].ndim > 2 or (
                        custom_measurement_noise[key].ndim == 2 and 
                        custom_measurement_noise[key].shape[0] != custom_measurement_noise[key].shape[1]):
                    # A block-diagonal matrix must be given!
                    # NOTE this is not tested very well, and there no asserts
                    # So use with caution!!
                    block_size = custom_measurement_noise[key].shape[-1]
                    
                    # Convert an n-D tensor to 3D tensor with first dimension being the batch
                    # over the actual 2D block diagonal matrices. This makes it independent of the
                    # outside vmapped/batched dimension
                    block_diagonal_custom_measurement_noise = custom_measurement_noise[key].reshape(
                        -1, block_size, block_size)
                    
                    # Chunk up the Jacobian matrix into blocks
                    F_t_blocked = F_t_dict[key].reshape(
                        block_diagonal_custom_measurement_noise.shape[0], -1, block_size)
                    
                    _propagate_covariance_block = torch.vmap(self._propagate_covariance, in_dims=(0, 0))(
                        F_t_blocked, block_diagonal_custom_measurement_noise)
                    
                    K_part_2 += torch.sum(_propagate_covariance_block, dim=0)
                else:
                    K_part_2 += self._propagate_covariance(F_t_dict[key], custom_measurement_noise[key])
        

        # if self.id == "PolarTargetPos":# and "RobotVel" in meas_dict.keys():
        #     for key in ["target_offset_angle", "target_offset_angle_dot"]:
        #         print(f"{key} mean: {meas_dict[key].item():.3f} | cov: {custom_measurement_noise[key].item():.3f}")
        #     print(f"F_t: {F_t_dict}")
        #     print(f"K_part_2: {K_part_2}")


        K_part_2 = torch.atleast_2d(K_part_2) # in case state is 1D 

        if implicit_meas_model.regularize_kalman_gain:
            K_part_2 += 1e-6 * torch.eye(K_part_2.shape[0],
                                         dtype=self.dtype,
                                         device=self.state_mean.device)

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

        self.state_mean = self.state_mean + torch.matmul(kalman_gain, innovation)
        self.state_cov = self.state_cov - torch.matmul(torch.matmul(kalman_gain, H_t), self.state_cov)

    def functional_call_to_update_with_specific_meas(self, buffers, implicit_meas_model, kwargs):
        args_to_be_passed = ('update_with_specific_meas', implicit_meas_model)
        return functional_call(
            self, buffers, args_to_be_passed, kwargs)

    @staticmethod
    def _propagate_covariance(jacobian_matrix, covariance_matrix):
        """Helper function to propagate covariance."""
        return torch.matmul(jacobian_matrix, torch.matmul(covariance_matrix, jacobian_matrix.t()))

    # ------------------- workaround for torch.functional_call() for forward_model and measurement ----------------------------

    def forward(self, type_of_compute, *args, **kwargs):
        # Multiple measurement models constrain how
        # KF goes through normal iteration. So predict
        # and multiple updates maybe asynchronous.
        # So instead of simply doing a predict() and update(),
        # allow the user to specify what to do as type_of_compute

        # Why not simply call the respective predict() and update() directly?
        # Well, functional_calls under torch.func only allow forward() calls
        # That's why the user specifies what to do as type_of_compute
        # And the parameters that go to _those_ functions need
        # to be passed as args and kwargs
        # NOTE:
        #   -Things that are meant to be mapped over vector 
        #   (e.g., batch dimension) need to be passed as kwargs
        #   Things that are meant to be single instance 
        #   (e.g., single instance of measurement model) need to be passed as args
        #
        # Also NOTE there are no complicated checks/asserts for args and kwargs!
        # So use with care!
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
            'state_mean': self.state_mean,
            'state_cov': self.state_cov
        }
    
    def call_predict(self, u, buffer_dict):
        args_to_be_passed = ('predict',)
        kwargs = {'u': u}
        return functional_call(self, buffer_dict[self.id], args_to_be_passed, kwargs)

    def call_update_with_active_interconnection(self, active_interconnection, buffer_dict: Dict[str, torch.Tensor]):
        args_to_be_passed = ('update_with_specific_meas', active_interconnection)
        kwargs = {'meas_dict': active_interconnection.get_state_dict(buffer_dict, self.id), 'custom_measurement_noise': active_interconnection.get_cov_dict(buffer_dict, self.id)}
        return functional_call(self, buffer_dict[self.id], args_to_be_passed, kwargs)
