from abc import ABC, abstractmethod
from typing import Dict, Optional
import torch
from torch.nn import Module
from torch.func import functional_call
from torch.func import jacrev

# =========================================================================================

class State(Module):
    def __init__(self, id, state_dim, device, dtype = torch.float64):
        super().__init__()
        self.id: str = id
        self.state_dim = state_dim
        self.device = device
        self.dtype = dtype

        self.register_buffer('state_mean', torch.zeros(self.state_dim, dtype=dtype))
        self.register_buffer('state_cov', torch.eye(self.state_dim, dtype=dtype))

        self.to(device)
        self.set_buffer_dict()

    # -----------------------------------------------------------------------------------------------------

    @property
    def state(self):
        # Just a convenience method to return both mean and cov
        # if required. Otherwise, simply access the buffers directly
        return self.state_mean, self.state_cov

    def set_buffer_dict(self):
        self.buffer_dict = dict(self.named_buffers())
        return self.buffer_dict
    
    def set_state(self, mean: torch.Tensor, cov: torch.Tensor=None):
        assert mean.shape == (self.state_dim,), ("Mean shape does not match state_dim")
        if cov is not None:
            assert cov.shape == (self.state_dim, self.state_dim), ("Covariance shape does not match state_dim")
        self.state_mean = mean
        self.state_cov = torch.eye(
            self.state_dim, dtype=self.dtype, device=self.state_mean.device) if cov is None else cov
        self.set_buffer_dict()

# =========================================================================================

class RecursiveEstimator(ABC, State):
    def __init__(self, id, state_dim, device, dtype = torch.float64):
        super().__init__(id, state_dim, device, dtype)
        
        # Initialize static process/motion noise to identity
        self.register_buffer('forward_noise', torch.eye(self.state_dim, device=self.device, dtype=self.dtype))
        self.set_buffer_dict()

    # --------------------------- properties, getters and setters ---------------------------------

    def set_static_motion_noise(self, noise_cov: torch.Tensor) -> None:
        assert noise_cov.shape == (self.state_dim, self.state_dim), (
            f"Motion noise shape {noise_cov.shape} does not match {self.state_dim}")
        setattr(self, '_R', noise_cov.to(self.dtype))

    # --------------------------- everything related to forward model ------------------------------------

    @abstractmethod
    def forward_model(self, state: torch.Tensor, control_input: Dict[str, torch.Tensor]):
        raise NotImplementedError

    def forward_model_jac(self, state, control_input):
        """Autograd jacobian w.r.t state.
        Override to implement own jacobian"""
        return jacrev(self.forward_model, 0)(state, control_input)

    def predict(self, u, custom_motion_noise=None):
        u = torch.atleast_1d(u) # Force scalar to be 1D

        self.state_mean = self.forward_model(self.state_mean, u)
        G_t = self.forward_model_jac(self.state_mean, u)

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

    def update_with_specific_meas(self, meas_dict,
                                  implicit_meas_model,
                                  custom_measurement_noise: Optional[Dict] = None):
        #TODO: this doesn't work with current ActiveInterconnection implementation
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
        
        K_part_2 = torch.atleast_2d(K_part_2) # in case state is 1D 

        if implicit_meas_model.regularize_kalman_gain:
            K_part_2 += 1e-6 * torch.eye(K_part_2.shape[0],
                                         dtype=self.dtype,
                                         device=self.state_mean.device)
        try:
            K_part_2_inv = torch.inverse(K_part_2)
        except RuntimeError as e:
            # TODO convert this to log() instead of print()            
            print(f"WARN: Kalman update was not possible as {e}. Filter should (normally) recover soon.")
            # HACK resetting covariances. Maybe a more systematic approach is warranted
            self.set_state(torch.zeros_like(self.state[0]))
            return

        kalman_gain = torch.matmul(K_part_1, K_part_2_inv)

        kalman_gain = torch.atleast_2d(kalman_gain) # in case state is 1D

        # TODO: Write down WHY MINUS instead of PLUS
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

    def call_update_with_specific_meas(self, active_interconnection, buffer_dict: Dict[str, torch.Tensor]):
        args_to_be_passed = ('update_with_specific_meas', active_interconnection)
        kwargs = {'meas_dict': active_interconnection.get_state_dict(buffer_dict, self.id)}
        return functional_call(self, buffer_dict[self.id], args_to_be_passed, kwargs)

# ==================================== Specific Implementations =====================================================

class Robot_Vel_Estimator(RecursiveEstimator):
    def __init__(self, device):
        super().__init__("RobotVel", 3, device)

    def forward_model(self, x_mean: torch.Tensor, u: torch.Tensor):
        """
        Args:
            x_mean: Mean of the state
            u: Control input
                u[0]: Frontal acceleration  float
                u[1]: Lateral acceleration  float
                u[2]: Angular acceleration  float
                u[3]: del_t                 float
            """
        ret_mean = torch.empty_like(x_mean)
        vel_rot_new = x_mean[2] + u[2] * u[3]
        theta = vel_rot_new * u[3]
        rotation_matrix = torch.stack([
            torch.stack([torch.cos(-theta), -torch.sin(-theta)]),
            torch.stack([torch.sin(-theta), torch.cos(-theta)]),
        ]).squeeze()
        vel_trans_new = torch.matmul(rotation_matrix, x_mean[:2] + u[:2] * u[3])
        ret_mean[:2] = vel_trans_new
        ret_mean[2] = vel_rot_new
        return ret_mean
    
class Polar_Pos_Estimator_External_Vel(RecursiveEstimator):
    """
    Estimator for Target state x:
    x[0]: target offset angle
    x[1]: target distance
    """
    def __init__(self, device, id: str):
        super().__init__(id, 2, device)

    def forward_model(self, x_mean: torch.Tensor, u: torch.Tensor):
        ret_mean = torch.empty_like(x_mean)
        robot_vel = u[:2]

        # timestep = u[3]
        # rotation_matrix = torch.stack([
        #     torch.stack([torch.cos(-u[2]*timestep), -torch.sin(-u[2]*timestep)]),
        #     torch.stack([torch.sin(-u[2]*timestep), torch.cos(-u[2]*timestep)]),
        # ]).squeeze()
        # robot_vel = torch.matmul(rotation_matrix, robot_vel)

        robot_target_frame_rotation_matrix = torch.stack([
            torch.stack([torch.cos(-x_mean[0]), -torch.sin(-x_mean[0])]),
            torch.stack([torch.sin(-x_mean[0]), torch.cos(-x_mean[0])]),
        ]).squeeze()
        robot_target_frame_vel = torch.matmul(robot_target_frame_rotation_matrix, robot_vel)

        ret_mean[0] = x_mean[0] + (u[0]*torch.sin(x_mean[0]) - u[1]*torch.cos(x_mean[0]) - u[2]) * u[3]
        ret_mean[1] = (x_mean[1].pow(2) + (robot_target_frame_vel[1]*u[3]).pow(2)).sqrt() - robot_target_frame_vel[0] * u[3]

        return ret_mean
    
class Polar_Pos_Estimator_Internal_Vel(RecursiveEstimator):
    """
    Estimator for Target state x:
    x[0]: target offset angle
    x[1]: target distance
    """
    def __init__(self, device, id: str):
        super().__init__(id, 4, device)

    def forward_model(self, x_mean: torch.Tensor, u: torch.Tensor):
        ret_mean = torch.empty_like(x_mean)
        ret_mean[0] = x_mean[0] + x_mean[2] * u[0]
        ret_mean[1] = x_mean[1] + x_mean[3] * u[0]
        ret_mean[2:] = x_mean[2:]
        return ret_mean

class Pos_Estimator_Internal_Vel(RecursiveEstimator):
    """
    Estimator for Target state x:
    x[0]: target frontal offset
    x[1]: target lateral offset
    """
    def __init__(self, device, id: str):
        super().__init__(id, 5, device)

    def forward_model(self, x_mean, u):
        ret_mean = torch.empty_like(x_mean)
        timestep = u[0]
        rotation_matrix = torch.stack([
            torch.stack([torch.cos(x_mean[4]*timestep), -torch.sin(x_mean[4]*timestep)]),
            torch.stack([torch.sin(x_mean[4]*timestep), torch.cos(x_mean[4]*timestep)]),
        ]).squeeze()
        ret_mean[:2] = torch.matmul(rotation_matrix, x_mean[:2] + timestep * x_mean[2:4])
        ret_mean[2:4] = torch.matmul(rotation_matrix, x_mean[2:4])
        ret_mean[4] = x_mean[4]
        return ret_mean

# Same as Target Pos estimator - unnecessary
class Pos_Estimator_External_Vel(RecursiveEstimator):
    """
    Estimator for Target state x:
    x[0]: target frontal offset
    x[1]: target lateral offset
    """
    def __init__(self, device, id: str):
        super().__init__(id, 2, device)

    def forward_model(self, x_mean, u):
        ret_mean = torch.empty_like(x_mean)
        robot_vel_rot = u[2]
        timestep = u[3]
        rotation_matrix = torch.stack([
            torch.stack([torch.cos(-robot_vel_rot*timestep), -torch.sin(-robot_vel_rot*timestep)]),
            torch.stack([torch.sin(-robot_vel_rot*timestep), torch.cos(-robot_vel_rot*timestep)]),
        ]).squeeze()
        ret_mean = torch.matmul(rotation_matrix, x_mean - timestep * u[:2])
        return ret_mean
    
class Target_Pos_Estimator_No_Forward(RecursiveEstimator):
    """
    Estimator for Target state x:
    x[0]: target frontal offset
    x[1]: target lateral offset
    """
    def __init__(self, device):
        super().__init__("TargetPosNoForward", 2, device)

    def forward_model(self, x_mean, u):
        return x_mean
    
class Obstacle_Rad_Estimator(RecursiveEstimator):
    """
    Estimator for obstacle radius state x:
    x[0]: obstacle radius
    """
    def __init__(self, device, id: str):
        super().__init__(id, 1, device)
        self.set_static_motion_noise(torch.eye(1, device=device)*1e-1)

    def forward_model(self, x_mean, u):
        return x_mean

class Target_Distance_Estimator(RecursiveEstimator):
    """
    Estimator for Target state x:
    x[0]: target angular offset
    x[1]: target distance
    x[2]: del target angular offset
    x[3]: del target distance
    """
    def __init__(self, device):
        super().__init__("TargetDist", 2, device)

    def forward_model(self, x_mean, u):
        ret_mean = torch.empty_like(x_mean)

        robot_vel_rot = u[2]
        target_angular_offset = x_mean[0]
        timestep = u[3]

        rotation_matrix = torch.stack([
            torch.stack([torch.cos(-target_angular_offset), -torch.sin(-target_angular_offset)]),
            torch.stack([torch.sin(-target_angular_offset), torch.cos(-target_angular_offset)]),
        ]).squeeze()
        target_frame_vel = torch.matmul(rotation_matrix, u[:2])
        ret_mean[0] = x_mean[0] - timestep * robot_vel_rot
        ret_mean[1] = x_mean[1] - timestep * target_frame_vel[0]
        return ret_mean

# =========================================================================================

if __name__ == "__main__":
    print("Testing: Multiple Implicit Measurement EKF")