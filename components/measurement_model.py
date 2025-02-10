from abc import ABC, abstractmethod
from typing import Dict, List, Tuple
import torch
from torch.nn import Module
from torch.func import jacrev

from components.estimator import Observation, State
from typing import Type
    
# ====================================================================================================================================

class ImplicitMeasurementModel(Module):
    """
    Code taken and adapted from Battaje, Aravind.
    Abstract interface to define a measurement model using implicit function and
    optionally its Jacobian.

    Notation as in Thrun et al., Probabilistic Robotics
    """
    def __init__(self,
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
        """Override to implement own jacobian"""
        # NOTE this check works correctly in Python 3+ only
        # NOTE: This dosn't work with current ActiveInterconnection implementation
        # assert meas_dict.keys() == self.meas_config.keys(), (
        #     'All measurements must be provided as mentioned in meas_config')
        jacobians, implicit_measurement_model_eval = jacrev(
            self._implicit_measurement_model_with_aux,
            argnums=(0, 1), # x and meas_dict (all measurements within the dict!)
            has_aux=True)(x, meas_dict)
        
        # NOTE following is a hacky code to ensure that the jacobians are atleast 2D
        # But it fails in the following:
        # For the following two cases, the jacobian will be a 1D tensor
        #   1) A function taking in a scalar as input and returning a vector
        #   2) A function taking in a vector as input and returning a scalar
        # and so it is impossible to tell the difference between the two
        # However, if the inputs are ensured to be atleast 1D, then the following
        # is not not necessary. Retaining this code for posterity.
        # H_t = torch.atleast_2d(jacobians[0])
        # F_t_dict = {}
        # for key in jacobians[1].keys():
        #     F_t_dict[key] = torch.atleast_2d(jacobians[1][key])
        # SO NOTE BIG TAKE AWAY: MAKE SURE INPUTS (meas) ARE ATLEAST 1D!
        
        H_t = jacobians[0]
        F_t_dict = jacobians[1]

        return H_t, F_t_dict, implicit_measurement_model_eval
    
    def forward(self):
        # Multiple measurement models constrain how
        # KF goes through normal iteration. So predict
        # and multiple updates maybe asynchronous.
        # So, don't allow forward calls!
        raise NotImplementedError
    
# =======================================================================================

class ActiveInterconnection(ABC, ImplicitMeasurementModel):
    def __init__(self, required_estimators, required_observations=[]):
        super().__init__()
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
        assert set(state.id for state in states) == set(self.required_estimators), f"Estimators should be {self.required_estimators}, but are {[state.id for state in states]}"
        assert set(obs.id for obs in observations) == set(self.required_observations), f"Observations should be {self.required_observations}, but are {[obs.id for obs in observations]}"
        self.connected_states: Dict[str, State] = {state.id: state for state in states}
        self.connected_observations: Dict[str, Observation] = {obs.id: obs for obs in observations}
        self.meas_config = {state.id: state.state_dim for state in states}
        self.meas_config.update({obs.id: obs.dim for obs in observations})

    def get_expected_meas_noise(self, buffer_dict: Dict[str,torch.Tensor]) -> Dict[str, Tuple[torch.Tensor,torch.Tensor]]:
        """
        returns the expected sensor noise (tuple of mean and stddev) for each sensory component.
        OVERWRITE for SMCs to include additional effects, such as foveal vision noise.
        """
        return {key: obs.static_sensor_noise for key, obs in self.connected_observations.items()}
    
    def all_observations_updated(self):
        return all(obs.updated for obs in self.connected_observations.values())

    def get_state_dict(self, buffer_dict: dict, estimator_id):
        state_dict = {state_key: buffer_dict[state_key]['mean'] for state_key in list(self.connected_states.keys()) if state_key != estimator_id}
        state_dict.update({obs_key: obs.last_measurement for obs_key, obs in self.connected_observations.items()})
        return state_dict

    def get_cov_dict(self, buffer_dict: dict, estimator_id) -> Tuple[Dict[str,torch.Tensor],Dict[str,torch.Tensor]]:
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
        pass

# =======================================================================================

class SensorimotorContingency(ActiveInterconnection):
    def __init__(self, state_component: str, action_component: str, sensory_components: List[str]):
        self.state_component = state_component
        self.action_component = action_component
        super().__init__(required_estimators=[state_component, action_component], required_observations=sensory_components)

    def implicit_interconnection_model(self, meas_dict):
        predicted_meas = self.get_predicted_meas(meas_dict[self.state_component], meas_dict[self.action_component])
        return torch.atleast_1d(torch.stack([
            predicted_meas[key] - meas_dict[key] for key in self.required_observations
        ]).squeeze())

    @abstractmethod
    def get_predicted_meas(self, state, action) -> Dict[str,torch.Tensor]:
        """
        returns tuple of estimated means of sensory components. NEEDS to be overwritten by user
        """
        pass

    def get_contingent_noise(self, state: Dict[str,torch.Tensor]) -> Dict[str,Tuple[torch.Tensor,torch.Tensor]]:
        """
        returns the expected state-dependent noise (mean and stddev) for each measurement, additional to basic sensor noise.
        OVERWRITE for SMCs to include additional effects, such as foveal vision noise
        """
        return {key: (0.0,0.0) for key in self.required_observations}

    def get_expected_meas_noise(self, buffer_dict: Dict[str,torch.Tensor]) -> Dict[str,torch.Tensor]:
        """
        returns the expected total sensor noise (tuple of mean and stddev) for each sensory component.
        OVERWRITE to include additional effects, such as noise scaling with value (e.g. for robot vel)
        """
        contingent_noise = self.get_contingent_noise(buffer_dict[self.state_component]['mean'])
        noise = {key: (
            obs.static_sensor_noise[0] + contingent_noise[key][0],
            obs.static_sensor_noise[1] + contingent_noise[key][1]
        ) for key, obs in self.connected_observations.items()}
        return noise
    
    def get_state_dict_with_predicted_meas(self, buffer_dict: dict, estimator_id):
        state_dict = {key: buffer_dict[key]['mean'] for key in self.required_estimators if key != estimator_id}
        predicted_meas = self.get_predicted_meas(buffer_dict[self.state_component]['mean'], buffer_dict[self.action_component]['mean'])
        for key in self.required_observations:
            state_dict[key] = predicted_meas[key]
        return state_dict
