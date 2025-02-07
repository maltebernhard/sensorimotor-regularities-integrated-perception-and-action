from abc import ABC, abstractmethod
from typing import Dict, List
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

    def set_static_measurement_noise(self, meas_name: str, noise_cov: torch.Tensor) -> None:
        assert meas_name in self.meas_config.keys(), (
            f"Measurement name {meas_name} not found in {self.meas_config.keys()}")
        assert noise_cov.shape == (self.meas_config[meas_name], self.meas_config[meas_name]), (
            f"Measurement noise shape {noise_cov.shape} does not match {self.meas_config[meas_name]}")
        self.register_buffer(
            f'_Q_{meas_name}',
            noise_cov.to(self.dtype),
            persistent=False)
        setattr(self, f'_Q_{meas_name}', noise_cov.to(self.dtype))

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

    def implicit_measurement_model(self, x, meas_dict):
        """
        Hacky function transforming the implicit measurement model for a specific x into an interconnection model for any connected state.
        """
        missing_key = next(key for key in self.connected_states.keys() if key not in meas_dict)
        assert sum(1 for key in self.connected_states.keys() if key not in meas_dict) == 1, "There should be exactly one missing key"
        meas_dict[missing_key] = x
        return self.implicit_interconnection_model(meas_dict)

    def set_connected_states(self, connected_states: List[State]) -> None:
        assert set(state.id for state in connected_states) == set(self.required_estimators + self.required_observations), f"Estimators should be {self.required_estimators} and Observations should be {self.required_observations}, but are {[state.id for state in connected_states]}"
        self.connected_states: Dict[str, State] = {state.id: state for state in connected_states}
        self.meas_config = {state.id: state.state_dim for state in connected_states}
        # Initialize static measurement noise to identity for each measurement
        for key in self.meas_config.keys():
            self.set_static_measurement_noise(key, torch.eye(self.meas_config[key]))

    def get_expected_obs_noise(self, buffer_dict: Dict[str,torch.Tensor]) -> Dict[str, torch.Tensor]:
        """
        returns the expected sensor noise for each sensory component. OVERWRITE for SMCs to include additional effects, such as foveal vision noise
        """
        return {key: self.connected_states[key].sensor_noise for key in self.required_observations}
    
    def all_observations_updated(self):
        all_obs = [obs for obs in self.connected_states.values() if type(obs) == Observation]
        if len(all_obs) == 0:
            raise ValueError("No observations connected to this model")
        return all(obs.updated for obs in all_obs)

    def get_state_dict(self, buffer_dict: dict, estimator_id):
        return {id: buffer_dict[id]['mean'] for id in self.connected_states.keys() if id != estimator_id}
    
    def get_cov_dict(self, buffer_dict: dict, estimator_id):
        cov_dict = {id: buffer_dict[id]['cov'] + buffer_dict[id]['update_uncertainty'] for id in self.required_estimators if id != estimator_id}
        obs_noise = self.get_expected_obs_noise(buffer_dict)
        for key in self.required_observations:
            cov_dict[key] = obs_noise[key].pow(2) + self.connected_states[key].update_uncertainty
        return cov_dict

    @abstractmethod
    def implicit_interconnection_model(self, meas_dict):
        pass

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
        returns tuple of estimated means and covariances of sensory components. NEEDS to be overwritten by user
        """
        pass

    def get_contingent_noise(self, state: Dict[str,torch.Tensor]) -> Dict[str,torch.Tensor]:
        """
        returns the expected noise additional to basic sensor noise. OVERWRITE for SMCs to include additional effects, such as foveal vision noise
        """
        return {key: 0.0 for key in self.required_observations}

    def get_expected_obs_noise(self, buffer_dict: Dict[str,torch.Tensor]) -> Dict[str,torch.Tensor]:
        """
        returns the expected sensor noise for each sensory component. OVERWRITE to include additional effects, such as foveal vision noise
        """
        predicted_meas = self.get_predicted_meas(buffer_dict[self.state_component]['mean'], buffer_dict[self.action_component]['mean'])
        contingent_noise = self.get_contingent_noise(buffer_dict[self.state_component]['mean'])
        noise = {}
        for key in self.required_observations:
            if "vel" in key:
                noise[key] = (self.connected_states[key].sensor_noise + contingent_noise[key]) * torch.abs(predicted_meas[key])
            elif "distance" in key:
                if not "dot" in key:
                    noise[key] = (self.connected_states[key].sensor_noise + contingent_noise[key]) * torch.abs(predicted_meas[key])
                elif "dot" in key:
                    noise[key] = (self.connected_states[key].sensor_noise + contingent_noise[key]) * predicted_meas[key.replace("_dot","")]
            else:
                noise[key] = self.connected_states[key].sensor_noise + contingent_noise[key]
        return noise
    
    def get_state_dict_with_expected_meas(self, buffer_dict: dict, estimator_id):
        state_dict = {key: buffer_dict[key]['mean'] for key in self.required_estimators if key != estimator_id}
        predicted_meas = self.get_predicted_meas(buffer_dict[self.state_component]['mean'], buffer_dict[self.action_component]['mean'])
        for key in self.required_observations:
            state_dict[key] = predicted_meas[key]
        return state_dict
