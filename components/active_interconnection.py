from abc import ABC, abstractmethod
from typing import Dict, List

from components.estimator import Observation, RecursiveEstimator
from components.measurement_model import ImplicitMeasurementModel
    
# ====================================================================================================================================

class MeasurementModel(ABC, ImplicitMeasurementModel):
    def __init__(self, estimator: str, required_observations: List[str], device):
        meas_config = {obs: 1 for obs in required_observations}
        super().__init__(meas_config=meas_config, device=device)
        self.estimator = estimator
        self.observations = required_observations

    @abstractmethod
    def implicit_measurement_model(self, x, meas_dict):
        raise NotImplementedError
    
# ====================================================================================================================================

class ActiveInterconnection(ABC, ImplicitMeasurementModel):
    def __init__(self, estimators: List[RecursiveEstimator], required_estimators, device, propagate_meas_uncertainty=True):
        assert set(estimator.id for estimator in estimators) == set(required_estimators), f"Estimators should be {required_estimators}"
        meas_config = {estimator.id: estimator.state_mean.size().numel() for estimator in estimators}
        super().__init__(meas_config=meas_config, device=device)
        self.connected_estimators: Dict[str, RecursiveEstimator] = {est.id: est for est in estimators}
        self.propagate_meas_uncertainty = propagate_meas_uncertainty

    def get_state_dict(self, buffer_dict, estimator_id):
        return {id: buffer_dict[id]['state_mean'] for id in self.connected_estimators.keys() if id != estimator_id}
    
    def get_cov_dict(self, buffer_dict, estimator_id):
        if self.propagate_meas_uncertainty:
            return {id: buffer_dict[id]['state_cov'] for id in self.connected_estimators.keys() if id != estimator_id}
        else:
            return None

    def implicit_measurement_model(self, x, meas_dict):
        missing_key = next(key for key in self.connected_estimators.keys() if key not in meas_dict)
        assert sum(1 for key in self.connected_estimators.keys() if key not in meas_dict) == 1, "There should be exactly one missing key"
        meas_dict[missing_key] = x
        return self.implicit_interconnection_model(meas_dict)

    @abstractmethod
    def implicit_interconnection_model(self, meas_dict):
        pass