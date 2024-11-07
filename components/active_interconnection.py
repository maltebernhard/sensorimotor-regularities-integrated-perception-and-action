from abc import ABC, abstractmethod
from typing import Dict

from components.estimator import RecursiveEstimator
from components.measurement_model import ImplicitMeasurementModel, StandardImplicitMeasurementModel

# =========================================================================================

class ActiveInterconnection:
    def __init__(self, id, innovation_space_dict: Dict[str, int] = None) -> None:
        self.id: str = id
        self.innovation_space_dict: Dict[str, int] = innovation_space_dict
        self.connected_estimators: Dict[str, RecursiveEstimator] = {}
        self.connected_measurement_models: Dict[str, ImplicitMeasurementModel] = {}

    def add_estimator(self, estimator: RecursiveEstimator, estimator_measurement_model: ImplicitMeasurementModel):
        self.connected_estimators[estimator.id] = estimator
        self.connected_measurement_models[estimator.id] = estimator_measurement_model

    def update_estimator(self, estimator_id):
        assert estimator_id in self.connected_estimators.keys(), f"Invalid estimator_id: {estimator_id}"
        print({id: self.connected_estimators[id].state_mean for id in [key for key in self.connected_estimators.keys() if key != estimator_id]})
        self.connected_estimators[estimator_id].update_with_specific_meas(
            {id: self.connected_estimators[id].state_mean for id in [key for key in self.connected_estimators.keys() if key != estimator_id]},
            self.connected_measurement_models[estimator_id]
        )

    def get_states(self):
        return {id: self.connected_estimators[id].state_mean for id in self.connected_estimators.keys()}