from abc import ABC, abstractmethod
from typing import Dict, List

from components.measurement_model import ImplicitMeasurementModel
    
# ====================================================================================================================================

class ActiveInterconnection(ABC, ImplicitMeasurementModel):
    def __init__(self, required_estimators, device=None):
        super().__init__(required_states=required_estimators, device=device)

    def implicit_measurement_model(self, x, meas_dict):
        """
        Hacky function transforming the implicit measurement model for a specific x into an interconnection model for any connected state.
        """
        missing_key = next(key for key in self.connected_states.keys() if key not in meas_dict)
        assert sum(1 for key in self.connected_states.keys() if key not in meas_dict) == 1, "There should be exactly one missing key"
        meas_dict[missing_key] = x
        return self.implicit_interconnection_model(meas_dict)

    @abstractmethod
    def implicit_interconnection_model(self, meas_dict):
        pass