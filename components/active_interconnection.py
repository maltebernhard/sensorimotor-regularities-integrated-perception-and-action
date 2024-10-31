from abc import ABC, abstractmethod

class ActiveInterconnection(ABC):
    def __init__(self):
        super().__init__()
        self.connected_estimators = {}

    @abstractmethod
    def connector_function(self, estimator):
        raise NotImplementedError