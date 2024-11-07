
from abc import ABC, abstractmethod
from typing import Dict, List
from components.active_interconnection import ActiveInterconnection
from components.estimator import RecursiveEstimator
from components.goal import Goal


class AICON(ABC):
    def __init__(self, device, env, REs, AIs, goals):
        self.device = device
        self.env = env
        self.REs: Dict[str, RecursiveEstimator] = REs
        self.AIs: Dict[str, ActiveInterconnection] = AIs
        self.goals: List[Goal] = goals

    @abstractmethod
    def eval_step(self, action):
        raise NotImplementedError
    
    