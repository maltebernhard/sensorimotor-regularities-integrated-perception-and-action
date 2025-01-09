
from typing import Dict
from abc import ABC, abstractmethod
import torch
from torch.nn import Module

# ===================================================================================

class Goal(ABC, Module):
    def __init__(self, device=None):
        super().__init__()
        self.device = device if device is not None else torch.get_default_device()
    
    @abstractmethod
    def loss_function(self, buffer_dict) -> float:
        raise NotImplementedError
    
    def loss_function_from_buffer(self, buffer_dict):
        return self.loss_function(buffer_dict)
