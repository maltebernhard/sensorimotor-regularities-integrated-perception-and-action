from typing import Dict, List
import numpy as np
import torch
import yaml
import os

# ==================================================================================

class AICONLogger:
    def __init__(self):
        self.data: List[dict] = []

    def log(self, step: int, time: float, estimators: Dict[str,Dict[str,torch.Tensor]], observations: Dict[str,torch.Tensor]):
        reality = {}
        for key in estimators.keys():
            if key[:5] == "Polar" and key[-3:] == "Pos":
                obj = key[5:-3]
                reality[key] = {
                    "state_mean": [
                        observations["target_distance"],
                        observations["target_offset_angle"],
                        observations["del_target_distance"],
                        observations["del_target_offset_angle"]
                    ],
                    "state_cov": torch.zeros(2,2)
                }

        self.data.append({
            "step": step,
            "time": time,
            "estimators": estimators,
            "reality": reality
        })

    def plot(self):
        pass

    def save(self, record_dir):
        def convert_tensors(obj):
            if isinstance(obj, torch.Tensor):
                return obj.tolist()
            elif isinstance(obj, dict):
                return {k: convert_tensors(v) for k, v in obj.items()}
            elif isinstance(obj, list):
                return [convert_tensors(i) for i in obj]
            elif isinstance(obj, np.ndarray):
                return obj.tolist()
            elif isinstance(obj, np.generic):
                return obj.item()
            else:
                return obj

        serializable_data = convert_tensors(self.data)
        os.makedirs(record_dir, exist_ok=True)
        with open(os.path.join(record_dir, "data.yaml"), 'w') as f:
            f.write(yaml.dump(serializable_data))