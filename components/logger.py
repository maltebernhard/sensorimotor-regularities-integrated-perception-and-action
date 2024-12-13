from typing import Dict, List
import numpy as np
import torch
import yaml
import os

# ==================================================================================

class AICONLogger:
    def __init__(self):
        self.data: List[dict] = []

    def log(self, step: int, time: float, estimators: Dict[str,Dict[str,torch.Tensor]], reality: Dict[str,torch.Tensor], observation: Dict[str,Dict[str,torch.Tensor]]):
        real_state = {}
        for key in estimators.keys():
            if key[:5] == "Polar" and key[-3:] == "Pos":
                obj = key[5:-3].lower()
                real_state[key] = [
                    reality[f"{obj}_distance"],
                    reality[f"{obj}_offset_angle"],
                    reality[f"del_{obj}_distance"],
                    reality[f"del_{obj}_offset_angle"]
                ]
            elif key == "RobotVel":
                real_state[key] = [
                    reality["vel_frontal"],
                    reality["vel_lateral"],
                    reality["vel_rot"],
                ]
            elif key[-6:] == "Radius":
                obj = key[:-6].lower()
                real_state[key] = reality[f"{obj}_radius"]

        self.data.append({
            "step": step,
            "time": time,
            "estimators": estimators,
            "observation": observation,
            "reality": real_state,
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