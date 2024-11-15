from abc import ABC, abstractmethod
from typing import Dict
import torch
from torch.nn import Module
from torch.func import jacrev

# ===================================================================================================

class ImplicitMeasurementModel(Module):
    """
    Abstract interface to define a measurement model using implicit function and
    optionally its Jacobian. This will be used by MIMEKF.

    Notation as in Thrun et al., Probabilistic Robotics
    """
    def __init__(self,
                 meas_config: dict,
                 device='cpu',
                 outlier_rejection_enabled=False,
                 outlier_threshold=1.0,
                 regularize_kalman_gain=False,
                 dtype=torch.float64) -> None:
        """
        Args: 
            state_dim: Dimension of the state this measurement model will be attached to
            meas_config: A dict of the form {'meas1': meas1_dim, 'meas2': meas2_dim, ...}
            dtype: Data type of the tensors
        """
        super().__init__()

        self.device = device
        self.meas_config = meas_config
        self.dtype = dtype

        # Initialize static measurement noise to identity for each measurement
        for key in self.meas_config.keys():
            self.register_buffer(
                f'_Q_{key}',
                torch.eye(self.meas_config[key], dtype=self.dtype),
                persistent=False)
            
        self.outlier_rejection_enabled = outlier_rejection_enabled
        self.register_buffer(
            'outlier_distance_threshold',
            torch.tensor(outlier_threshold, dtype=dtype),
            persistent=False)
        self.regularize_kalman_gain = regularize_kalman_gain

        self.to(device)
            
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
        # TODO: This dosn't work with current ActiveInterconnection implementation
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

# ========================= Standard Implicit Measurement Model: Innovation Space = State Space ===========================

class StandardImplicitMeasurementModel(ImplicitMeasurementModel):
    def __init__(self, meas_config: dict, device, dtype=torch.float64) -> None:
        super().__init__(meas_config, device, dtype=dtype)
    
    def implicit_measurement_model(self, x, meas_dict):
        return torch.stack([torch.norm(x[i] - meas_dict[key]) for i, key in enumerate(self.meas_config.keys())]).squeeze()
    
# ======================================== Specific Implementations =======================================================

# measurement model between robot velocity state and robot_velocity measurement
class Robot_Vel_MM(ImplicitMeasurementModel):
    """
    Measurement Model:
    state:          robot velocity
    measurement:    robot velocity
    """
    def __init__(self, device) -> None:
        super().__init__({"robot_vel" : 3}, device)
    
    def implicit_measurement_model(self, x, meas_dict):
        robot_vel = meas_dict["robot_vel"]
        return torch.stack([robot_vel[0] - x[0], robot_vel[1] - x[1], robot_vel[2] - x[2]]).squeeze()

# measurement model between polar position state and angular offset measurement
class Polar_Pos_Vel_MM(ImplicitMeasurementModel):
    """
    Measurement Model:
    state:          estimated position
                    x[0] = theta
                    x[1] = r
                    x[2] = del_theta
                    x[3] = del_r
    measurement:    angular offset
                    robot_vel
    """
    def __init__(self, device) -> None:
        super().__init__({'offset_angle': 1, 'del_offset_angle': 1, 'robot_vel': 3}, device)

    def implicit_measurement_model(self, x, meas_dict):
        robot_target_frame_rotation_matrix = torch.stack([
            torch.stack([torch.cos(-x[0]), -torch.sin(-x[0])]),
            torch.stack([torch.sin(-x[0]), torch.cos(-x[0])]),
        ]).squeeze()
        robot_target_frame_vel = torch.matmul(robot_target_frame_rotation_matrix, meas_dict['robot_vel'][:2])
        return torch.stack([
            meas_dict['offset_angle'] - x[0],
            #torch.tensor([x[2] + meas_dict['robot_vel'][2] - torch.atan2(robot_target_frame_vel[1], x[1])], device=x.device),
            meas_dict['del_offset_angle'] - x[2],
            torch.tensor([x[3] - robot_target_frame_vel[0]], device=x.device),
        ]).squeeze()

# measurement model between position state and angular offset measurement
class Pos_MM(ImplicitMeasurementModel):
    """
    Measurement Model:
    state:          target position
    measurement:    angular offset
    """
    def __init__(self, device) -> None:
        super().__init__({'offset_angle': 1}, device)

    def implicit_measurement_model(self, x, meas_dict):
        # avoids NaN from atan2(0,0)
        if x[1] == 0.0 and x[0] == 0.0:
            angle = torch.tensor([0.0]).to(x.device)
        else:
            angle = torch.atan2(x[1],x[0])
        angle_diff = meas_dict['offset_angle'] - angle
        return ((angle_diff + torch.pi) % (2*torch.pi)) - torch.pi
    
# measurement model between target position state and angular offset measurement
class Pos_Vel_MM(ImplicitMeasurementModel):
    """
    Measurement Model:
    state:          target position
    measurement:    angular offset
    """
    def __init__(self, device) -> None:
        super().__init__({'offset_angle': 1, 'robot_vel': 3}, device)

    def implicit_measurement_model(self, x, meas_dict):
        # avoids NaN from atan2(0,0)
        if x[1] == 0.0 and x[0] == 0.0:
            angle = torch.tensor([0.0]).to(x.device)
        else:
            angle = torch.atan2(x[1],x[0])
        angle_diff = meas_dict['offset_angle'] - angle
        return torch.concat([
            ((angle_diff + torch.pi) % (2*torch.pi)) - torch.pi,
            - meas_dict['robot_vel'] - x[2:],
        ]).squeeze()

class Radius_MM(ImplicitMeasurementModel):
    """
    Measurement Model:
    state:          object radius
    measurement:    object distance and angle in visual field
    """
    def __init__(self, device) -> None:
        super().__init__({'pos': 2, 'visual_angle': 1}, device)

    def implicit_measurement_model(self, x, meas_dict):
        # TODO: bad fix: minimum to ensure asin doesn't output NaN
        return torch.asin(torch.minimum(torch.ones_like(x[0]), x[0] / meas_dict['pos'].norm())) - meas_dict['visual_angle'] / 2

# measurement model between target position state and angular offset measurement
class Target_Dist_MM(ImplicitMeasurementModel):
    """
    Measurement Model:
    state:          target position
    measurement:    angular offset
    """
    def __init__(self, device) -> None:
        super().__init__({'robot_vel': 3, 'target_offset_angle': 1, 'del_target_offset_angle': 1}, device)

    def implicit_measurement_model(self, x, meas_dict):
        meas_theta = - meas_dict['target_offset_angle']
        meas_rotation_matrix = torch.tensor([[torch.cos(meas_theta), -torch.sin(meas_theta)], [torch.sin(meas_theta), torch.cos(meas_theta)]], device=x.device)
        meas_robot_target_frame_vel = torch.matmul(meas_rotation_matrix, meas_dict['robot_vel'][:2])
        
        measured_del_angle_translation = meas_dict['del_target_offset_angle'] + meas_dict['robot_vel'][2]   # if robot doesn't move lateral to target, these should cancel out
        estimated_del_angle_translation = - torch.atan2(meas_robot_target_frame_vel[1], x[1])               # estimated angular vel of target due to lateral translation, based on distance estimate
        
        return torch.stack([
            meas_dict['target_offset_angle'] - x[0],
            torch.atleast_1d(measured_del_angle_translation - estimated_del_angle_translation),
        ]).squeeze()