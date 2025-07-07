from typing import Dict
import mujoco
import os
import subprocess
import numpy as np
import mujoco.viewer
import time

from components.environment import BaseEnv, Observation
from components.helpers import world_to_rtf_numpy

# =================================================================================

class MujocoEnv(BaseEnv):
    def __init__(self, config_path):
        super().__init__()

        self.setup_gpu_rendering()

        # Load the XML model for the environment from a local file
        with open(config_path, 'r') as f:
            MODEL_XML = f.read()

        # Load the model
        self.model = mujoco.MjModel.from_xml_string(MODEL_XML)
        # Create the data object for the simulation
        self.data = mujoco.MjData(self.model)
        # Initialize the viewer
        self.viewer = None

        self.generate_observation_space()

        self.current_step = 0
        self.time = 0.0
        self.real_state_history = {}
        self.observation_history = {}
        
    def setup_gpu_rendering(self):
        # Check GPU availability
        if subprocess.run('nvidia-smi').returncode:
            raise RuntimeError(
                'Cannot communicate with GPU. '
                'Make sure you are using a GPU runtime. '
                'Go to the Runtime menu and select "Choose runtime type".'
            )
        # Add an ICD config for Nvidia EGL driver
        NVIDIA_ICD_CONFIG_PATH = '/usr/share/glvnd/egl_vendor.d/10_nvidia.json'
        if not os.path.exists(NVIDIA_ICD_CONFIG_PATH):
            with open(NVIDIA_ICD_CONFIG_PATH, 'w') as f:
                f.write("""{
                "file_format_version" : "1.0.0",
                "ICD" : {
                    "library_path" : "libEGL_nvidia.so.0"
                }
            }
            """)
        # Configure MuJoCo to use the EGL rendering backend
        os.environ['MUJOCO_GL'] = 'egl'
        # Clear any previous console prints
        os.system('clear')

    def step(self, action):
        self.data.qvel[:3] = action

        # TODO: moving Target
        # self.data.qvel[6] = 1.0

        self.current_step += 1
        self.time += 0.01
        mujoco.mj_step(self.model, self.data)
        #mujoco.mj_forward(self.model, self.data)

        print(self.data.body('robot').xpos)
        print(self.data.body('target').xpos)

        self.last_state = self._get_state()

        self.last_observation, noise = self.get_observation_from_state(self.last_state.copy())
        # add observation to history
        self.real_state_history[self.current_step] = self.last_state.copy()
        self.observation_history[self.current_step] = (self.last_observation.copy(), noise.copy())
        if self.current_step - 2 in self.observation_history:
            del self.observation_history[self.current_step - 2]
            del self.real_state_history[self.current_step - 2]
        
        return np.array(list(self.last_observation.values())), None, None, None, None

    def reset(self, seed=None, **kwargs):
        # Reset the mujoco simulation
        mujoco.mj_resetData(self.model, self.data)
        super().reset(seed=seed)
        np.random.seed(seed)
        self.current_step = 0

        self.last_state, info = self._get_state(), None
        self.last_observation, noise = self.get_observation_from_state(self.last_state.copy())
        self.real_state_history[self.current_step] = self.last_state.copy()
        self.observation_history[self.current_step] = (self.last_observation.copy(), noise.copy())

        return np.array(list(self.last_observation.values())), None

    def render(self):
        if self.viewer is None:
            # If the viewer is not initialized, create it
            self.viewer = mujoco.viewer.launch_passive(
                model=self.model,
                data=self.data,
                show_left_ui=False,
                show_right_ui=False
            )
            # Enable the global coordinate frame visualization
            self.viewer.opt.frame = mujoco.mjtFrame.mjFRAME_GEOM
        self.viewer.sync()

    def close(self):
        self.viewer.close() if self.viewer is not None else None
    
    # --------------------------------------------------

    def generate_observation_space(self):
        self.observations: Dict[str, Observation] = {
            "vel_x":        Observation(-1, 1, self.get_vel_x),
            "vel_y":        Observation(-1, 1, self.get_vel_y),
            "vel_z":        Observation(-1, 1, self.get_vel_z),

            "target_distance": Observation(0.0, np.inf, self.get_target_distance),
            "target_distance_dot": Observation(-1, 1, self.get_target_distance_dot),

            "target_phi":   Observation(-np.pi, np.pi, self.get_target_phi),
            "target_theta": Observation(-np.pi, np.pi, self.get_target_theta),
            "target_phi_dot": Observation(-np.inf, np.inf, self.get_target_phi_dot),
            "target_theta_dot": Observation(-np.inf, np.inf, self.get_target_theta_dot),
        }

    def get_vel_x(self):
        """Get the x-component of the robot's velocity."""
        return self.data.qvel[0]
    def get_vel_y(self):
        """Get the y-component of the robot's velocity."""
        return self.data.qvel[1]
    def get_vel_z(self):
        """Get the z-component of the robot's velocity."""
        return self.data.qvel[2]
    def get_target_distance(self):
        """Get the distance between the robot and the target."""
        return np.linalg.norm(self.data.qpos[7:10] - self.data.qpos[:3])
    def get_target_distance_dot(self):
        """Get the rate of change of the distance between the robot and the target."""
        relative_pos = self.data.body('target').xpos - self.data.body('robot').xpos
        relative_vel = self.data.qvel[6:9] - self.data.qvel[:3]
        return np.dot(relative_pos, relative_vel) / np.linalg.norm(relative_pos) if np.linalg.norm(relative_pos) > 0 else 0.0
    def get_target_phi(self):
        """Get the angle between the robot and the target in the x-y plane."""
        relative_pos = self.data.body('target').xpos - self.data.body('robot').xpos
        if np.allclose(relative_pos[:2], 0):
            return 0.0
        angle = np.arctan2(relative_pos[1], relative_pos[0])
        return (angle + np.pi) % (2 * np.pi) - np.pi
    def get_target_theta(self):
        relative_pos = self.data.body('target').xpos - self.data.body('robot').xpos
        relative_dist = np.linalg.norm(relative_pos)
        if relative_dist == 0:
            return 0.0
        z_div_dist = np.clip(relative_pos[2] / relative_dist, -1.0, 1.0)
        return np.arccos(z_div_dist)
    def get_target_phi_dot(self):
        """Get the rate of change of the target's phi (azimuthal) angle."""
        # Relative position and velocity in world coordinates
        relative_pos = self.data.body('target').xpos - self.data.body('robot').xpos
        vel = self.data.qvel[6:9] - self.data.qvel[:3]
        xy_dist_sq = relative_pos[0]**2 + relative_pos[1]**2

        if xy_dist_sq == 0:
            return 0.0

        # d(phi)/dt = (x*vy - y*vx) / (x^2 + y^2)
        phi_dot = (relative_pos[0] * vel[1] - relative_pos[1] * vel[0]) / xy_dist_sq
        return phi_dot

    # def get_target_theta_dot(self):
    #     """Get the rate of change of the target's theta (polar) angle."""
    #     # Relative position and velocity in world coordinates
    #     relative_pos = self.data.body('target').xpos - self.data.body('robot').xpos
    #     relative_vel = self.data.qvel[6:9] - self.data.qvel[:3]
    #     rtf_vel = world_to_rtf_numpy(relative_vel, self.get_target_phi(), self.get_target_theta())
    #     theta_dot = np.linalg.norm(relative_pos) / rtf_vel[2]
    #     return theta_dot

    def get_target_theta_dot(self):
        """
        Get the rate of change of the target's theta (polar) angle.
        """
        # Relative position and velocity in world coordinates
        relative_pos = self.data.body('target').xpos - self.data.body('robot').xpos
        relative_vel = self.data.qvel[6:9] - self.data.qvel[:3]
        x, y, z = relative_pos
        vx, vy, vz = relative_vel
        r = np.linalg.norm(relative_pos)
        if r == 0:
            return 0.0
        z_over_r = z / r
        # Avoid division by zero in denominator
        denom = r**2 * np.sqrt(1 - z_over_r**2)
        if denom == 0:
            return 0.0
        num = r * vz - z * (x * vx + y * vy + z * vz) / r
        theta_dot = num / denom
        return theta_dot  # Sign flip because theta is measured from the top (z axis)

    def generate_action_space(self):
        raise NotImplementedError
    
    # --------------------------------------------------

    def _get_state(self):
        """Computes a new observation."""
        return {key: obs.calculate_value() for key, obs in self.observations.items()}

    def get_observation_from_state(self, state: dict[str, float]):
        """Get the observation from the state."""
        # TODO: apply noise n stuff
        return state, {key: 0.0 for key in state.keys()}

    def get_state(self):
        """Return the current environment state."""
        try:
            obs = self.real_state_history[self.current_step].copy()
        except:
            raise Exception("I think this should never happen.")
            obs = self._get_state()
        return obs

    def get_observation(self):
        """Return the current, unnormalized observation."""
        try:
            obs = self.observation_history[self.current_step]
        except:
            raise Exception("I think this should never happen.")
            obs = self._get_state()
        return obs
