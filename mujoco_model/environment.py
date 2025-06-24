from typing import Dict
import mujoco
import os
import subprocess
import numpy as np
import mujoco.viewer
import time

from components.environment import BaseEnv, Observation

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

    def step(self, action):
        robot_pos = self.data.qpos[:3]
        target_pos = self.data.qpos[7:10]
        robot_vel = self.data.qvel[:3]
        target_vel = self.data.qvel[6:9]

        self.data.qvel[:3] = action

        mujoco.mj_step(self.model, self.data)

        self.last_state = self._get_state()

        for key in self.observations.keys():
            print(key, self.last_state[key])

        self.last_observation, noise = self.get_observation_from_state(self.last_state.copy())
        # add observation to history
        self.real_state_history[self.current_step] = self.last_state.copy()
        self.observation_history[self.current_step] = (self.last_observation.copy(), noise.copy())
        if self.current_step - 2 in self.observation_history:
            del self.observation_history[self.current_step - 2]
            del self.real_state_history[self.current_step - 2]

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

    def render(self):
        if self.viewer is None:
            # If the viewer is not initialized, create it
            self.viewer = mujoco.viewer.launch_passive(self.model, self.data)
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
        relative_pos = self.data.qpos[7:10] - self.data.qpos[:3]
        relative_vel = self.data.qvel[6:9] - self.data.qvel[:3]
        return np.dot(relative_pos, relative_vel) / np.linalg.norm(relative_pos) if np.linalg.norm(relative_pos) > 0 else 0.0
    def get_target_phi(self):
        """Get the angle between the robot and the target in the x-y plane."""
        relative_pos = self.data.qpos[7:10] - self.data.qpos[:3]
        return np.arctan2(relative_pos[1], relative_pos[0])
    def get_target_theta(self):
        relative_pos = self.data.qpos[7:10] - self.data.qpos[:3]
        relative_dist = np.linalg.norm(relative_pos)
        return np.pi/2 - np.arcsin(relative_pos[2] / relative_dist)
    def get_target_phi_dot(self):
        """Get the rate of change of the target's phi angle."""
        relative_pos = self.data.qpos[7:10] - self.data.qpos[:3]
        relative_vel = self.data.qvel[6:9] - self.data.qvel[:3]
        if np.linalg.norm(relative_pos) > 0:
            return (relative_vel[1] * relative_pos[0] - relative_vel[0] * relative_pos[1]) / np.linalg.norm(relative_pos)**2
        return 0.0
    def get_target_theta_dot(self):
        """Get the rate of change of the target's theta angle."""
        relative_pos = self.data.qpos[7:10] - self.data.qpos[:3]
        relative_vel = self.data.qvel[6:9] - self.data.qvel[:3]
        relative_dist = np.linalg.norm(relative_pos)
        if relative_dist > 0:
            return (relative_vel[2] * relative_dist - relative_pos[2] * np.dot(relative_vel, relative_pos) / relative_dist) / (relative_dist**2 * np.sqrt(1 - (relative_pos[2] / relative_dist)**2))
        return 0.0

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
