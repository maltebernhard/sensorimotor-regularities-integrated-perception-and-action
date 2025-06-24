import mujoco
import os
import subprocess
import numpy as np
import mujoco.viewer
import time

from components.environment import BaseEnv

# =================================================================================

class MujocoEnv(BaseEnv):
    def __init__(self):
        super().__init__()

        self.setup_gpu_rendering()

        # Load the XML model for the environment from a local file
        MODEL_XML_PATH = './two_spheres.xml'
        with open(MODEL_XML_PATH, 'r') as f:
            MODEL_XML = f.read()

        # Load the model
        self.model = mujoco.MjModel.from_xml_string(MODEL_XML)
        # Create the data object for the simulation
        self.data = mujoco.MjData(self.model)
        # Initialize the viewer
        self.viewer = None
        
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

    def step(self, partial_action):
        # TODO: apply the action to the mujoco sim
        mujoco.mj_step(self.model, self.data)

    def reset(self, seed=None, **kwargs):
        # Reset the mujoco simulation
        mujoco.mj_resetData(self.model, self.data)
        super().reset(seed=seed)
        np.random.seed(seed)

    def render(self):
        if self.viewer is None:
            # If the viewer is not initialized, create it
            self.viewer = mujoco.viewer.launch_passive(self.model, self.data)
        self.viewer.sync()

    def close(self):
        self.viewer.close() if self.viewer is not None else None
    
    # --------------------------------------------------

    def generate_observation_space(self):
        raise NotImplementedError

    def generate_action_space(self):
        raise NotImplementedError

# =================================================================================

if __name__ == "__main__":
    env = MujocoEnv()
    env.reset()
    for _ in range(100):
        env.step(np.random.rand(3))  # Random action
        env.render()
        # Get the simulation timestep and wait for this duration
        time_step = env.model.opt.timestep
        time.sleep(time_step)  # Sleep for the simulation timestep in seconds
    env.reset()
    for _ in range(100):
        env.step(np.random.rand(3))  # Random action
        env.render()
        # Get the simulation timestep and wait for this duration
        time_step = env.model.opt.timestep
        time.sleep(time_step)  # Sleep for the simulation timestep in seconds
    env.close()
    os._exit(0)  # Forcefully terminate the script to ensure no background process remains