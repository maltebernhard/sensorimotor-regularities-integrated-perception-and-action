from mujoco_model.aicon import MujocoAICON
from mujoco_model.environment import MujocoEnv
import os
import numpy as np
import time

# =================================================================================

if __name__ == "__main__":
    env = MujocoEnv('./mujoco_model/two_spheres.xml')
    env.reset()
    # try:
    #     for _ in range(10000):
    #         env.step(np.array([0.0, 0.0, 0.0]))  # Random action
    #         env.render()
    #         # Get the simulation timestep and wait for this duration
    #         time_step = env.model.opt.timestep
    #         #time.sleep(time_step)  # Sleep for the simulation timestep in seconds
    #         input(f"Press Enter to continue... (timestep: {time_step:.4f}s)")
    # except KeyboardInterrupt:
    #     print("Simulation interrupted by user.")
    

    aicon = MujocoAICON()
    aicon.run(1000, step_by_step=True, prints=1)


    env.close()
    os._exit(0)  # Forcefully terminate the script to ensure no background process remains