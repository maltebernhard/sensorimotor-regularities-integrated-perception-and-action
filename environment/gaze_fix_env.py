import os
from typing import Dict, List, Tuple
import gymnasium as gym
import numpy as np
import pygame
import vidmaker
import math
from environment.base_env import BaseEnv, Observation
import svgwrite
import cairosvg

# =====================================================================================================

os.environ['SDL_VIDEO_WINDOW_POS'] = "%d, %d" %(1000, 50)

# Colors
WHITE = (255, 255, 255)
BLUE = (0, 0, 255)
RED = (255, 0, 0)
GREEN = (0, 255, 0)
DARK_GREEN = (0, 180, 0)
LIGHT_RED = (255, 200, 200)
BLACK = (0, 0, 0)
GREY = (200, 200, 200)

COLORS = [(255, 0, 0), (0, 255, 0), (0, 0, 255), (255, 255, 0), (255, 0, 255), (0, 255, 255)]

SCREEN_SIZE = 900

# =====================================================================================================

class Robot:
    def __init__(self, pos, sensor_angle, max_vel, max_vel_rot, max_acc, max_acc_rot):
        self.start_pos: np.ndarray = pos
        self.pos: np.ndarray = pos
        self.orientation: float = 0.0
        self.vel: np.ndarray = np.array([0.0, 0.0], dtype=np.float64)
        self.vel_rot: float = 0.0
        self.sensor_angle: float = min(sensor_angle, 2*np.pi)
        self.size: float = 0.5
        self.max_vel: float = max_vel
        self.max_vel_rot: float = max_vel_rot
        self.max_acc: float = max_acc
        self.max_acc_rot: float = max_acc_rot

    def reset(self):
        self.pos: np.ndarray = self.start_pos.copy()
        self.orientation: float = 0.0
        self.vel: np.ndarray = np.array([0.0, 0.0], dtype=np.float64)
        self.vel_rot: float = 0.0

class EnvObject:
    def __init__(self, pos=np.zeros(2), config:tuple=("stationary", 0.0, 1.0), max_vel=0.0):
        self.pos = pos
        self.motion_config = config[0]
        self.set_base_movement_direction()
        self.vel = config[1] * max_vel
        self.radius = config[2]

    def set_base_movement_direction(self):
        self.base_movement_direction = np.random.uniform(-np.pi, np.pi)
        self.current_movement_direction = self.base_movement_direction

class Target(EnvObject):
    def __init__(self, pos=np.zeros(2), config:tuple=("stationary", 0.0, 1.0), max_vel=0.0, distance=0.0):
        super().__init__(pos, config, max_vel)
        self.distance = distance

# =====================================================================================================

class GazeFixEnv(BaseEnv):
    def __init__(self, config: dict):
        super().__init__()
        self.config = config

        self.episode_duration = config["episode_duration"]
        self.timestep: float = config["timestep"]
        self.current_step: int = 0
        self.time = 0.0
        self.total_reward = 0.0

        self.action_mode: int = config["action_mode"]
        self.action = np.array([0.0, 0.0, 0.0])

        self.max_target_distance:     float = config["target_distance"]
        self.desired_target_distance: float = config["target_distance"]
        self.spawn_distance:          float = config["start_distance"]
        self.reward_margin:           float = config["reward_margin"]
        self.penalty_margin:          float = config["penalty_margin"]
        self.wall_collision:          bool  = config["wall_collision"]
        self.num_obstacles:           int   = len(config["obstacles"])
        self.use_obstacles:           bool  = config["use_obstacles"]
        # target_config variations:
        # "stationary"  - target is stationary
        # "linear" - target moves linearly
        # "sine"   - target moves in a sine wave
        # "flight" - target moves in a flight pattern
        self.target_config: str = config["target_config"]
        self.obstacle_config: list[tuple] = config["obstacles"]
        self.observation_noise: Dict[str,Tuple[float,float]] = config["observation_noise"]
        self.observation_loss: Dict[str,List[Tuple[float,float]]] = config["observation_loss"]
        self.fv_noise: Dict[str,Tuple[float,float]] = config["fv_noise"]
        self.wind = np.array(config["wind"])

        self.robot = Robot(np.array([0.0, 0.0], dtype=np.float64), config["robot_sensor_angle"], config["robot_max_vel"], config["robot_max_vel_rot"], config["robot_max_acc"], config["robot_max_acc_rot"])
        self.target = None
        self.generate_target()
        self.obstacles: List[EnvObject] = None
        self.generate_obstacles()

        self.observation_history: Dict[int,Tuple[Dict[str,float],Dict[str,float]]] = {}
        self.real_state_history: Dict[int,Dict[str,float]] = {}

        self.generate_observation_space()
        self.generate_action_space()

        self.collision: bool = False

        # rendering window
        self.screen_size = SCREEN_SIZE
        self.viewer = None
        metadata = {'render_modes': ['human'], 'render_fps': 1/self.timestep}
        self.render_relative_to_robot = 3 # 1: fixed background, 2: fixed robot (incl. orientation), 3: fixed robot (excl. orientation)
        self.reward_render_mode = 1
        self.record_video = False
        self.video_path = ""
        self.video = None
        # NOTE: make true to save frames as svg
        self.render_svg = False
        if self.render_svg:
            self.render_relative_to_robot = 1
            self.screen_size = 5000
        # env dimensions
        self.world_size = config["world_size"]
        self.scale = self.screen_size / self.world_size

    def step(self, action):
        self.current_step += 1
        self.time += self.timestep
        if self.action_mode == 2:
            action = np.array([float(action[0]-1), float(action[1]-1), float(action[2]-1)])
        self.action = self.limit_action(action) # make sure acceleration / velocity vector is within bounds
        self.update_robot_velocity()
        self.update_target_movement_direction()
        self.update_obstacle_movement_direction()
        self.move_robot()
        self.move_target()
        self.move_obstacles()
        self.last_state, rewards, done, trun, info = self._get_state(), self.get_rewards(), self.get_terminated(), False, self.get_info()
        self.last_observation, noise = self.get_observation_from_state(self.last_state.copy())

        # add observation to history
        self.real_state_history[self.current_step] = self.last_state.copy()
        self.observation_history[self.current_step] = (self.last_observation.copy(), noise.copy())
        if self.current_step - 2 in self.observation_history:
            del self.observation_history[self.current_step - 2]
            del self.real_state_history[self.current_step - 2]

        rew = np.sum(rewards)
        self.total_reward += rew

        return np.array(list(self.last_observation.values())), rewards, done, trun, info

    def reset(self, seed=None, video_path = None, **kwargs):
        if seed is not None:
            super().reset(seed=seed)
        if self.video is not None:
            self.video.export(verbose=False)
            self.video = None

        if video_path is not None:
            self.record_video = True
            self.video_path = video_path
        else:
            self.record_video = False
        self.viewer = None
        self.time = 0.0
        self.current_step = 0
        self.total_reward = 0.0
        self.action = np.array([0.0, 0.0, 0.0])
        self.real_state_history = {}
        self.observation_history = {}
        self.robot.reset()
        self.collision = False
        self.generate_target()
        self.generate_obstacles()

        self.last_state, info = self._get_state(), self.get_info()
        self.last_observation, noise = self.get_observation_from_state(self.last_state.copy())
        self.real_state_history[self.current_step] = self.last_state.copy()
        self.observation_history[self.current_step] = (self.last_observation.copy(), noise.copy())

        return np.array(list(self.last_observation.values())), info

    def close(self):
        pygame.quit()
        self.screen = None

    def _get_state(self):
        """Computes a new observation."""
        return {key: obs.calculate_value() for key, obs in self.observations.items()}

    def get_observation(self):
        """Return the current, unnormalized observation."""
        try:
            obs = self.observation_history[self.current_step]
        except:
            raise Exception("I think this should never happen.")
            obs = self._get_state()
        return obs

    def get_state(self):
        """Return the current environment state."""
        try:
            obs = self.real_state_history[self.current_step].copy()
        except:
            raise Exception("I think this should never happen.")
            obs = self._get_state()
        return obs

    def get_rewards(self):
        time_left = 1.0 - self.time / self.episode_duration
        # reward for being close to target distance
        target_proximity_reward = 0.0
        dist = self.compute_distance(self.target)
        if abs(dist-self.target.distance) < self.reward_margin:
            target_proximity_reward = 1.0 / (abs(dist-self.target.distance) + 1.0) * self.timestep
        # time penalty
        time_penalty = - self.timestep / self.episode_duration
        # penalty for being close to obstacle
        obstacle_proximity_penalty = 0.0
        if self.use_obstacles:
            for obstacle in self.obstacles:
                dist = self.compute_distance(obstacle)
                if abs(dist-obstacle.radius) < self.penalty_margin:
                    obstacle_proximity_penalty -= 1.0 / (abs(dist-obstacle.radius) + 1.0) * self.timestep / self.num_obstacles / 10
        # penalize energy waste
        energy_waste_penalty = 0.0
        if self.action_mode == 1 or self.action_mode == 2:
            energy_waste_penalty -= (self.timestep / self.episode_duration / 2) * np.linalg.norm(self.action[:2]) / self.robot.max_acc
            energy_waste_penalty -= (self.timestep / self.episode_duration / 2) * abs(self.action[2]) / self.robot.max_acc_rot
        elif self.action_mode == 3:
            energy_waste_penalty -= (self.timestep / self.episode_duration / 2) * np.linalg.norm(self.action[:2]) / self.robot.max_vel
            energy_waste_penalty -= (self.timestep / self.episode_duration / 2) * abs(self.action[2]) / self.robot.max_vel_rot

        # penalize collision
        #collision_penalty = - time_left if self.collision else 0.0
        collision_penalty = - 1.0 if self.collision else 0.0

        return np.array([
            target_proximity_reward,
            time_penalty,
            obstacle_proximity_penalty,
            energy_waste_penalty,
            collision_penalty
        ])

    def get_terminated(self):
        return self.time > self.episode_duration or self.collision

    def get_info(self):
        return {}

    # ----------------------------------------- init functions -------------------------------------------------

    def generate_observation_space(self):
        self.observations: Dict[str, Observation] = {
            "vel_rot": Observation(-self.robot.max_vel_rot, self.robot.max_vel_rot, self.get_vel_rot),
            "vel_frontal": Observation(-self.robot.max_vel, self.robot.max_vel, self.get_vel_frontal),
            "vel_lateral": Observation(-self.robot.max_vel, self.robot.max_vel, self.get_vel_lateral),
            "desired_target_distance": Observation(0.0, self.config["target_distance"], self.get_desired_target_distance),
            "target_offset_angle": Observation(-self.robot.sensor_angle / 2, np.pi, self.create_object_state_function(self.compute_offset_angle, self.target)),
            "target_offset_angle_dot": Observation(-2 * np.pi / self.timestep / self.robot.max_vel_rot, 2 * np.pi / self.timestep / self.robot.max_vel_rot, self.create_object_state_function(self.compute_offset_angle_dot, self.target)),
            "target_offset_angle_dot_global": Observation(-np.inf, np.inf, self.create_object_state_function(self.compute_offset_angle_dot_object_component, self.target)),
            "target_distance": Observation(0.0, np.inf, self.create_object_state_function(self.compute_distance, self.target)),
            "target_distance_dot": Observation(-1.5 * self.robot.max_vel, 1.5 * self.robot.max_vel, self.create_object_state_function(self.compute_distance_dot, self.target)),
            "target_distance_dot_global": Observation(-self.robot.max_vel, self.robot.max_vel, self.create_object_state_function(self.compute_distance_dot_object_component, self.target)),
            "target_radius": Observation(0.0, np.inf, self.create_object_state_function(self.get_radius, self.target)),
            "target_visual_angle": Observation(0.0, self.robot.sensor_angle, self.create_object_state_function(self.compute_visual_angle, self.target)),
            "target_visual_angle_dot": Observation(-np.inf, np.inf, self.create_object_state_function(self.compute_visual_angle_dot, self.target)),
            "target_collision": Observation(0.0, 1.0, self.create_object_state_function(self.compute_collision, self.target)),
        }
        for o in range(self.num_obstacles):
            self.observations[f"obstacle{o+1}_offset_angle"] = Observation(-self.robot.sensor_angle / 2, np.pi, self.create_object_state_function(self.compute_offset_angle, self.obstacles[o]))
            self.observations[f"obstacle{o+1}_offset_angle_dot"] = Observation(-np.inf, np.inf, self.create_object_state_function(self.compute_offset_angle_dot, self.obstacles[o]))
            self.observations[f"obstacle{o+1}_offset_angle_dot_global"] = Observation(-np.inf, np.inf, self.create_object_state_function(self.compute_offset_angle_dot_object_component, self.obstacles[o]))
            self.observations[f"obstacle{o+1}_distance"] = Observation(-1.0, np.inf, self.create_object_state_function(self.compute_distance, self.obstacles[o]))
            self.observations[f"obstacle{o+1}_distance_dot"] = Observation(-1.0, 1.0, self.create_object_state_function(self.compute_distance_dot, self.obstacles[o]))
            self.observations[f"obstacle{o+1}_distance_dot_global"] = Observation(-self.robot.max_vel, self.robot.max_vel, self.create_object_state_function(self.compute_distance_dot_object_component, self.obstacles[o]))
            self.observations[f"obstacle{o+1}_radius"] = Observation(0.0, np.inf, self.create_object_state_function(self.get_radius, self.obstacles[o]))
            self.observations[f"obstacle{o+1}_visual_angle"] = Observation(0.0, self.robot.sensor_angle, self.create_object_state_function(self.compute_visual_angle, self.obstacles[o]))
            self.observations[f"obstacle{o+1}_visual_angle_dot"] = Observation(-np.inf, np.inf, self.create_object_state_function(self.compute_visual_angle_dot, self.obstacles[o]))
            self.observations[f"obstacle{o+1}_collision"] = Observation(0.0, 1.0, self.create_object_state_function(self.compute_collision, self.obstacles[o]))

        self.update_obs_noise_dict("target")
        for o in range(self.num_obstacles):
            self.update_obs_noise_dict(f"obstacle{o+1}")

        self.observation_indices = np.array([i for i in range(len(self.observations))])
        self.last_state = None

        self.required_observations = [key for key in self.observations.keys()]

        self.observation_space = gym.spaces.Box(
            low=np.array([obs.low for obs in self.observations.values()]),
            high=np.array([obs.high for obs in self.observations.values()]),
            shape=(len(self.observations),),
            dtype=np.float64
        )

    def update_obs_noise_dict(self, obj: str):
        for key in ["offset_angle", "offset_angle_dot", "visual_angle", "visual_angle_dot", "distance", "distance_dot"]:
            if key in self.observation_noise:
                self.observation_noise.update({obj+"_"+key: self.observation_noise[key]})
            if key in self.fv_noise:
                self.fv_noise.update({obj+"_"+key: self.fv_noise[key]})
            if key in self.observation_loss:
                self.observation_loss.update({obj+"_"+key: self.observation_loss[key]})

    def get_desired_target_distance(self):
        return self.target.distance

    def get_vel_rot(self):
        return self.robot.vel_rot

    def get_vel_frontal(self):
        wind_robot_frame = self.rotation_matrix(-self.robot.orientation) @ (self.wind * self.robot.max_vel) if self.action_mode == 3 else np.zeros(2)
        return self.robot.vel[0] + wind_robot_frame[0]

    def get_vel_lateral(self):
        wind_robot_frame = self.rotation_matrix(-self.robot.orientation) @ (self.wind * self.robot.max_vel) if self.action_mode == 3 else np.zeros(2)
        return self.robot.vel[1] + wind_robot_frame[1]

    def create_object_state_function(self, state_function, obj):
        def func():
            return state_function(obj)
        return func

    def get_radius(self, obj):
        return obj.radius

    def compute_collision(self, obj):
        return 1.0 if self.compute_distance(obj) < obj.radius + self.robot.size / 2 else 0.0

    def generate_action_space(self):
        if self.action_mode == 1 or self.action_mode == 3:
            self.action_space = gym.spaces.Box(
                low=np.array([-1.0]*3),
                high=np.array([1.0]*3),
                shape=(3,),
                dtype=np.float64
            )
        elif self.action_mode == 2:
            self.action_space = gym.spaces.MultiDiscrete(
                np.array([3, 3, 3])
            )
        else: raise NotImplementedError

    def generate_target(self):
        #distance = np.random.uniform(self.world_size / 4, self.world_size / 2)
        #angle = np.random.uniform(-np.pi/2, np.pi/2)
        angle = np.random.uniform(-np.pi, np.pi)
        # x = distance * np.cos(angle)
        # y = distance * np.sin(angle)
        #desired_distance = np.random.uniform(5*radius, self.max_target_distance)
        desired_distance = self.desired_target_distance
        x = self.robot.pos[0] + (self.spawn_distance + desired_distance) * np.cos(angle)
        y = self.robot.pos[1] + (self.spawn_distance + desired_distance) * np.sin(angle)
        if self.target is None:
            self.target = Target(
                pos = np.array([x,y]),
                config = self.target_config,
                max_vel = self.robot.max_vel,
                distance = desired_distance
            )
        else:
            self.target.pos = np.array([x,y])
            self.target.set_base_movement_direction()

    def generate_obstacles(self):
        target_distance = np.linalg.norm(self.target.pos)
        std_dev = target_distance / 8
        midpoint = (self.target.pos + self.robot.pos) / 2
        if self.obstacles is None:
            self.obstacles = []
        obst_index = 0
        for config in self.obstacle_config:
            while True:
                # TODO: adapt obstacle generation according to desired experiment
                #pos = np.random.normal(loc=midpoint, scale=std_dev, size=2)
                distance = np.random.uniform(self.world_size / 4, self.world_size / 2)
                angle = np.random.uniform(-np.pi, np.pi)
                pos = (distance * np.cos(angle), distance * np.sin(angle))

                # Ensure the obstacle doesn't spawn too close to robot
                if np.linalg.norm(pos-self.robot.pos) > config[2] + 5 * self.robot.size:
                    if len(self.obstacles) == obst_index:
                        self.obstacles.append(EnvObject(
                            pos = pos,
                            config = config,
                            max_vel = self.robot.max_vel,
                        ))
                    else:
                        self.obstacles[obst_index].pos = pos
                        self.obstacles[obst_index].set_base_movement_direction()
                    obst_index += 1
                    break

    # -------------------------------------- helpers -------------------------------------------

    def get_observation_field(self, n = 20):
        observation_field = [[None]*n for _ in range(n)]
        x_positions = np.linspace(-self.world_size/4, 3*self.world_size/4, n)
        y_positions = np.linspace(-self.world_size/2, self.world_size/2, n)
        for i, x in enumerate(x_positions):
            for j, y in enumerate(y_positions):
                self.set_robot_position(np.array([x,y]), np.arctan2(self.target.pos[1]-self.robot.pos[1], self.target.pos[0]-self.robot.pos[0]))
                observation_field[i][j] = np.array(list(self._get_state().values()))
        return observation_field

    def set_robot_position(self, pos, orientation):
        self.robot.pos = pos
        self.robot.orientation = orientation

    def limit_action(self, action):
        translation, rotation = action[:2], action[2:]
        translation_abs = np.linalg.norm(translation)
        if translation_abs > 1:
            translation = translation / translation_abs
        rotation_abs = abs(rotation[0])
        if rotation_abs > 1:
            rotation = rotation / rotation_abs
        return np.concatenate([translation, rotation])

    def update_robot_velocity(self):
        # set xy and angular accelerations
        if self.action_mode == 1 or self.action_mode == 2:
            acc = (self.action[:2] + self.wind) * self.robot.max_acc
            acc_rot = self.action[2] * self.robot.max_acc_rot
            self.robot.vel += acc * self.timestep                   # update robot velocity vector
            self.robot.vel_rot += acc_rot * self.timestep           # update rotational velocity
            self.limit_robot_velocity()
        elif self.action_mode == 3:
            self.robot.vel = self.action[:2] * self.robot.max_vel
            self.robot.vel_rot = self.action[2] * self.robot.max_vel_rot
            return

    def update_target_movement_direction(self):
        if self.target.motion_config in ["stationary", "linear"]:
            pass
        elif self.target.motion_config == "sine":
            self.target.current_movement_direction = self.target.base_movement_direction + np.pi/3 * np.sin(self.time/4)
        elif self.target.motion_config == "flight":
            self.target.current_movement_direction = np.atan2(self.target.pos[1]-self.robot.pos[1], self.target.pos[0]-self.robot.pos[0])
        elif self.target.motion_config == "chase":
            self.target.current_movement_direction = np.atan2(self.target.pos[1]-self.robot.pos[1], self.target.pos[0]-self.robot.pos[0]) + np.pi
        else:
            raise ValueError(f"{self.target.motion_config} is not a valid moving target config.")

    def update_obstacle_movement_direction(self):
        for o in self.obstacles:
            if o.motion_config == "stationary":
                pass
            elif o.motion_config == "chase":
                o.current_movement_direction = np.atan2(o.pos[1]-self.robot.pos[1], o.pos[0]-self.robot.pos[0]) + np.pi
            else:
                raise ValueError(f"{o.motion_config} is not a valid moving obstacles config.")

    def move_robot(self):
        # move robot
        total_vel = self.rotation_matrix(self.robot.orientation) @ self.robot.vel
        if self.action_mode == 3:
            total_vel += self.wind * self.robot.max_vel
        self.robot.pos += total_vel * self.timestep
        self.robot.orientation += self.robot.vel_rot * self.timestep
        self.robot.vel = self.rotation_matrix(-self.robot.vel_rot * self.timestep) @ self.robot.vel
        # handle orientation overflow to range [-pi, pi]
        if self.robot.orientation < -np.pi: self.robot.orientation = self.robot.orientation + 2*np.pi
        elif self.robot.orientation > np.pi: self.robot.orientation = self.robot.orientation - 2*np.pi
        self.check_collision()

    def limit_robot_velocity(self):
        vel = np.linalg.norm(self.robot.vel)                                            # compute absolute translational velocity
        if vel > self.robot.max_vel:                                                    # make sure translational velocity is within bounds
            self.robot.vel = self.robot.vel / vel * self.robot.max_vel
        if abs(self.robot.vel_rot) > self.robot.max_vel_rot:                            # make sure rotational velocity is within bounds
            self.robot.vel_rot = self.robot.vel_rot/abs(self.robot.vel_rot) * self.robot.max_vel_rot

    def move_target(self):
        if self.target.motion_config != "stationary":
            self.target.pos += np.array([np.cos(self.target.current_movement_direction), np.sin(self.target.current_movement_direction)]) * self.target.vel * self.timestep

    def move_obstacles(self):
        for o in self.obstacles:
            if o.motion_config != "stationary":
                o.pos += np.array([np.cos(o.current_movement_direction), np.sin(o.current_movement_direction)]) * o.vel * self.timestep

    def check_collision(self):
        if self.compute_distance(self.target) < self.target.radius + self.robot.size / 2:
            self.collision = True
            return
        if self.use_obstacles:
            for o in self.obstacles:
                if self.compute_distance(o) < o.radius + self.robot.size / 2:
                    self.collision = True
                    return
        if self.wall_collision:
            if self.robot.pos[0] < -self.world_size / 2 + self.robot.size/2:
                self.robot.pos[0] = -self.world_size / 2 + self.robot.size/2
                self.collision = True
            elif self.robot.pos[0] > self.world_size / 2 - self.robot.size/2:
                self.robot.pos[0] = self.world_size / 2 - self.robot.size/2
                self.collision = True
            if self.robot.pos[1] < -self.world_size / 2 + self.robot.size/2:
                self.robot.pos[1] = -self.world_size / 2 + self.robot.size/2
                self.collision = True
            elif self.robot.pos[1] > self.world_size / 2 - self.robot.size/2:
                self.robot.pos[1] = self.world_size / 2 - self.robot.size/2
                self.collision = True

    def observe_obstacles(self):
        if self.use_obstacles:
            obstacles = []
            for obstacle in self.obstacles:
                obs = []
                # angle
                angle = self.normalize_angle(np.arctan2(obstacle.pos[1]-self.robot.pos[1], obstacle.pos[0]-self.robot.pos[0]) - self.robot.orientation)
                if not (angle>-self.robot.sensor_angle/2 and angle<self.robot.sensor_angle/2):
                    obstacles += [np.pi, 0.0, -1.0]
                    continue
                obs.append(angle)
                # coverage
                obs.append(self.compute_visual_angle(obstacle))
                # distance
                obs.append(self.compute_distance(obstacle)-obstacle.radius)
                obstacles += obs
        else:
            obs = [np.pi, 0.0, -1.0]
            obstacles = self.num_obstacles * obs
        return np.array(obstacles)

    def normalize_angle(self, angle):
        """Normalize an angle to the range [-pi, pi]."""
        return np.arctan2(np.sin(angle), np.cos(angle))

    def rotation_matrix(self, angle):
        return np.array(
            [[np.cos(angle), -np.sin(angle)],
             [np.sin(angle), np.cos(angle)]]
        )

    # -------------------------------------- observation functions -------------------------------------------

    # TODO: restructure for-loops to work object-wise
    def isolate_sensor_readings_from_observations(self, env_state: Dict[str, float]) -> Dict[str, float]:
        observation = env_state.copy()
        # delete all observations not provided to any measurement model
        keys_to_delete = list(set([key for key in observation if key not in self.required_observations] + [key for key in observation if any([key[-len(sensor_key):]==sensor_key for sensor_key in [skey for skey, step_ranges in self.observation_loss.items() if any([self.current_step >= step_range[0] and self.current_step < step_range[1] for step_range in step_ranges])]])]))
        for key in keys_to_delete:
            del observation[key]
        # delete all occluded observations
        for key, value in observation.items():
            if self.robot.sensor_angle < 2*np.pi and key[-12:] == "offset_angle":
                if "target" in key: obj = self.target
                else: obj = self.obstacles[int(key[8])-1]
                if abs(observation[key]) > self.robot.sensor_angle / 2 + env_state[f"{key[:-12]}visual_angle"] / 2:   # if target/obstacle is outside of camera angle, remove observations:
                    observation[key] = None                                                    # angle
                    if key+"_dot" in observation.keys():
                        observation[key+"_dot"] = None                                         # angle dot
                    if key[:-12] + "visual_angle" in observation.keys():
                        observation[key[:-12] + "visual_angle"] = None                         # visual angle
                    if key[:-12] + "visual_angle_dot" in observation.keys():
                        observation[key[:-12]+"visual_angle_dot"] = None                       # visual angle dot
        return observation

    def apply_sensor_noise(self, observation: Dict[str, float]) -> Dict[str, float]:
        observation_noise_means = {}
        observation_noise_stddevs = {}
        observation_noise_factor = {}
        for key, value in observation.items():
            if observation[key] is not None:
                if key in self.observation_noise.keys():
                    observation_noise_means[key] = self.observation_noise[key][0]
                    observation_noise_stddevs[key] = self.observation_noise[key][1]
                    if key == "vel_frontal":
                        observation_noise_factor[key] = np.abs(self.robot.vel[0])
                    elif key == "vel_lateral":
                        observation_noise_factor[key] = np.abs(self.robot.vel[1])
                    elif key == "vel_rot":
                        observation_noise_factor[key] = np.abs(self.robot.vel_rot)
                    elif "distance_dot" in key:
                        observation_noise_factor[key] = np.abs(observation[key[:-4]]/10)
                    elif "distance" in key:
                        observation_noise_factor[key] = np.abs(observation[key])
                    elif "visual_angle" in key:
                        observation_noise_factor[key] = np.abs(observation[key])
                    elif "offset_angle_dot" in key:
                        observation_noise_factor[key] = np.abs(observation[key])
                    else:
                        observation_noise_factor[key] = 1.0
                if key in self.fv_noise.keys():
                    if "target" in key: obj = self.target
                    elif "obstacle1" in key: obj = self.obstacles[0]
                    elif "obstacle2" in key: obj = self.obstacles[1]
                    else:
                        raise NotImplementedError
                    observation_noise_means[key] += self.fv_noise[key][0] * np.abs(self.compute_offset_angle(obj) / (self.robot.sensor_angle/2))
                    observation_noise_stddevs[key] += self.fv_noise[key][1] * np.abs(self.compute_offset_angle(obj) / (self.robot.sensor_angle/2))
                observation[key] = value + (observation_noise_means[key]*((value/abs(value) if value!=0 else 1.0)) + observation_noise_stddevs[key]*np.random.randn()) * observation_noise_factor[key]
        return observation, {key: (observation_noise_means[key]*observation_noise_factor[key], observation_noise_stddevs[key]*observation_noise_factor[key]) if key in observation_noise_stddevs.keys() else (0.0,0.0) for key in observation.keys()}

    def get_observation_from_state(self, env_state: Dict[str, float]) -> Dict[str, float]:
        """Applies gaussion noise and occlusions to real state."""
        observation = self.isolate_sensor_readings_from_observations(env_state)
        observation, observation_noise = self.apply_sensor_noise(observation)
        return observation, observation_noise

    def compute_offset_angle(self, obj: EnvObject) -> float:
        return self.normalize_angle(np.arctan2(obj.pos[1]-self.robot.pos[1], obj.pos[0]-self.robot.pos[0]) - self.robot.orientation)

    def compute_offset_angle_dot(self, obj: EnvObject):
        offset_angle = self.compute_offset_angle(obj)
        # angular vel due to robot's rotation
        offset_angle_dot = - self.robot.vel_rot
        # angular vel due to robot's translation
        offset_angle_dot -= (self.rotation_matrix(-offset_angle) @ self.robot.vel)[1] / self.compute_distance(obj)
        # angular vel due to object's movement
        offset_angle_dot += self.compute_offset_angle_dot_object_component(obj)
        return offset_angle_dot

    def compute_offset_angle_dot_object_component(self, obj: EnvObject):
        if obj.motion_config != "stationary":
            offset_angle = self.compute_offset_angle(obj)
            return obj.vel * np.sin(obj.current_movement_direction - offset_angle - self.robot.orientation) / self.compute_distance(obj)
        else:
            return 0.0

    def compute_distance(self, obj: EnvObject):
        return np.linalg.norm(obj.pos-self.robot.pos)

    def compute_distance_dot(self, obj: EnvObject):
        offset_angle = self.compute_offset_angle(obj)
        object_distance_dot = - (self.rotation_matrix(-offset_angle) @ self.robot.vel)[0]
        object_distance_dot += self.compute_distance_dot_object_component(obj)
        return object_distance_dot

    def compute_distance_dot_object_component(self, obj: EnvObject):
        if obj.motion_config != "stationary":
            offset_angle = self.compute_offset_angle(obj)
            return obj.vel * np.cos(obj.current_movement_direction - offset_angle - self.robot.orientation)
        else:
            return 0.0

    def compute_visual_angle(self, obj: EnvObject):
        distance = self.compute_distance(obj)
        # angular size with respect to distance
        if obj.radius / distance >= 1.0:
            return None
        else:
            return 2 * math.asin(obj.radius / distance)

    def compute_visual_angle_dot(self, obj: EnvObject):
        distance = self.compute_distance(obj)
        distance_dot = self.compute_distance_dot(obj)
        # Derivative of the angular size with respect to distance
        if obj.radius / distance >= 1.0:
            return None
        else:
            distance_dot = self.compute_distance_dot(obj)
            return -2 * obj.radius * distance_dot / (distance**2 * math.sqrt(1 - (obj.radius / distance)**2))

    # ----------------------------------- render stuff -----------------------------------------

    def render(self, real_time_factor = 1.0, robot_frame_means: Dict[str, np.ndarray] = None, robot_frame_covs: Dict[str, np.ndarray] = None):
        if self.viewer is None:
            pygame.init()
            self.svg_exporter = SVGExporter(self.screen_size, self.screen_size) if self.render_svg else None
            # Clock to control frame rate
            self.rt_clock = pygame.time.Clock()
            # set window
            self.viewer = pygame.display.set_mode((self.screen_size, self.screen_size))
            pygame.display.set_caption("Gaze Fixation")
            if self.record_video:
                self.video = vidmaker.Video(self.video_path, fps=int(1/self.timestep), resolution=(self.screen_size,self.screen_size), late_export=True)
        if self.svg_exporter is not None: self.svg_exporter.create_new_drawing()  # start a fresh SVG for each render call
        self.draw_env()
        if abs(self.action[0]) > 0.0 or abs(self.action[1]) > 0.0:
            # draw an arrow for the robot's action
            self.draw_arrow(self.robot.pos, self.robot.orientation+math.atan2(self.action[1],self.action[0]), self.robot.size*10*(np.linalg.norm(self.action[:2])), self.robot.size*2, RED if self.action_mode in [1,2] else BLUE)
            if self.action_mode in [1,2]:
                # draw an arrow for the robot's velocity
                self.draw_arrow(self.robot.pos, self.robot.orientation+math.atan2(self.robot.vel[1],self.robot.vel[0]), self.robot.size*10*(np.linalg.norm(self.robot.vel)/self.robot.max_vel), self.robot.size*1.5, BLUE)
        if self.target.motion_config != "stationary":
            # draw an arrow for the target's movement
            self.draw_arrow(self.target.pos, self.target.current_movement_direction, self.robot.size*10*self.target.vel/self.robot.max_vel, self.robot.size*1.5, DARK_GREEN)

        if robot_frame_means is not None and robot_frame_covs is not None:
            assert len(robot_frame_means) == len(robot_frame_covs), "Number of means and covariances must be equal"
            assert len(robot_frame_means) <= 6, "Only up to 6 estimator states can be visualized"
            for i, key in enumerate(robot_frame_means.keys()):
                mean = self.rotation_matrix(self.robot.orientation) @ robot_frame_means[key][:2] + self.robot.pos
                self.draw_gaussian(mean, robot_frame_covs[key][:2,:2], COLORS[i])
                pygame.draw.circle(self.viewer, COLORS[i], self.pxl_coordinates(mean), int(self.screen_size/150))
                if self.svg_exporter is not None: self.svg_exporter.draw_circle(COLORS[i],self.pxl_coordinates(mean),int(self.screen_size/150))
        #self.display_info()
        if self.record_video:
            self.video.update(pygame.surfarray.pixels3d(self.viewer).swapaxes(0, 1), inverted=False)
        pygame.display.flip()
        # NOTE: use this to save individual pictures of the env
        if self.render_relative_to_robot == 1:
            if self.current_step <= 30:
                if "counter" not in self.__dict__:
                    self.counter = 0
                else:
                    self.counter += 1
                #pygame.image.save(self.viewer , f"{self.counter}.jpg")
                if self.svg_exporter is not None: self.svg_exporter.export()  # export the SVG to file
            if self.current_step == 31:
                raise Exception("Printed Images. Set environment render mode to 3 for normal testing.")
        self.rt_clock.tick(1/self.timestep*real_time_factor)

    def draw_gaussian(self, world_frame_mean, robot_frame_cov, color):
        eigvals, eigvecs = np.linalg.eig(robot_frame_cov)
        eigvals = np.real(eigvals)
        eigvecs = np.real(eigvecs)
        if self.render_relative_to_robot == 2:
            angle = - np.arctan2(eigvecs[1,0], eigvecs[0,0]) + np.pi/2
        elif self.render_relative_to_robot == 3:
            angle = - np.arctan2(eigvecs[1,0], eigvecs[0,0]) + np.pi/2 - self.robot.orientation
        elif self.render_relative_to_robot == 1:
            angle = - np.arctan2(eigvecs[1,0], eigvecs[0,0]) + np.pi/2 - self.robot.orientation
        width, height = 2*np.sqrt(eigvals)
        width = max(0.1, width)
        height = max(0.1, height)
        ellipse_surface = pygame.Surface((width*self.scale, height*self.scale), pygame.SRCALPHA)
        pygame.draw.ellipse(ellipse_surface, (*color, 128), ellipse_surface.get_rect())
        if self.svg_exporter is not None: self.svg_exporter.draw_ellipse(color,center=self.pxl_coordinates(world_frame_mean),width=width*self.scale,height=height*self.scale,angle=np.degrees(angle),alpha=128/255)
        rotated_ellipse = pygame.transform.rotate(ellipse_surface, -np.degrees(angle))
        ellipse_rect = rotated_ellipse.get_rect(center=self.pxl_coordinates(world_frame_mean))
        self.viewer.blit(rotated_ellipse, ellipse_rect)

    def draw_env(self, reward_render_mode = 1):
        if reward_render_mode == 1:
            self.viewer.fill(GREY)
            if self.svg_exporter is not None: self.svg_exporter.draw_rect(GREY, (0, 0), (self.screen_size, self.screen_size))
            self.draw_fov()
        else:
            color_map = self.get_reward_color_map()
            pygame.surfarray.blit_array(self.viewer, color_map)
            # For a direct pixel-based map, we skip detailed svg for color_map

        self.draw_grid()
        if reward_render_mode == 1 and self.reward_margin < 20.0:
            # draw target reward margin
            self.transparent_circle(self.target.pos, self.target.distance+self.reward_margin, GREEN)
        # draw target distance
        pygame.draw.circle(self.viewer, DARK_GREEN, self.pxl_coordinates((self.target.pos[0],self.target.pos[1])), self.target.distance*self.scale, width=int(self.screen_size/500))
        # draw target
        pygame.draw.circle(self.viewer, DARK_GREEN, self.pxl_coordinates((self.target.pos[0],self.target.pos[1])), self.target.radius*self.scale)
        # draw vision axis
        pygame.draw.line(self.viewer, BLACK, self.pxl_coordinates((self.robot.pos[0],self.robot.pos[1])), self.pxl_coordinates(self.polar_point(self.robot.orientation,self.world_size*3)), int(self.screen_size/1000))
        # draw Agent
        pygame.draw.circle(self.viewer, BLUE, self.pxl_coordinates((self.robot.pos[0],self.robot.pos[1])), self.robot.size*self.scale)
        pygame.draw.polygon(self.viewer, BLUE, [self.pxl_coordinates(self.polar_point(self.robot.orientation+np.pi/2, self.robot.size/1.25)), self.pxl_coordinates(self.polar_point(self.robot.orientation-np.pi/2, self.robot.size/1.25)), self.pxl_coordinates(self.polar_point(self.robot.orientation, self.robot.size*2.0))])
        # draw obstacles
        if self.use_obstacles:
            for o in self.obstacles:
                pygame.draw.circle(self.viewer, BLACK, self.pxl_coordinates((o.pos[0],o.pos[1])), o.radius*self.scale)
                if reward_render_mode == 1 and self.penalty_margin < 20.0:
                    self.transparent_circle(o.pos, o.radius+self.penalty_margin, RED)

        if self.svg_exporter is not None:
            self.svg_exporter.draw_circle(DARK_GREEN,
                                        self.pxl_coordinates((self.target.pos[0], self.target.pos[1])),
                                        self.target.distance*self.scale,
                                        stroke=True,
                                        stroke_width=int(self.screen_size/500),
                                        fill_opacity=0)
            self.svg_exporter.draw_circle(DARK_GREEN,
                                        self.pxl_coordinates((self.target.pos[0], self.target.pos[1])),
                                        self.target.radius*self.scale)
            self.svg_exporter.draw_line(BLACK,
                                        self.pxl_coordinates((self.robot.pos[0], self.robot.pos[1])),
                                        self.pxl_coordinates(self.polar_point(self.robot.orientation,self.world_size*3)),
                                        width=int(self.screen_size/1000))
            self.svg_exporter.draw_circle(BLUE,
                                        self.pxl_coordinates((self.robot.pos[0], self.robot.pos[1])),
                                        self.robot.size*self.scale)
            self.svg_exporter.draw_polygon(BLUE,
                [self.pxl_coordinates(self.polar_point(self.robot.orientation+np.pi/2, self.robot.size/1.25)),
                self.pxl_coordinates(self.polar_point(self.robot.orientation-np.pi/2, self.robot.size/1.25)),
                self.pxl_coordinates(self.polar_point(self.robot.orientation, self.robot.size*2.0))])
            if self.use_obstacles:
                for o in self.obstacles:
                    self.svg_exporter.draw_circle(BLACK, self.pxl_coordinates((o.pos[0],o.pos[1])), o.radius*self.scale)

    def draw_action_field(self, action_field, savepath=None):
        if self.viewer is None:
            self.viewer = pygame.display.set_mode((self.screen_size, self.screen_size))
        self.set_robot_position(np.array([self.world_size/4,0.0]), 0.0)
        self.draw_env(reward_render_mode = 2)
        x_positions = np.linspace(-self.world_size/4, 3*self.world_size/4, len(action_field))
        y_positions = np.linspace(-self.world_size / 2, self.world_size / 2, len(action_field[0]))
        for i in range(len(action_field)):
            for j in range(len(action_field[0])):
                action = action_field[i][j]
                self.draw_arrow((x_positions[i],y_positions[j]), np.arctan2(self.target.pos[1]-y_positions[j], self.target.pos[0]-x_positions[i]) + np.arctan2(action[1],action[0]), self.robot.size*5*np.linalg.norm(action[:2]), 1*self.robot.size, BLACK)
        if savepath is not None:
            pygame.image.save(self.viewer, savepath + "/action_vector_field.png")
        else:
            pygame.display.flip()

    def draw_arrow(self, pos, angle, length, side_length, color):
        end_point = self.polar_point(angle, length, pos)
        tip_point_1 = self.polar_point(self.normalize_angle(angle+3*np.pi/4), side_length, end_point)
        tip_point_2 = self.polar_point(self.normalize_angle(angle-3*np.pi/4), side_length, end_point)
        pygame.draw.line(self.viewer, color, self.pxl_coordinates(pos), self.pxl_coordinates(end_point), width=int(self.screen_size/500))
        pygame.draw.line(self.viewer, color, self.pxl_coordinates(end_point), self.pxl_coordinates(tip_point_1), width=int(self.screen_size/500))
        pygame.draw.line(self.viewer, color, self.pxl_coordinates(end_point), self.pxl_coordinates(tip_point_2), width=int(self.screen_size/500))
        if self.svg_exporter is not None:
            self.svg_exporter.draw_line(color,
                                        self.pxl_coordinates(pos),
                                        self.pxl_coordinates(end_point),
                                        width=int(self.screen_size/500))
            self.svg_exporter.draw_line(color,
                                        self.pxl_coordinates(end_point),
                                        self.pxl_coordinates(tip_point_1),
                                        width=int(self.screen_size/500))
            self.svg_exporter.draw_line(color,
                                        self.pxl_coordinates(end_point),
                                        self.pxl_coordinates(tip_point_2),
                                        width=int(self.screen_size/500))

    def transparent_circle(self, pos, radius, color):
        target_rect = pygame.Rect(self.pxl_coordinates(pos), (0, 0)).inflate((radius*2*self.scale, radius*2*self.scale))
        shape_surf = pygame.Surface(target_rect.size, pygame.SRCALPHA)
        pygame.draw.circle(shape_surf, (color[0],color[1],color[2],50), (radius*self.scale, radius*self.scale), radius*self.scale)
        self.viewer.blit(shape_surf, target_rect)

    def draw_fov(self):
        if self.robot.sensor_angle < np.pi:
            robot_point = self.pxl_coordinates(self.robot.pos)
            left_angle = self.pxl_coordinates(self.polar_point(self.robot.orientation+self.robot.sensor_angle/2, self.world_size*3))
            right_angle = self.pxl_coordinates(self.polar_point(self.robot.orientation-self.robot.sensor_angle/2, self.world_size*3))
            left_corner = self.pxl_coordinates(self.polar_point(self.robot.orientation+np.pi/4, self.world_size))
            right_corner = self.pxl_coordinates(self.polar_point(self.robot.orientation-np.pi/4, self.world_size))
            pygame.draw.polygon(self.viewer, WHITE, [robot_point, left_angle, left_corner, right_corner, right_angle])
            if self.svg_exporter is not None: self.svg_exporter.draw_polygon(WHITE, [robot_point, left_angle, left_corner, right_corner, right_angle])
        elif abs(self.robot.sensor_angle - 2*np.pi) < 0.01:
            self.viewer.fill(WHITE)
            if self.svg_exporter is not None: self.svg_exporter.draw_rect(WHITE, (0, 0), (self.screen_size, self.screen_size))
        else:
            self.viewer.fill(WHITE)
            if self.svg_exporter is not None: self.svg_exporter.draw_rect(WHITE, (0, 0), (self.screen_size, self.screen_size))
            robot_point = self.pxl_coordinates(self.robot.pos)
            left_angle = self.pxl_coordinates(self.polar_point(self.robot.orientation+self.robot.sensor_angle/2, self.world_size*3))
            right_angle = self.pxl_coordinates(self.polar_point(self.robot.orientation-self.robot.sensor_angle/2, self.world_size*3))
            left_corner = self.pxl_coordinates(self.polar_point(self.robot.orientation+3*np.pi/4, self.world_size))
            right_corner = self.pxl_coordinates(self.polar_point(self.robot.orientation-3*np.pi/4, self.world_size))
            pygame.draw.polygon(self.viewer, GREY, [robot_point, left_angle, left_corner, right_corner, right_angle])
            if self.svg_exporter is not None: self.svg_exporter.draw_polygon(GREY, [robot_point, left_angle, left_corner, right_corner, right_angle])

    def polar_point(self, angle, distance, start_pos = None):
        if start_pos is None:
            start_pos = self.robot.pos
        return start_pos[0] + distance * math.cos(angle), start_pos[1] + distance * math.sin(angle)

    def display_info(self):
        font = pygame.font.Font(None, 24)
        legend = [
            (f'Step:', f'{self.current_step}'),                                 # clock
            (f'Time:', f'{self.time:.2f}'),                                     # time
            # (f'Step reward:', f'{np.sum(self.get_rewards()):.4f}'),           # step reward
            # (f'Total reward:', f'{self.total_reward:.4f}'),                   # episode reward
            (f'Target Distance:', f'{self.compute_distance(self.target):.2f}'), # target distance
            (f'Desired Distance:', f'{self.target.distance:.2f}'),              # desired target distance
        ]
        for i, (text, value) in enumerate(legend):
            self.viewer.blit(font.render(text, True, BLACK), (10, 5+25*i))
            self.viewer.blit(font.render(value, True, BLACK), (160, 5+25*i))
        self.viewer.blit(font.render("Robot", True, BLACK), (self.pxl_coordinates(self.robot.pos)) + np.array([10,-20]))
        self.viewer.blit(font.render("Target", True, BLACK), (self.pxl_coordinates(self.target.pos)) + np.array([10,-20]))

    def draw_grid(self):
        thick_lines = []
        thin_lines = []
        for x in range(int(self.robot.pos[0]-self.world_size/2), int(self.robot.pos[0]+self.world_size/2+1)):
            if x % 10 == 0:
                thick_lines.append(((float(x),self.robot.pos[1]+self.world_size), (float(x),self.robot.pos[1]-self.world_size)))
            elif x % 2 == 0:
                thin_lines.append(((float(x),self.robot.pos[1]+self.world_size), (float(x),self.robot.pos[1]-self.world_size)))
        for y in range(int(self.robot.pos[1]-self.world_size/2), int(self.robot.pos[1]+self.world_size/2+1)):
            if y % 10 == 0:
                thick_lines.append(((self.robot.pos[0]+self.world_size,float(y)), (self.robot.pos[0]-self.world_size,float(y))))
            elif y % 2 == 0:
                thin_lines.append(((self.robot.pos[0]+self.world_size,float(y)), (self.robot.pos[0]-self.world_size,float(y))))
        for line in thick_lines:
            pygame.draw.line(self.viewer, BLACK, self.pxl_coordinates(line[0]), self.pxl_coordinates(line[1]), width=int(self.screen_size/800))
            # if self.svg_exporter is not None: self.svg_exporter.draw_line(BLACK, self.pxl_coordinates(line[0]), self.pxl_coordinates(line[1]), width=int(self.screen_size/800))
        for line in thin_lines:
            pygame.draw.line(self.viewer, BLACK, self.pxl_coordinates(line[0]), self.pxl_coordinates(line[1]), width=int(self.screen_size/1600))
            # if self.svg_exporter is not None: self.svg_exporter.draw_line(BLACK, self.pxl_coordinates(line[0]), self.pxl_coordinates(line[1]), width=int(self.screen_size/1600))

    def pxl_coordinates(self, xy):
        if self.render_relative_to_robot == 1:
            x = int(self.screen_size/2 + xy[0] * self.scale)
            y = int(self.screen_size/2 - xy[1] * self.scale)
            xy_a = self.rotation_matrix(-np.pi/2) @ np.array([x-self.screen_size/2,y-self.screen_size/2])
            x_pxl = int(xy_a[0] + self.screen_size/2)
            y_pxl = int(xy_a[1] + self.screen_size/2) 
        elif self.render_relative_to_robot == 2:
            x_pxl = int(self.screen_size/2 + (self.rotation_matrix(-self.robot.orientation + np.pi/2) @ (xy-self.robot.pos))[0] * self.scale)
            y_pxl = int(self.screen_size/2 - (self.rotation_matrix(-self.robot.orientation + np.pi/2) @ (xy-self.robot.pos))[1] * self.scale)
        elif self.render_relative_to_robot == 3:
            x_pxl = int(self.screen_size/2 + (self.rotation_matrix(np.pi/2) @ (xy-self.robot.pos))[0] * self.scale)
            y_pxl = int(self.screen_size/2 - (self.rotation_matrix(np.pi/2) @ (xy-self.robot.pos))[1] * self.scale)
        return (x_pxl, y_pxl)

    def get_reward_color_map(self) -> np.ndarray:
        color_map = np.zeros((self.screen_size, self.screen_size, 3), dtype=np.uint8)
        pixel_positions = np.zeros((self.screen_size, self.screen_size, 2), dtype=np.float64)
        # Create a grid of pixel coordinates
        x_coords = (np.arange(self.screen_size) - self.screen_size / 2) / self.scale
        y_coords = (np.arange(self.screen_size) - self.screen_size / 2) / self.scale
        x_grid, y_grid = np.meshgrid(x_coords, y_coords)
        # Rotate the grid positions
        rotation_matrix = self.rotation_matrix(self.robot.orientation - np.pi)
        rotated_positions = np.einsum('ij,jkl->ikl', rotation_matrix, np.stack([x_grid, y_grid], axis=0))
        # Translate the positions to the robot's position
        pixel_positions = self.robot.pos + rotated_positions.transpose(1, 2, 0)
        # Calculate position rewards
        target_distances = np.linalg.norm(pixel_positions - self.target.pos, axis=2)
        target_proximity_rewards = 1.0 / (np.abs(target_distances - self.target.distance) + 1.0)
        obstacle_proximity_penalties = np.zeros_like(target_proximity_rewards)
        for obstacle in self.obstacles:
            obstacle_distances = np.linalg.norm(pixel_positions - obstacle.pos, axis=2)
            obstacle_proximity_penalties -= 1.0 / (np.abs(obstacle_distances - obstacle.radius) + 1.0)

        position_rewards = target_proximity_rewards + obstacle_proximity_penalties
        # Normalize rewards and create color map
        max_negative_reward = self.num_obstacles * 1.0
        positive_mask = position_rewards > 0
        negative_mask = ~positive_mask

        color_map[:, :, :] = 255  # Set the background to white
        color_map[positive_mask, 0] -= (255 * np.minimum(position_rewards[positive_mask], 1.0)).astype(np.uint8)
        color_map[positive_mask, 2] -= (255 * np.minimum(position_rewards[positive_mask], 1.0)).astype(np.uint8)
        color_map[negative_mask, 1] -= (255 * np.minimum(np.abs(position_rewards[negative_mask]), max_negative_reward) / max_negative_reward).astype(np.uint8)
        color_map[negative_mask, 2] -= (255 * np.minimum(np.abs(position_rewards[negative_mask]), max_negative_reward) / max_negative_reward).astype(np.uint8)
        return color_map

# --------------------------- SVG Exporter Class ---------------------------

class SVGExporter:
    def __init__(self, width, height):
        self.width = width
        self.height = height
        self.export_counter = 0
        self.dwg = svgwrite.Drawing(size=(self.width, self.height))
        # Create local folder ./env_svgs/ if it doesn't exist
        self.output_folder = "./env_svgs/"
        if not os.path.exists(self.output_folder):
            os.makedirs(self.output_folder)

    def create_new_drawing(self):
        self.dwg = svgwrite.Drawing(size=(self.width, self.height))

    def export(self):
        svg_file = f"./env_svgs/frame_{self.export_counter}.svg"
        #png_file = f"./env_svgs/frame_{self.export_counter}.png"
        self.dwg.saveas(svg_file)
        #cairosvg.svg2png(url=svg_file, write_to=png_file)
        self.export_counter += 1

    def draw_circle(self, color, center, radius, stroke=False, stroke_width=1, fill_opacity=1.0):
        fill_color = "rgb({},{},{})".format(*color)
        circle = self.dwg.circle(center=center, r=radius, fill=fill_color, fill_opacity=fill_opacity)
        if stroke:
            circle.stroke(fill_color, width=stroke_width)
        self.dwg.add(circle)

    def draw_line(self, color, start, end, width=1):
        stroke_color = "rgb({},{},{})".format(*color)
        line = self.dwg.line(start=start, end=end, stroke=stroke_color, stroke_width=width)
        self.dwg.add(line)

    def draw_polygon(self, color, points):
        fill_color = "rgb({},{},{})".format(*color)
        poly = self.dwg.polygon(points=points, fill=fill_color)
        self.dwg.add(poly)

    def draw_rect(self, color, top_left, size):
        fill_color = "rgb({},{},{})".format(*color)
        rect = self.dwg.rect(insert=top_left, size=size, fill=fill_color)
        self.dwg.add(rect)

    def draw_ellipse(self, color, center, width, height, angle=0.0, alpha=1.0):
        fill_color = "rgb({},{},{})".format(*color)
        ellipse = self.dwg.ellipse(center=center, r=(width/2, height/2), fill=fill_color, fill_opacity=alpha)
        if angle != 0.0:
            ellipse.rotate(angle, center=center)
        self.dwg.add(ellipse)