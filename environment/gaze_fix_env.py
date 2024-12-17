import os
from typing import Dict, List
import gymnasium as gym
import numpy as np
import pygame
import vidmaker
import math
from environment.base_env import BaseEnv, Observation

# =====================================================================================================

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

class Obstacle:
    def __init__(self, radius = 1.0, pos = np.array([0.0, 0.0], dtype=np.float64)):
        self.radius: float = radius
        self.pos: np.ndarray = pos
        self.current_movement_direction = 0.0
        self.vel = 0.0

class Target:
    def __init__(self, x=0.0, y=0.0, distance=0.0, vel=0.0, base_movement_direction=0.0):
        self.pos = np.array([x, y], dtype=np.float64)
        self.base_movement_direction = base_movement_direction
        self.current_movement_direction = base_movement_direction
        self.distance = distance
        self.vel = vel

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

        self.max_target_distance: float = config["target_distance"]
        self.reward_margin: float = config["reward_margin"]
        self.penalty_margin: float = config["penalty_margin"]
        self.wall_collision: bool = config["wall_collision"]
        self.num_obstacles: int = config["num_obstacles"]
        self.use_obstacles: bool = config["use_obstacles"]
        self.moving_target: bool = config["moving_target"]
        self.moving_obstacles: bool = config["moving_obstacles"]
        self.observation_noise: Dict[str,float] = config["observation_noise"]

        # env dimensions
        self.world_size = config["world_size"]
        self.screen_size = SCREEN_SIZE
        self.scale = SCREEN_SIZE / self.world_size

        self.robot = Robot(np.array([-self.world_size/4, 0.0], dtype=np.float64), config["robot_sensor_angle"], config["robot_max_vel"], config["robot_max_vel_rot"], config["robot_max_acc"], config["robot_max_acc_rot"])
        self.obstacles: List[Obstacle] = []
        self.generate_target()
        self.generate_obstacles()

        self.observation_history: Dict[int,Dict[str,float]] = {}
        self.real_state_history: Dict[int,Dict[str,float]] = {}

        self.generate_observation_space()
        self.generate_action_space()

        self.collision: bool = False

        # rendering window
        self.viewer = None
        metadata = {'render_modes': ['human'], 'render_fps': 1/self.timestep}
        self.render_relative_to_robot = 3
        self.reward_render_mode = 1
        self.record_video = False
        self.video_path = ""
        self.video = None
    
    def step(self, action):
        self.current_step += 1
        self.time += self.timestep
        if self.action_mode == 2:
            action = np.array([float(action[0]-1), float(action[1]-1), float(action[2]-1)])
        self.action = self.limit_action(action) # make sure acceleration / velocity vector is within bounds
        self.update_robot_velocity()
        self.move_robot()
        if self.moving_target:
            self.move_target()
        if self.moving_obstacles:
            self.move_obstacles()
        self.last_state, rewards, done, trun, info = self._get_state(), self.get_rewards(), self.get_terminated(), False, self.get_info()
        self.last_observation = self.apply_noise(self.last_state.copy())

        # add observation to history
        self.real_state_history[self.current_step] = self.last_state.copy()
        self.observation_history[self.current_step] = self.last_observation.copy()
        if self.current_step - 2 in self.observation_history:
            del self.observation_history[self.current_step - 2]
            del self.real_state_history[self.current_step - 2]

        rew = np.sum(rewards)
        self.total_reward += rew

        return np.array(list(self.apply_noise(self.last_state).values())), rewards, done, trun, info
    
    def reset(self, seed=None, video_path = None, **kwargs):
        if seed is not None:
            super().reset(seed=seed)
        if self.video is not None:
            dirs = self.video_path.split("/")
            if len(dirs) > 1:
                os.makedirs("/".join(dirs[:-1]), exist_ok=True)
            self.video.export(verbose=True)
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
        self.last_observation = self.apply_noise(self.last_state.copy())
        self.real_state_history[self.current_step] = self.last_state.copy()
        self.observation_history[self.current_step] = self.last_observation.copy()

        return np.array(list(self.apply_noise(self.last_state).values())), info
    
    def close(self):
        pygame.quit()
        self.screen = None
        
    def _get_state(self):
        """Computes a new observation."""
        return {key: obs.calculate_value() for key, obs in self.observations.items()}
        
    def get_observation(self):
        """Return the current, unnormalized observation."""
        try:
            obs = self.observation_history[self.current_step].copy()
        except:
            raise Exception("I think this should never happen.")
            obs = self._get_state()
        return obs

    def get_reality(self):
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
            "desired_target_distance" : Observation(0.0, self.config["target_distance"], lambda: self.target.distance),
            "target_offset_angle" :     Observation(-self.robot.sensor_angle/2, np.pi, lambda: self.compute_offset_angle(self.target)),
            "del_target_offset_angle" : Observation(-2*np.pi/self.timestep/self.robot.max_vel_rot, 2*np.pi/self.timestep/self.robot.max_vel_rot, lambda: self.compute_del_offset_angle(self.target)),
            "vel_rot" :                 Observation(-1.0, 1.0, lambda: self.robot.vel_rot),
            "vel_frontal" :             Observation(-1.0, 1.0, lambda: self.robot.vel[0]),
            "vel_lateral" :             Observation(-1.0, 1.0, lambda: self.robot.vel[1]),
            "target_distance" :         Observation(0.0, np.inf, lambda: self.compute_distance(self.target)),
            "del_target_distance" :     Observation(-1.0, 1.0, lambda: self.compute_del_distance(self.target)),
        }
        for o in range(self.num_obstacles):
            self.observations[f"obstacle{o+1}_offset_angle"] = Observation(-self.robot.sensor_angle/2, np.pi, lambda o=o: self.compute_offset_angle(self.obstacles[o]))
            self.observations[f"obstacle{o+1}_radius"] = Observation(0.0, np.inf, lambda o=o: self.obstacles[o].radius)
            self.observations[f"obstacle{o+1}_coverage"] = Observation(0.0, 1.0, lambda o=o: self.compute_circle_coverage(self.obstacles[o]))
            self.observations[f"obstacle{o+1}_visual_angle"] = Observation(0.0, np.inf, lambda o=o: 2 * math.asin(min(1.0, self.obstacles[o].radius / self.compute_distance(self.obstacles[o]))))
            self.observations[f"obstacle{o+1}_distance"] = Observation(-1.0, np.inf, lambda o=o: self.compute_distance(self.obstacles[o]))
            self.observations[f"del_obstacle{o+1}_distance"] = Observation(-1.0, 1.0, lambda o=o: self.compute_del_distance(self.obstacles[o]))

        self.observation_indices = np.array([i for i in range(len(self.observations))])
        self.last_state = None

        self.required_observations = [key for key in self.observations.keys()]

        self.observation_space = gym.spaces.Box(
            low=np.array([obs.low for obs in self.observations.values()]),
            high=np.array([obs.high for obs in self.observations.values()]),
            shape=(len(self.observations),),
            dtype=np.float64
        )

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
        distance = np.random.uniform(self.world_size / 4, self.world_size / 2)
        #angle = np.random.uniform(-np.pi/2, np.pi/2)
        angle = np.random.uniform(-np.pi, np.pi)
        x = distance * np.cos(angle)
        y = distance * np.sin(angle)
        self.target = Target(x, y, np.random.uniform(0.0, self.max_target_distance), 0.5 * self.robot.max_vel, np.random.uniform(-np.pi, np.pi))
    
    def generate_obstacles(self):
        self.obstacles = []
        target_distance = np.linalg.norm(self.target.pos)
        std_dev = target_distance / 8
        midpoint = (self.target.pos + self.robot.pos) / 2
        for _ in range(self.num_obstacles):
            while True:
                radius = self.world_size / 10
                pos = np.random.normal(loc=midpoint, scale=std_dev, size=2)
                # Ensure the obstacle doesn't spawn too close to robot
                if np.linalg.norm(pos-self.robot.pos) > radius + 5 * self.robot.size:
                    self.obstacles.append(Obstacle(radius, pos))
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
            acc = self.action[:2] * self.robot.max_acc
            acc_rot = self.action[2] * self.robot.max_acc_rot
            self.robot.vel += acc * self.timestep                   # update robot velocity vector
            self.robot.vel_rot += acc_rot * self.timestep           # update rotational velocity
            self.limit_robot_velocity()
        elif self.action_mode == 3:
            self.robot.vel = self.action[:2] * self.robot.max_vel
            self.robot.vel_rot = self.action[2] * self.robot.max_vel_rot
            return

    def move_robot(self):
        # move robot
        self.robot.pos += (self.rotation_matrix(self.robot.orientation) @ self.robot.vel) * self.timestep
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
        #self.target.current_movement_direction = self.target.base_movement_direction + np.pi/3 * np.sin(self.time/4)
        self.target.current_movement_direction = np.atan2(self.target.pos[1]-self.robot.pos[1], self.target.pos[0]-self.robot.pos[0])
        self.target.pos += np.array([np.cos(self.target.current_movement_direction), np.sin(self.target.current_movement_direction)]) * self.target.vel * self.timestep
    
    def move_obstacles(self):
        for o in self.obstacles:
            #o.current_movement_direction = o.current_movement_direction + np.pi/3 * np.sin(self.time/4)
            o.pos += np.array([np.cos(o.current_movement_direction), np.sin(o.current_movement_direction)]) * o.vel * self.timestep

    def check_collision(self):
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
                obs.append(self.compute_circle_coverage(obstacle))
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

    def apply_noise(self, observation: Dict[str, float]) -> Dict[str, float]:
        """Applies gaussion noise and occlusions to real state."""
        real_observation = observation.copy()
        keys_to_delete = [key for key in real_observation if key not in self.required_observations]
        for key in keys_to_delete:
            del real_observation[key]
        for key, value in real_observation.items():
            if key[0] != "d" and key[-12:] == "offset_angle":
                if abs(real_observation[key]) > self.robot.sensor_angle / 2:
                    real_observation[key] = None
                    if "del_"+key in real_observation.keys():
                        real_observation["del_"+key] = None
            if real_observation[key] is not None:
                if key in self.observation_noise.keys():
                    real_observation[key] = value + self.observation_noise[key] * np.random.randn()
        return real_observation
    
    def compute_offset_angle(self, obj: Target|Obstacle) -> float:
        return self.normalize_angle(np.arctan2(obj.pos[1]-self.robot.pos[1], obj.pos[0]-self.robot.pos[0]) - self.robot.orientation)
    
    def compute_del_offset_angle(self, obj: Target|Obstacle):
        offset_angle = self.compute_offset_angle(obj)
        # angular vel due to robot's rotation
        del_offset_angle = - self.robot.vel_rot
        # angular vel due to robot's translation
        del_offset_angle -= (self.rotation_matrix(-offset_angle) @ self.robot.vel)[1] / self.compute_distance(obj)
        # angular vel due to object's movement
        if (type(obj) == Target and self.moving_target) or (type(obj) == Obstacle and self.moving_obstacles):
            del_offset_angle += obj.vel * np.sin(obj.current_movement_direction - offset_angle - self.robot.orientation) / self.compute_distance(obj)
        return del_offset_angle
    
    def compute_distance(self, obj: Target|Obstacle):
        return np.linalg.norm(obj.pos-self.robot.pos)

    def compute_del_distance(self, obj: Target|Obstacle):
        offset_angle = self.compute_offset_angle(obj)
        del_object_distance = - (self.rotation_matrix(-offset_angle) @ self.robot.vel)[0]
        if (type(obj) == Target and self.moving_target) or (type(obj) == Obstacle and self.moving_obstacles):
            del_object_distance += obj.vel * np.cos(obj.current_movement_direction - offset_angle - self.robot.orientation)
        return del_object_distance
    
    def compute_circle_coverage(self, obstacle: Obstacle):
        angular_size = 2 * math.asin(min(1.0, obstacle.radius / self.compute_distance(obstacle)))
        coverage = angular_size / self.robot.sensor_angle
        return min(coverage, 1.0)

    # ----------------------------------- render stuff -----------------------------------------

    def render(self, real_time_factor = 1.0, robot_frame_means: Dict[str, np.ndarray] = None, robot_frame_covs: Dict[str, np.ndarray] = None):
        if self.viewer is None:
            pygame.init()
            # Clock to control frame rate
            self.rt_clock = pygame.time.Clock()
            # set window
            self.viewer = pygame.display.set_mode((self.screen_size, self.screen_size))
            pygame.display.set_caption("Gaze Fixation")
            if self.record_video:
                self.video = vidmaker.Video(self.video_path, fps=int(1/self.timestep), resolution=(self.screen_size,self.screen_size), late_export=True)
        
        self.draw_env()

        # draw an arrow for the robot's action
        self.draw_arrow(self.robot.pos, self.robot.orientation+math.atan2(self.action[1],self.action[0]), self.robot.size*10*(np.linalg.norm(self.action[:2])), self.robot.size*2, RED if self.action_mode in [1,2] else BLUE)
        if self.action_mode in [1,2]:
            # draw an arrow for the robot's velocity
            self.draw_arrow(self.robot.pos, self.robot.orientation+math.atan2(self.robot.vel[1],self.robot.vel[0]), self.robot.size*10*(np.linalg.norm(self.robot.vel)/self.robot.max_vel), self.robot.size*1.5, BLUE)
        if self.moving_target:
            # draw an arrow for the target's movement
            self.draw_arrow(self.target.pos, self.target.current_movement_direction, self.robot.size*10*self.target.vel/self.robot.max_vel, self.robot.size*1.5, DARK_GREEN)

        if robot_frame_means is not None and robot_frame_covs is not None:
            assert len(robot_frame_means) == len(robot_frame_covs), "Number of means and covariances must be equal"
            assert len(robot_frame_means) <= 6, "Only up to 6 estimator states can be visualized"
            for i, key in enumerate(robot_frame_means.keys()):
                mean = self.rotation_matrix(self.robot.orientation) @ robot_frame_means[key][:2] + self.robot.pos
                self.draw_gaussian(mean, robot_frame_covs[key][:2,:2], COLORS[i])
                pygame.draw.circle(self.viewer, COLORS[i], self.pxl_coordinates(mean), 5)

        self.display_info()
        if self.record_video:
            self.video.update(pygame.surfarray.pixels3d(self.viewer).swapaxes(0, 1), inverted=False)
        pygame.display.flip()
        self.rt_clock.tick(1/self.timestep*real_time_factor)

    def draw_gaussian(self, world_frame_mean, robot_frame_cov, color):
        eigvals, eigvecs = np.linalg.eig(robot_frame_cov)
        eigvals = np.real(eigvals)
        eigvecs = np.real(eigvecs)
        if self.render_relative_to_robot == 2:
            angle = - np.arctan2(eigvecs[1,0], eigvecs[0,0]) + np.pi/2
        else:
            angle = - np.arctan2(eigvecs[1,0], eigvecs[0,0]) + np.pi/2 - self.robot.orientation
        width, height = 2*np.sqrt(eigvals)
        width = max(0.1, width)
        height = max(0.1, height)
        ellipse_surface = pygame.Surface((width*self.scale, height*self.scale), pygame.SRCALPHA)
        pygame.draw.ellipse(ellipse_surface, (*color, 128), ellipse_surface.get_rect())
        rotated_ellipse = pygame.transform.rotate(ellipse_surface, -np.degrees(angle))
        ellipse_rect = rotated_ellipse.get_rect(center=self.pxl_coordinates(world_frame_mean))
        self.viewer.blit(rotated_ellipse, ellipse_rect)

    def draw_env(self, reward_render_mode = 1):
        # Fill the screen
        if reward_render_mode == 1:
            self.viewer.fill(GREY)
            self.draw_fov()
        else:
            color_map = self.get_reward_color_map()
            pygame.surfarray.blit_array(self.viewer, color_map)

        self.render_grid()
        if reward_render_mode == 1 and self.reward_margin < 20.0:
            # TODO: deal better with reward and penalty margins
            # draw target reward margin
            self.transparent_circle(self.target.pos, self.target.distance+self.reward_margin, GREEN)
        # draw target distance
        pygame.draw.circle(self.viewer, DARK_GREEN, self.pxl_coordinates((self.target.pos[0],self.target.pos[1])), self.target.distance*self.scale, width=2)
        # draw target
        pygame.draw.circle(self.viewer, DARK_GREEN, self.pxl_coordinates((self.target.pos[0],self.target.pos[1])), self.robot.size*self.scale)
        # draw vision axis
        pygame.draw.line(self.viewer, BLACK, self.pxl_coordinates((self.robot.pos[0],self.robot.pos[1])), self.pxl_coordinates(self.polar_point(self.robot.orientation,self.world_size*3)))
        # draw Agent
        pygame.draw.circle(self.viewer, BLUE, self.pxl_coordinates((self.robot.pos[0],self.robot.pos[1])), self.robot.size*self.scale)
        pygame.draw.polygon(self.viewer, BLUE, [self.pxl_coordinates(self.polar_point(self.robot.orientation+np.pi/2, self.robot.size/1.25)), self.pxl_coordinates(self.polar_point(self.robot.orientation-np.pi/2, self.robot.size/1.25)), self.pxl_coordinates(self.polar_point(self.robot.orientation, self.robot.size*2.0))])
        # draw obstacles
        if self.use_obstacles:
            for o in self.obstacles:
                pygame.draw.circle(self.viewer, BLACK, self.pxl_coordinates((o.pos[0],o.pos[1])), o.radius*self.scale)
                # TODO: deal better with reward and penalty margins
                if reward_render_mode == 1 and self.penalty_margin < 20.0:
                    self.transparent_circle(o.pos, o.radius+self.penalty_margin, RED)

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
        pygame.draw.line(self.viewer, color, self.pxl_coordinates(pos), self.pxl_coordinates(end_point), width=1)
        pygame.draw.line(self.viewer, color, self.pxl_coordinates(end_point), self.pxl_coordinates(tip_point_1), width=1)
        pygame.draw.line(self.viewer, color, self.pxl_coordinates(end_point), self.pxl_coordinates(tip_point_2), width=1)

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
        elif abs(self.robot.sensor_angle - 2*np.pi) < 0.01:
            self.viewer.fill(WHITE)
        else:
            self.viewer.fill(WHITE)
            robot_point = self.pxl_coordinates(self.robot.pos)
            left_angle = self.pxl_coordinates(self.polar_point(self.robot.orientation+self.robot.sensor_angle/2, self.world_size*3))
            right_angle = self.pxl_coordinates(self.polar_point(self.robot.orientation-self.robot.sensor_angle/2, self.world_size*3))
            left_corner = self.pxl_coordinates(self.polar_point(self.robot.orientation+3*np.pi/4, self.world_size))
            right_corner = self.pxl_coordinates(self.polar_point(self.robot.orientation-3*np.pi/4, self.world_size))
            pygame.draw.polygon(self.viewer, GREY, [robot_point, left_angle, left_corner, right_corner, right_angle])

    def polar_point(self, angle, distance, start_pos = None):
        if start_pos is None:
            start_pos = self.robot.pos
        return start_pos[0] + distance * math.cos(angle), start_pos[1] + distance * math.sin(angle)
    
    def display_info(self):
        font = pygame.font.Font(None, 24)
        legend = [
            (f'Step:', f'{self.current_step}'),                                # clock
            (f'Time:', f'{self.time:.2f}'),                                 # time
            # (f'Step reward:', f'{np.sum(self.get_rewards()):.4f}'),         # step reward
            # (f'Total reward:', f'{self.total_reward:.4f}'),                 # episode reward
            (f'Target Distance:', f'{self.compute_distance(self.target):.2f}'),   # target distance
            (f'Desired Distance:', f'{self.target.distance:.2f}'),          # desired target distance
        ]
        for i, (text, value) in enumerate(legend):
            self.viewer.blit(font.render(text, True, BLACK), (10, 5+25*i))
            self.viewer.blit(font.render(value, True, BLACK), (160, 5+25*i))
        self.viewer.blit(font.render("Robot", True, BLACK), (self.pxl_coordinates(self.robot.pos)) + np.array([10,-20]))
        self.viewer.blit(font.render("Target", True, BLACK), (self.pxl_coordinates(self.target.pos)) + np.array([10,-20]))

    def render_grid(self):
        lines = []
        for x in range(int(self.robot.pos[0]-self.world_size/2), int(self.robot.pos[0]+self.world_size/2+1)):
            if x % 10 == 0:
                lines.append(((float(x),self.robot.pos[1]+self.world_size), (float(x),self.robot.pos[1]-self.world_size)))
        for y in range(int(self.robot.pos[1]-self.world_size/2), int(self.robot.pos[1]+self.world_size/2+1)):
            if y % 10 == 0:
                lines.append(((self.robot.pos[0]+self.world_size,float(y)), (self.robot.pos[0]-self.world_size,float(y))))
        for line in lines:
            pygame.draw.line(self.viewer, BLACK, self.pxl_coordinates(line[0]), self.pxl_coordinates(line[1]))

    def pxl_coordinates(self, xy):
        if self.render_relative_to_robot == 1:
            x_pxl = int(self.screen_size/2 + xy[0] * self.scale)
            y_pxl = int(self.screen_size/2 - xy[1] * self.scale)
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