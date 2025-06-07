import gymnasium as gym
from gymnasium import spaces
from gymnasium.utils import seeding
import numpy as np
from .utils.physics import PhysicsEngine
from .utils.renderer import GameRenderer
from .settings import EnvSettings
import pygame


class AirHockeyEnv(gym.Env):
    metadata = {"render_modes": ["human", "rgb_array"], "render_fps": 30}

    def __init__(
        self, config: EnvSettings = EnvSettings(), render_mode: str | None = None
    ):
        super().__init__()
        self._config = config
        self.render_mode = render_mode

        # Observation space: [puck_x, puck_y, paddle_x, paddle_y, puck_vx, puck_vy]
        self.observation_space = spaces.Box(
            low=np.array(
                [0, 0, 0, 0, -self._config.max_velocity, -self._config.max_velocity],
                dtype=np.float32,
            ),
            high=np.array(
                [
                    self._config.width,
                    self._config.height,
                    self._config.width,
                    self._config.height,
                    self._config.max_velocity,
                    self._config.max_velocity,
                ],
                dtype=np.float32,
            ),
            dtype=np.float32,
        )

        # Action space: [force_x, force_y]
        self.action_space = spaces.Box(low=-1.0, high=1.0, shape=(2,), dtype=np.float32)
        self.reward_range = (-100, 100)

        self.state = np.zeros(6, dtype=np.float32)
        self.score = 0
        self.current_step = 0

        # Initialize physics
        self._physics = PhysicsEngine(self._config)

        # Initialize renderer if needed
        self._renderer = None
        if render_mode in self.metadata["render_modes"]:
            self._renderer = GameRenderer(self._config)

        # For human rendering
        self.screen = None
        self.clock = None

    def reset(self, seed=None, options=None):
        super().reset(seed=seed)

        # Reset the physics simulation
        self.state = self._physics.reset()

        # Reset tracking variables
        self.current_step = 0
        self.score = 0

        if self.render_mode == "human":
            self.render()

        return self.state, {}

    def step(self, action):
        # Clip and scale the action
        action = np.clip(action, -1.0, 1.0) * self._config.action_scale

        # Step the physics
        self.state, goal = self._physics.step(action)

        # Update step counter
        self.current_step += 1

        # Calculate reward
        if goal:
            terminated = True
            reward = self._calculate_reward(goal)
        elif self.state[2] > self._config.width / 2:
            terminated = True
            reward = self._config.reward_lose
        else:
            terminated = False
            reward = self._calculate_reward(goal)

        truncated = self.current_step >= self._config.max_steps

        # Additional info
        info = {
            "score": self.score,
            "steps": self.current_step,
            "puck_pos": self.state[:2],
            "paddle_pos": self.state[2:4],
            "puck_vel": self.state[4:6],
        }

        if self.render_mode == "human":
            self.render()

        return self.state, reward, terminated, truncated, info

    def _calculate_reward(self, goal):
        if goal == 1:
            self.score += 1
            return self._config.reward_win
        elif goal == -1:
            self.score -= 1
            return self._config.reward_lose
        puck_pos = self.state[:2]
        paddle_pos = self.state[2:4]
        diff = np.linalg.norm(puck_pos - paddle_pos)
        return self._config.reward_default + self._config.reward_alignment * (diff / self._config.width)**2

    def render(self):
        if self.render_mode == "human":
            if self.screen is None:
                pygame.init()
                self.screen = pygame.display.set_mode(
                    (self._config.width, self._config.height)
                )
                pygame.display.set_caption("Air Hockey")
            if self.clock is None:
                self.clock = pygame.time.Clock()
            self._renderer.render(self.screen, self.state)
            pygame.display.flip()
            self.clock.tick(self.metadata["render_fps"])
        elif self.render_mode == "rgb_array":
            return self._renderer.get_rgb_array(self.state)

    def close(self):
        if self._renderer is not None:
            if self.render_mode == "human" and self.screen is not None:
                pygame.quit()
                self.screen = None
            self._renderer.close()

    def seed(self, seed=None):
        self.np_random, seed = seeding.np_random(seed)
        return [seed]

    def __del__(self):
        self.close()
