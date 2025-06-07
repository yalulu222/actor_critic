import numpy as np
import Box2D
from Box2D.b2 import world, circleShape, edgeShape
from ..settings import EnvSettings


class PhysicsEngine:
    def __init__(self, config: EnvSettings):
        self.config = config
        self.PPM = 100
        self.world = Box2D.b2World(gravity=(0, 0), doSleep=True)
        self._setup_world()

    def _setup_world(self):
        width_m = self.config.width / self.PPM
        height_m = self.config.height / self.PPM
        puck_radius_m = self.config.puck_radius / self.PPM
        paddle_radius_m = self.config.paddle_radius / self.PPM

        self._create_boundaries(width_m, height_m)

        self.puck = self.world.CreateDynamicBody(
            position=(width_m / 2, height_m / 2), bullet=True
        )
        self.puck.CreateCircleFixture(
            radius=puck_radius_m, density=0.1, restitution=0.9, friction=0.1
        )

        self.paddle = self.world.CreateDynamicBody(
            position=(width_m / 4, height_m / 2), bullet=True
        )
        self.paddle.CreateCircleFixture(
            radius=paddle_radius_m, density=1.0, restitution=0.3, friction=0.8
        )

    def _create_boundaries(self, width_m, height_m):
        self.world.CreateStaticBody(
            position=(0, 0),
            shapes=[
                edgeShape(vertices=[(0, 0), (width_m, 0)]),
                edgeShape(vertices=[(width_m, 0), (width_m, height_m)]),
                edgeShape(vertices=[(width_m, height_m), (0, height_m)]),
                edgeShape(vertices=[(0, height_m), (0, 0)]),
            ],
        )

        goal_height_m = self.config.goal_height / self.PPM
        side_gap_m = (height_m - goal_height_m) / 2

        self.world.CreateStaticBody(
            position=(0, 0),
            shapes=[
                edgeShape(vertices=[(0, 0), (0, side_gap_m)]),
                edgeShape(vertices=[(0, side_gap_m + goal_height_m), (0, height_m)]),
            ],
        )

        self.world.CreateStaticBody(
            position=(width_m, 0),
            shapes=[
                edgeShape(vertices=[(0, 0), (0, side_gap_m)]),
                edgeShape(vertices=[(0, side_gap_m + goal_height_m), (0, height_m)]),
            ],
        )

    def reset(self):
        width_m = self.config.width / self.PPM
        height_m = self.config.height / self.PPM
        puck_radius = self.config.puck_radius / self.PPM

        self.puck.position = width_m / 3, np.random.uniform(puck_radius, height_m - puck_radius)
        self.puck.linearVelocity = (0, 0)
        self.puck.angularVelocity = 0

        self.paddle.position = (width_m / 5, height_m / 2)
        self.paddle.linearVelocity = (0, 0)
        self.paddle.angularVelocity = 0

        return self.get_state()


    def step(self, action):
        if action is not None:
            self.paddle.linearVelocity = Box2D.b2Vec2(
                float(action[0]) / self.PPM, float(action[1]) / self.PPM
            )
        self.world.Step(1.0 / self.config.fps, 6, 2)
        return self.get_state(), self._check_goal()

    def get_state(self):
        puck_pos = self.puck.position * self.PPM
        paddle_pos = self.paddle.position * self.PPM
        puck_vel = self.puck.linearVelocity * self.PPM

        return np.array(
            [
                puck_pos.x,
                puck_pos.y,
                paddle_pos.x,
                paddle_pos.y,
                puck_vel.x,
                puck_vel.y,
            ],
            dtype=np.float32,
        )

    def _check_goal(self):
        puck_x = self.puck.position.x * self.PPM
        puck_y = self.puck.position.y * self.PPM
        screen_width = self.config.width
        goal_x = self.config.puck_radius * 2
        goal_top = (self.config.height - self.config.goal_height) / 2
        goal_bottom = goal_top + self.config.goal_height

        if puck_x < goal_x and goal_top <= puck_y <= goal_bottom:
            return -1
        if (
            puck_x > screen_width - goal_x
            and goal_top <= puck_y <= goal_bottom
        ):
            return 1
        return 0

    def set_paddle_position(self, position):
        self.paddle.position = (
            position[0] / self.PPM,
            position[1] / self.PPM,
        )
        self.paddle.linearVelocity = (0, 0)
