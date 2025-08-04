import numpy as np
import networkx as nx
from gymnasium import Env, spaces
from gymnasium.utils import seeding
from highway_env.envs.common.abstract import AbstractEnv


class Manhattan6x6Env(AbstractEnv):
    """Simple 6x6 Manhattan grid driving environment.

    The environment is a minimal grid-world style simulator that mimics the
    :class:`highway_env` API while remaining lightweight for testing and
    demonstration purposes.  It features an ego vehicle starting at (0, 0)
    facing east and a goal at (5, 5).

    Observation:
        ``[x/5, y/5, cos(theta), sin(theta), lidar(60)]`` where lidar rays are
        binary occupancy indicators sampled around the vehicle.

    Action space:
        ``0: maintain`` – move forward with current speed (1 cell/step)
        ``1: accelerate`` – set speed to 1 and move forward
        ``2: brake`` – set speed to 0 (no movement)
        ``3: left`` – turn left and move one cell
        ``4: right`` – turn right and move one cell

    Reward:
        ``-0.1`` per step, ``-1`` for collision/out-of-road, ``+100`` for
        reaching the goal.
    """

    metadata = {"render_modes": ["rgb_array"], "render_fps": 15}

    def __init__(self, render_mode: str | None = None):
        Env.__init__(self)
        self.render_mode = render_mode
        self.grid_size = 6
        self.start = (0, 0)
        self.goal = (5, 5)
        self.max_steps = 1000
        self.action_space = spaces.Discrete(5)
        # ego kinematics (4) + 60 lidar rays
        self.observation_space = spaces.Box(
            low=-np.inf, high=np.inf, shape=(64,), dtype=np.float32
        )
        self.viewer = None
        self.np_random, _ = seeding.np_random()
        self.reset()

    def reset(self, seed: int | None = None, options: dict | None = None):
        if seed is not None:
            self.np_random, _ = seeding.np_random(seed)
        self.position = np.array(self.start, dtype=int)
        self.heading = 0  # 0:E,1:N,2:W,3:S
        self.speed = 1
        self.steps = 0
        # spawn background vehicles away from any shortest path
        self.background = self._spawn_background()
        self.graph = self._build_graph()
        obs = self._get_obs()
        return obs, {}

    def _spawn_background(self):
        path1 = [(0, j) for j in range(1, self.grid_size)] + [
            (i, self.grid_size - 1) for i in range(1, self.grid_size)
        ]
        path2 = [(i, 0) for i in range(1, self.grid_size)] + [
            (self.grid_size - 1, j) for j in range(1, self.grid_size)
        ]
        forbidden = set(path1 + path2 + [self.start, self.goal])
        cells = [
            (i, j)
            for i in range(self.grid_size)
            for j in range(self.grid_size)
            if (i, j) not in forbidden
        ]
        positions = [cells[self.np_random.integers(len(cells))] for _ in range(200)]
        return set(positions)

    def _build_graph(self):
        g = nx.DiGraph()
        for i in range(self.grid_size):
            for j in range(self.grid_size):
                if i < self.grid_size - 1:
                    g.add_edge((i, j), (i + 1, j))
                    g.add_edge((i + 1, j), (i, j))
                if j < self.grid_size - 1:
                    g.add_edge((i, j), (i, j + 1))
                    g.add_edge((i, j + 1), (i, j))
        return g

    def _get_obs(self):
        x, y = self.position
        dx, dy = [(1, 0), (0, 1), (-1, 0), (0, -1)][self.heading]
        lidar = np.zeros(60, dtype=np.float32)
        obs = np.array([x / 5, y / 5, dx, dy], dtype=np.float32)
        return np.concatenate([obs, lidar])

    def step(self, action: int):
        assert self.action_space.contains(action)
        self.steps += 1
        reward = -0.1
        terminated = False
        truncated = False
        if action == 1:  # accelerate
            self.speed = 1
        elif action == 2:  # brake
            self.speed = 0
        elif action == 3:  # left turn
            self.heading = (self.heading + 1) % 4
        elif action == 4:  # right turn
            self.heading = (self.heading - 1) % 4

        if self.speed > 0 and action != 2:
            dx, dy = [(1, 0), (0, 1), (-1, 0), (0, -1)][self.heading]
            new_pos = self.position + (dx, dy)
            if (
                new_pos[0] < 0
                or new_pos[0] >= self.grid_size
                or new_pos[1] < 0
                or new_pos[1] >= self.grid_size
            ):
                terminated = True
                reward -= 1
            elif tuple(new_pos) in self.background:
                terminated = True
                reward -= 1
            else:
                self.position = new_pos

        if tuple(self.position) == self.goal:
            terminated = True
            reward += 100

        if self.steps >= self.max_steps:
            truncated = True

        obs = self._get_obs()
        return obs, float(reward), terminated, truncated, {}

    def render(self, mode: str = "rgb_array"):
        cell = 20
        img = np.ones((self.grid_size * cell, self.grid_size * cell, 3), dtype=np.uint8) * 255
        for (x, y) in self.background:
            img[
                y * cell : (y + 1) * cell,
                x * cell : (x + 1) * cell,
            ] = 0
        x, y = self.position
        img[
            y * cell : (y + 1) * cell,
            x * cell : (x + 1) * cell,
        ] = [255, 0, 0]
        gx, gy = self.goal
        img[
            gy * cell : (gy + 1) * cell,
            gx * cell : (gx + 1) * cell,
        ] = [0, 255, 0]
        return img
