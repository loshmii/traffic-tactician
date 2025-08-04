"""Custom 6×6 Manhattan traffic-grid environment built on Highway-Env."""

from __future__ import annotations

from typing import Dict, Tuple

import numpy as np
from gymnasium import spaces
from highway_env.envs.common.abstract import AbstractEnv
from highway_env.envs.common.action import DiscreteMetaAction
from highway_env.envs.common.observation import ObservationType
from highway_env.road.lane import StraightLane
from highway_env.road.road import Road, RoadNetwork
from highway_env.vehicle.behavior import IDMVehicle
from highway_env.vehicle.controller import MDPVehicle

LaneIndex = Tuple[Tuple[int, int], Tuple[int, int], int]


# --------------------------------------------------------------------------- #
# Road-network builder                                                         #
# --------------------------------------------------------------------------- #
def build_grid_road(
    grid_size: int = 6,
    lane_length: float = 200.0,
    lanes_per_direction: int = 2,
) -> RoadNetwork:
    """Return a RoadNetwork forming a Manhattan 6×6 lattice plus outer lanes."""
    width = StraightLane.DEFAULT_WIDTH
    net = RoadNetwork()

    # inner lattice ---------------------------------------------------------
    for i in range(grid_size):
        for j in range(grid_size):
            if i < grid_size - 1:  # horizontal pair
                s = np.array([i * lane_length, j * lane_length])
                e = np.array([(i + 1) * lane_length, j * lane_length])
                for k in range(lanes_per_direction):
                    off = (k + 0.5) * width
                    net.add_lane((i, j), (i + 1, j), StraightLane(s + [0,  off], e + [0,  off]))
                    net.add_lane((i + 1, j), (i, j), StraightLane(e + [0, -off], s + [0, -off]))
            if j < grid_size - 1:  # vertical pair
                s = np.array([i * lane_length, j * lane_length])
                e = np.array([i * lane_length, (j + 1) * lane_length])
                for k in range(lanes_per_direction):
                    off = (k + 0.5) * width
                    net.add_lane((i, j), (i, j + 1), StraightLane(s + [-off, 0], e + [-off, 0]))
                    net.add_lane((i, j + 1), (i, j), StraightLane(e + [ off, 0], s + [ off, 0]))

    # one outer lane west (spawn) & north (goal) ----------------------------
    for k in range(lanes_per_direction):
        off = (k + 0.5) * width
        net.add_lane((-1, 0), (0, 0), StraightLane(np.array([-lane_length, 0]) + [0,  off],
                                                   np.array([0, 0])            + [0,  off]))
        net.add_lane((0, 0), (-1, 0), StraightLane(np.array([0, 0])            + [0, -off],
                                                   np.array([-lane_length, 0]) + [0, -off]))

        last = grid_size - 1
        net.add_lane((last, last),
                     (last, grid_size),
                     StraightLane(np.array([last * lane_length, last * lane_length]) + [-off, 0],
                                  np.array([last * lane_length, grid_size * lane_length]) + [-off, 0]))
        net.add_lane((last, grid_size),
                     (last, last),
                     StraightLane(np.array([last * lane_length, grid_size * lane_length]) + [ off, 0],
                                  np.array([last * lane_length, last * lane_length]) + [ off, 0]))
    return net


# --------------------------------------------------------------------------- #
# Custom action & observation                                                 #
# --------------------------------------------------------------------------- #
class GridAction(DiscreteMetaAction):
    ACTIONS = {0: "IDLE", 1: "FASTER", 2: "SLOWER", 3: "LANE_LEFT", 4: "LANE_RIGHT"}

    def __init__(self, env: AbstractEnv):
        super().__init__(env)
        self.actions = self.ACTIONS
        self.actions_indexes = {v: k for k, v in self.actions.items()}

    def space(self) -> spaces.Space:            # keeps SB3 happy
        return spaces.Discrete(len(self.actions))


class GridObservation(ObservationType):
    N_RAYS, MAX_RANGE = 16, 100.0  # lidar spec

    def space(self) -> spaces.Space:
        return spaces.Box(-np.inf, np.inf, (3 + self.N_RAYS,), np.float32)

    def observe(self):
        v = self.observer_vehicle
        speed = v.speed
        acc   = speed - self.env._last_speed
        jerk  = acc   - self.env._last_acc
        self.env._last_speed, self.env._last_acc, self.env._last_jerk = speed, acc, jerk

        dists = np.full(self.N_RAYS, self.MAX_RANGE)
        for other in self.env.road.vehicles:
            if other is v:
                continue
            rel = other.position - v.position
            dist = np.linalg.norm(rel)
            if dist < self.MAX_RANGE:
                ang = (np.arctan2(rel[1], rel[0]) - v.heading) % (2 * np.pi)
                idx = int(ang / (2 * np.pi / self.N_RAYS))
                dists[idx] = min(dists[idx], dist)
        return np.concatenate(([speed, acc, jerk], dists / self.MAX_RANGE)).astype(np.float32)


# --------------------------------------------------------------------------- #
# Main environment                                                            #
# --------------------------------------------------------------------------- #
class Manhattan6x6Env(AbstractEnv):
    metadata = {"render_modes": ["rgb_array"], "render_fps": 15}

    def __init__(self, config: Dict | None = None, render_mode: str | None = None):
        super().__init__(config)
        self.render_mode = render_mode
        self._last_speed = self._last_acc = self._last_jerk = 0.0
        self.goal_position = None

    @classmethod
    def default_config(cls) -> Dict:
        cfg = super().default_config()
        cfg.update(
            duration=2000,
            simulation_frequency=2,
            policy_frequency=2,
            grid_size=6,
            lane_length=200.0,
            spawn_vehicles=20,
        )
        return cfg

    # -- scene ----------------------------------------------------------------
    def define_spaces(self):
        self.action_type = GridAction(self)
        self.observation_type = GridObservation(self)
        self.action_space, self.observation_space = self.action_type.space(), self.observation_type.space()

    def _create_road(self, cfg):               # hooks called by AbstractEnv
        self.road = Road(build_grid_road(cfg["grid_size"], cfg["lane_length"]), np_random=self.np_random)

    def _create_vehicles(self, cfg):
        g = cfg["grid_size"]
        # ——— Ego vehicle ———
        ego_start: LaneIndex = ((-1, 0), (0, 0), 0)
        ego = MDPVehicle.make_on_lane(self.road, ego_start, longitudinal=5, speed=5)
        ego.plan_route_to((g - 1, g))
        self.goal_position = self.road.network.get_lane(
            ((g - 1, g - 1), (g - 1, g), 0)
        ).end
        self.vehicle = ego
        self.road.vehicles.append(ego)

        # ——— Background vehicles ———
        nodes = list(self.road.network.graph)  # keys are Python tuples
        n_nodes = len(nodes)
        for _ in range(cfg["spawn_vehicles"]):
            # pick a random start node
            start_idx = int(self.np_random.integers(n_nodes))
            s = nodes[start_idx]

            # pick a random neighbor of s
            neighbors = list(self.road.network.graph[s])
            tgt_idx = int(self.np_random.integers(len(neighbors)))
            t = neighbors[tgt_idx]

            # choose lane index and speed
            lane_id = int(self.np_random.integers(2))
            speed = float(self.np_random.uniform(0, 100))

            # spawn
            v = IDMVehicle.make_on_lane(self.road, (s, t, lane_id), speed)
            self.road.vehicles.append(v)
            
    def _reset(self):
        self._create_road(self.config)
        self._create_vehicles(self.config)
        self._last_speed = self._last_acc = self._last_jerk = 0.0

    # -- RL glue --------------------------------------------------------------
    def _reward(self, action: int) -> float:
        r = -0.1 - 0.01 * abs(self._last_jerk)
        if self.vehicle.crashed:
            r -= 100
        if self._goal_reached():
            r += 100
        return float(r)

    def _goal_reached(self) -> bool:
        return np.linalg.norm(self.vehicle.position - self.goal_position) < 5.0

    def _is_terminated(self) -> bool:
        return self.vehicle.crashed or self._goal_reached()

    def _is_truncated(self) -> bool:
        return self.steps >= self.config["duration"]

    def _info(self, obs, action):              # keep gym API happy
        return {}


__all__ = ["Manhattan6x6Env", "build_grid_road"]