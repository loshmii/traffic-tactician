"""Custom Manhattan traffic-grid environments built on Highway-Env.

This version drives the ego car with **direct acceleration & steering commands**
so it can be controlled by a continuous RL policy.  Key changes w.r.t. the
original draft:

* The ego is now a kinematic ``Vehicle`` (rather than a ``ControlledVehicle``)
  that obeys raw ``{"acceleration", "steering"}`` inputs.
* ``ContinuousAction`` is created with ``dynamical=True`` so the first element
  of the action vector is treated as *acceleration*.
* Background traffic still uses ``IDMVehicle`` for realism.
"""

from __future__ import annotations

from typing import Dict, Tuple

import numpy as np
from gymnasium import spaces
from gymnasium.envs.registration import register
from highway_env.envs.common.abstract import AbstractEnv
from highway_env.envs.common.action import ContinuousAction
from highway_env.envs.common.observation import ObservationType
from highway_env.road.lane import StraightLane
from highway_env.road.road import Road, RoadNetwork
from highway_env.vehicle.behavior import IDMVehicle
from highway_env.vehicle.kinematics import Vehicle  # ✨ direct-control car

LaneIndex = Tuple[Tuple[int, int], Tuple[int, int], int]

# --------------------------------------------------------------------------- #
# Road-network builder                                                         #
# --------------------------------------------------------------------------- #

def build_grid_road(
    grid_size: int = 6,
    lane_length: float = 200.0,
    lanes_per_direction: int = 2,
) -> RoadNetwork:
    """Return a RoadNetwork forming a Manhattan grid plus two outer lanes."""
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
        net.add_lane((last, last), (last, grid_size),
                     StraightLane(np.array([last * lane_length, last * lane_length]) + [-off, 0],
                                  np.array([last * lane_length, grid_size * lane_length]) + [-off, 0]))
        net.add_lane((last, grid_size), (last, last),
                     StraightLane(np.array([last * lane_length, grid_size * lane_length]) + [ off, 0],
                                  np.array([last * lane_length, last * lane_length]) + [ off, 0]))
    return net


# --------------------------------------------------------------------------- #
# Observation                                                                 #
# --------------------------------------------------------------------------- #

class GridObservation(ObservationType):
    """Simple 1-D vector of ego kinematics, goal bearing and lidar rays."""

    N_RAYS, MAX_RANGE = 16, 100.0  # lidar spec

    def space(self) -> spaces.Space:  # 3 kin + 3 goal + 1 ttc + 16 lidar = 22
        return spaces.Box(-np.inf, np.inf, (7 + self.N_RAYS,), np.float32)

    def observe(self):
        v = self.observer_vehicle

        speed = v.speed
        acc   = speed - self.env._last_speed
        jerk  = acc   - self.env._last_acc
        self.env._last_speed, self.env._last_acc, self.env._last_jerk = speed, acc, jerk

        # goal-vector -------------------------------------------------------
        vec     = self.env.goal_position - v.position
        dist    = np.linalg.norm(vec)
        bearing = (np.arctan2(vec[1], vec[0]) - v.heading) % (2 * np.pi)
        goal_obs = np.array([
            dist / self.MAX_RANGE,
            np.sin(bearing),
            np.cos(bearing),
        ])

        # lidar -------------------------------------------------------------
        dists = np.full(self.N_RAYS, self.MAX_RANGE)
        for other in self.env.road.vehicles:
            if other is v:
                continue
            rel = other.position - v.position
            d   = np.linalg.norm(rel)
            if d < self.MAX_RANGE:
                ang = (np.arctan2(rel[1], rel[0]) - v.heading) % (2 * np.pi)
                idx = int(ang / (2 * np.pi / self.N_RAYS))
                dists[idx] = min(dists[idx], d)
        # ttc -------------------------------------------------------------
        ttc = self.env._front_ttc()
        ttc_n = ttc / 10.0

        return np.concatenate(( [speed, acc, jerk], goal_obs, [ttc_n], dists / self.MAX_RANGE )).astype(np.float32)


# --------------------------------------------------------------------------- #
# Base env helper                                                             #
# --------------------------------------------------------------------------- #

def _add_background_traffic(env, count: int):
    nodes     = list(env.road.network.graph)
    n_nodes   = len(nodes)
    for _ in range(count):
        s = nodes[int(env.np_random.integers(n_nodes))]
        t = list(env.road.network.graph[s])[int(env.np_random.integers(len(env.road.network.graph[s])))]
        lane_id = int(env.np_random.integers(2))
        speed   = float(env.np_random.uniform(0, 100))
        v = IDMVehicle.make_on_lane(env.road, (s, t, lane_id), speed)
        env.road.vehicles.append(v)


# --------------------------------------------------------------------------- #
# Generic Manhattan env class                                                 #
# --------------------------------------------------------------------------- #

class _BaseManhattanEnv(AbstractEnv):
    """Shared logic between 2×2 and 6×6 variants."""

    metadata = {"render_modes": ["rgb_array"], "render_fps": 15}

    GRID_SIZE: int  # must be set by subclass

    # ------------------------------------------------------------------
    # Configuration template
    # ------------------------------------------------------------------
    @classmethod
    def default_config(cls) -> Dict:
        cfg = super().default_config()
        cfg.update(
            duration              = 2000,
            simulation_frequency  = 2,
            policy_frequency      = 2,
            lane_length           = 200.0,
            spawn_vehicles        = 20,
            # reward shaping -----------------------------------------
            lane_centering_coef   = 2.0,
            heading_coef          = 2**5,
            shaping_coef          = 2**2,
            step_cost             = -0.01,
            jerk_cost             = -0.01,
            goal_reward           = 100.0,
            crash_penalty         = -100.0,
            off_road_penalty      = -50.0,
            steer_smoothness_coef = -0.01,
            ttc_penalty_coef      = 5.0,
        )
        return cfg

    # ------------------------------------------------------------------
    # Space definitions
    # ------------------------------------------------------------------
    def define_spaces(self):
        # direct acceleration + steering (dynamical=True)
        self.action_type       = ContinuousAction(self, dynamical=True)
        self.observation_type  = GridObservation(self)
        self.action_space      = self.action_type.space()
        self.observation_space = self.observation_type.space()

    # ------------------------------------------------------------------
    # Scene creation
    # ------------------------------------------------------------------
    def _create_road(self, cfg):
        self.road = Road(build_grid_road(self.GRID_SIZE, cfg["lane_length"]), np_random=self.np_random)

    def _create_vehicles(self, cfg):
        ego_start: LaneIndex = ((-1, 0), (0, 0), 0)
        ego = Vehicle.make_on_lane(self.road, ego_start, longitudinal=0.0, speed=0.0)  # ✨ direct-control
        self.vehicle = ego
        self.controlled_vehicles.clear()
        self.controlled_vehicles.append(ego)
        self.road.vehicles.append(ego)

        # goal position --------------------------------------------------
        g = self.GRID_SIZE
        self.goal_position = self.road.network.get_lane(((g - 1, g - 1), (g - 1, g), 0)).end

        # traffic --------------------------------------------------------
        _add_background_traffic(self, cfg["spawn_vehicles"])

    # ------------------------------------------------------------------
    # TTC helpers
    # ------------------------------------------------------------------
    def _front_ttc(self, horizon = 10.0) -> float:
        # closest vehicle in front of the ego, or inf if none
        front, _ = self.road.neighbour_vehicles(self.vehicle)
        if front is None:
            return horizon
        gap = front.position[0] - self.vehicle.position[0] - front.LENGTH
        closing = max(self.vehicle.speed - front.speed, 1e-1) # avoid div by zero
        return np.clip(gap / closing, 0.0, horizon)

    # ------------------------------------------------------------------
    # Reset helpers
    # ------------------------------------------------------------------
    def _reset(self):
        self._create_road(self.config)
        self._create_vehicles(self.config)
        self._last_speed = self._last_acc = self._last_jerk = self._last_steer = 0.0
        self._prev_dist_to_goal = np.linalg.norm(self.vehicle.position - self.goal_position)
        self._ep_stats = {
            "progress": 0.0,
            "lane_centering": 0.0,
            "off_road": 0.0,
            "steering_smoothness": 0.0,
            "heading_alignment": 0.0,
            "ttc": 0.0,
            "step_cost": 0.0,
            "jerk": 0.0,
            "total_reward": 0.0,
            "crash": 0.0,
            "goal": 0.0,
        }
    # ------------------------------------------------------------------
    # RL glue
    # ------------------------------------------------------------------
    def _reward(self, action) -> float:
        cfg = self.config
        dist_now  = np.linalg.norm(self.vehicle.position - self.goal_position)
        progress  = self._prev_dist_to_goal - dist_now
        self._prev_dist_to_goal = dist_now

        # road centering ------------------------------------------------
        lane = self.road.network.get_lane(self.vehicle.lane_index)
        long, lat = lane.local_coordinates(self.vehicle.position)
        w = lane.width_at(long)
        lane_r = 1 - 2 * abs(lat / w) ** 2

        # steering smoothness -----------------------------------
        steer_rate = abs(action[1] - self._last_steer)
        self._last_steer = action[1]

        # heading alignment --------------------------------------------
        vec = self.goal_position - self.vehicle.position
        err = (np.arctan2(vec[1], vec[0]) - self.vehicle.heading + np.pi) % (2 * np.pi) - np.pi

        # time-to-collision (TTC) ----------------------------------------
        ttc = self._front_ttc()

        self._rc = {
            "progress" : cfg["shaping_coef"] * progress,
            "lane_centering": cfg["lane_centering_coef"] * lane_r,
            "off_road" : cfg["off_road_penalty"] if abs(lat) > w / 2 else 0.0,
            "steering_smoothness": cfg["steer_smoothness_coef"] * steer_rate,
            "heading_alignment": cfg["heading_coef"] * np.cos(err),
            "ttc" : cfg["ttc_penalty_coef"] * (ttc - 3.0) if ttc < 3.0 else 0.0,
            "step_cost": cfg["step_cost"],
            "jerk": cfg["jerk_cost"] * abs(self._last_jerk),
            "crash": cfg["crash_penalty"] if self.vehicle.crashed else 0.0,
            "goal": cfg["goal_reward"] if self._goal_reached() else 0.0,
        }

        for k, v in self._rc.items():
            self._ep_stats[k] += v

        r = sum(self._rc.values())
        self._ep_stats["total_reward"] += r
        return float(r)
    
    def _info(self, obs, action):
        if self._is_terminated() or self._is_truncated():
            return self._ep_stats.copy()
        return {}

    def _goal_reached(self) -> bool:
        return np.linalg.norm(self.vehicle.position - self.goal_position) < 5.0

    def _is_terminated(self):
        return self.vehicle.crashed or self._goal_reached()

    def _is_truncated(self):
        return self.steps >= self.config["duration"]

    # curriculum -----------------------------------------------------------
    def set_curriculum(self, **kwargs):
        self.config.update(kwargs)


# --------------------------------------------------------------------------- #
# Public envs                                                                 #
# --------------------------------------------------------------------------- #

class Manhattan6x6Env(_BaseManhattanEnv):
    """6×6 Manhattan grid."""

    GRID_SIZE = 6


class Manhattan2x2Env(_BaseManhattanEnv):
    """2×2 Manhattan grid – easier curriculum stage."""

    GRID_SIZE = 2


__all__ = ["Manhattan6x6Env", "Manhattan2x2Env", "build_grid_road"]

# --------------------------------------------------------------------------- #
# Gymnasium registration                                                       #
# --------------------------------------------------------------------------- #

# We expose the environments under stable IDs so that users can simply do
# `gym.make("Manhattan6x6-v0")` or `gym.make("Manhattan2x2-v0")`.
register(
    id="Manhattan6x6-v0",
    entry_point="manhattan6x6.env:Manhattan6x6Env",
    max_episode_steps=Manhattan6x6Env.default_config()["duration"],
)

register(
    id="Manhattan2x2-v0",
    entry_point="manhattan6x6.env:Manhattan2x2Env",
    max_episode_steps=Manhattan2x2Env.default_config()["duration"],
)