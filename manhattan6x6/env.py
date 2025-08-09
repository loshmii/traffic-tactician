"""Custom Manhattan traffic-grid environments built on Highway-Env.

This version drives the ego car with **direct acceleration & steering commands**
so it can be controlled by a continuous RL policy.

Key changes w.r.t. the original draft:

* The ego is now a kinematic `Vehicle` (rather than a `ControlledVehicle`)
  that obeys raw `{"acceleration", "steering"}` inputs.
* `ContinuousAction` is created with `dynamical=True` so the first element
  of the action vector is treated as *acceleration*.
* Background traffic still uses `IDMVehicle` for realism.
* Soft off-road buffer handling (0.5 m):
    - inside buffer – small per-step penalty, no episode termination
    - outside buffer – hard penalty, episode termination

new features (Aug-2025)
---------------------------------------------------------------------
* Two extra observations per step:
    1. **Heading-to-lane** (∆ψ) → provided as sine & cosine
    2. **Next-waypoint bearing** → sine & cosine to a look-ahead point 20 m ahead
* Reward shaping now uses **heading to next waypoint** (turn-friendly)
  instead of heading-to-lane.
* **Intersection-aware off-road** check: in the last ~10 m of a lane, we
  consider both the current lane and the best outgoing lane to avoid
  spurious hard off-road flags while initiating a turn.
* Observation length grows from `8 + N_RAYS` to `12 + N_RAYS`.
* Lane-aware TTC penalty.
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
from highway_env.vehicle.kinematics import Vehicle  # direct-control car

LaneIndex = Tuple[Tuple[int, int], Tuple[int, int], int]

# --------------------------------------------------------------------------- #
# Utility helpers                                                             #
# --------------------------------------------------------------------------- #

def wrap_to_pi(angle: float | np.ndarray) -> float | np.ndarray:
    """Wrap `angle` to (−π, π]."""
    return (angle + np.pi) % (2 * np.pi) - np.pi


# --------------------------------------------------------------------------- #
# Road-network builder                                                        #
# --------------------------------------------------------------------------- #

def build_grid_road(
    grid_size: int = 2,           # core is 2×2
    lane_length: float = 200.0,
    lanes_per_direction: int = 2,
) -> RoadNetwork:
    """2×2 core plus exactly one-cell extension on each side (“#” with long bars)."""
    width = StraightLane.DEFAULT_WIDTH
    net   = RoadNetwork()

    # ── Horizontal bars (core rows j=0,1; i goes from -1→0, 0→1, 1→2) ─────
    for j in range(0, grid_size):           # j = 0,1
        for i in range(-1, grid_size):      # i = -1,0,1
            s = np.array([i * lane_length,     j * lane_length])
            e = np.array([(i + 1) * lane_length, j * lane_length])
            for k in range(lanes_per_direction):
                off = (k + 0.5) * width
                # forward lane
                net.add_lane(
                    (i, j), (i + 1, j),
                    StraightLane(s + [0,  off], e + [0,  off])
                )
                # backward lane
                net.add_lane(
                    (i + 1, j), (i, j),
                    StraightLane(e + [0, -off], s + [0, -off])
                )

    # ── Vertical bars (core columns i=0,1; j goes from -1→0, 0→1, 1→2) ────
    for i in range(0, grid_size):           # i = 0,1
        for j in range(-1, grid_size):      # j = -1,0,1
            s = np.array([i * lane_length,     j * lane_length])
            e = np.array([i * lane_length, (j + 1) * lane_length])
            for k in range(lanes_per_direction):
                off = (k + 0.5) * width
                # “upward” lane
                net.add_lane(
                    (i, j), (i, j + 1),
                    StraightLane(s + [-off, 0], e + [-off, 0])
                )
                # “downward” lane
                net.add_lane(
                    (i, j + 1), (i, j),
                    StraightLane(e + [ off, 0], s + [ off, 0])
                )

    return net


# --------------------------------------------------------------------------- #
# Observation                                                                 #
# --------------------------------------------------------------------------- #

class GridObservation(ObservationType):
    """1-D vector of ego kinematics, goal bearing, TTC, lane metrics and lidar."""

    N_RAYS, MAX_RANGE = 16, 100.0  # lidar spec

    def space(self) -> spaces.Space:
        return spaces.Box(-np.inf, np.inf, (12 + self.N_RAYS,), np.float32)

    def observe(self):
        v = self.observer_vehicle

        # Kinematics
        speed = v.speed
        acc   = speed - self.env._last_speed
        jerk  = acc   - self.env._last_acc
        self.env._last_speed, self.env._last_acc, self.env._last_jerk = speed, acc, jerk

        # Goal vector
        vec     = self.env.goal_position - v.position
        dist    = np.linalg.norm(vec)
        bearing = (np.arctan2(vec[1], vec[0]) - v.heading) % (2 * np.pi)
        goal_obs = np.array([
            dist / self.MAX_RANGE,
            np.sin(bearing),
            np.cos(bearing),
        ])

        # Lane-offset
        lane_idx, _ = self.env._closest_lane_index()
        lane = self.env.road.network.get_lane(lane_idx)
        long, lat    = lane.local_coordinates(v.position)
        w            = lane.width_at(long)
        off_centre   = np.clip(lat / (w / 2), -1.0, 1.0)

        # Heading-to-lane (kept in observation)
        dpsi      = wrap_to_pi(v.heading - lane.heading_at(long))
        dpsi_obs  = np.array([np.sin(dpsi), np.cos(dpsi)])

        # Next-waypoint bearing (used by policy + reward)
        wp_vec   = self.env._next_waypoint() - v.position
        wp_bear  = (np.arctan2(wp_vec[1], wp_vec[0]) - v.heading) % (2 * np.pi)
        wp_obs   = np.array([np.sin(wp_bear), np.cos(wp_bear)])

        # Lidar
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

        # Time-to-collision
        ttc   = self.env._front_ttc()
        ttc_n = ttc / 10.0

        return np.concatenate((
            [speed, acc, jerk],
            goal_obs,
            [ttc_n],
            [off_centre],
            dpsi_obs,
            wp_obs,
            dists / self.MAX_RANGE,
        )).astype(np.float32)


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
    GRID_SIZE: int

    # -------------- safe lane edge-distance fallback -------------------- #
    @staticmethod
    def _lane_edge_distance(lane: StraightLane, position: np.ndarray) -> float:
        """Return signed distance to lane edge.
        Falls back to geometry if the lane class has no `edge_distance`.
        Positive → inside; negative → outside.
        """
        if hasattr(lane, "edge_distance"):
            return float(lane.edge_distance(position))
        # Fallback using local coordinates
        long, lat = lane.local_coordinates(position)
        half_w = 0.5 * lane.width_at(long)
        return float(half_w - abs(lat))

    def _edge_distance_intersection_aware(self, lane_idx: LaneIndex, lookahead: float = 10.0) -> float:
        """Union-of-lanes edge distance near intersections.
        In the last `lookahead` meters, take the max edge-distance between the
        current lane and the best outgoing lane (towards goal).
        """
        pos  = self.vehicle.position
        lane = self.road.network.get_lane(lane_idx)
        long, _ = lane.local_coordinates(pos)
        d_curr = self._lane_edge_distance(lane, pos)
        if long <= lane.length - lookahead:
            return d_curr
        # consider best outgoing lane close to the end of current lane
        outs = self.road.network.outgoing_lanes(lane_idx)
        if not outs:
            return d_curr
        best = min(
            outs,
            key=lambda idx: self._network_distance_to_goal(
                self.road.network.get_lane(idx).position(0, 0), idx)
        )
        d_next = self._lane_edge_distance(self.road.network.get_lane(best), pos)
        return max(d_curr, d_next)

    @classmethod
    def default_config(cls) -> Dict:
        cfg = super().default_config()
        cfg.update(
            duration=2000,
            simulation_frequency=2,
            policy_frequency=2,
            lane_length=200.0,
            spawn_vehicles=20,
            # reward shaping using potential-based reward
            alpha=5.0/200.0,
            beta=1.0,
            gamma=1.0,
            step_cost=-0.01,
            jerk_cost=0.0,
            goal_reward=100.0,
            crash_penalty=-100.0,
            off_road_penalty=-50.0,
            soft_offroad_margin=-2.0,
            soft_offroad_step=-0.1,
            steer_smoothness_coef=0.0,
            ttc_penalty_coef=5.0,
        )
        return cfg

    def define_spaces(self):
        self.action_type       = ContinuousAction(self, dynamical=True)
        self.observation_type  = GridObservation(self)
        self.action_space      = self.action_type.space()
        self.observation_space = self.observation_type.space()

    def _create_road(self, cfg):
        self.road = Road(build_grid_road(self.GRID_SIZE, cfg["lane_length"]), np_random=self.np_random)

    def _create_vehicles(self, cfg):
        ego_start: LaneIndex = ((-1, 0), (0, 0), 0)
        ego = Vehicle.make_on_lane(self.road, ego_start, longitudinal=0.0, speed=0.0)
        self.vehicle = ego
        self.controlled_vehicles.clear()
        self.controlled_vehicles.append(ego)
        self.road.vehicles.append(ego)

        g = self.GRID_SIZE
        self.goal_position = self.road.network.get_lane(((g - 1, g - 1), (g - 1, g), 0)).end

        _add_background_traffic(self, cfg["spawn_vehicles"])

    def _closest_lane_index(self, position=None, heading=None):
        pos = self.vehicle.position if position is None else position
        idx = self.road.network.get_closest_lane_index(pos, heading)
        # use robust edge-distance (with fallback)
        edge_d = self._lane_edge_distance(self.road.network.get_lane(idx), pos)
        return idx, float(edge_d)

    def _front_ttc(self, horizon: float = 10.0) -> float:
        front, _ = self.road.neighbour_vehicles(self.vehicle)
        if front is None:
            return horizon
        lane_idx, _ = self._closest_lane_index()
        lane = self.road.network.get_lane(lane_idx)
        ego_s, _ = lane.local_coordinates(self.vehicle.position)
        front_s, _ = lane.local_coordinates(front.position)
        gap = front_s - ego_s - front.LENGTH
        ego_v = np.dot(self.vehicle.velocity, lane.direction)
        front_v = np.dot(front.velocity, lane.direction)
        closing = max(ego_v - front_v, 1e-1)
        return float(np.clip(gap / closing, 0.0, horizon))

    def _next_waypoint(self, lookahead: float = 30.0) -> np.ndarray:
        """Position of a look-ahead point 20 m ahead (in next lane if upcoming exit)."""
        lane_idx, _ = self._closest_lane_index()
        lane = self.road.network.get_lane(lane_idx)
        long, _ = lane.local_coordinates(self.vehicle.position)
        if long < lane.length - lookahead - 5.0:
            return lane.position(long + lookahead, 0)
        candidates = self.road.network.outgoing_lanes(lane_idx)
        best = min(
            candidates,
            key=lambda idx: self._network_distance_to_goal(
                self.road.network.get_lane(idx).position(0,0), idx)
        )
        return self.road.network.get_lane(best).position(lookahead, 0)

    def _network_distance_to_goal(self, position: np.ndarray, lane_index: LaneIndex) -> float:
        lane      = self.road.network.get_lane(lane_index)
        lane_len  = self.config["lane_length"]
        long, _   = lane.local_coordinates(position)
        remaining_on_lane = max(lane_len - long, 0.0)
        _, end_node, _ = lane_index
        curr_i, curr_j = end_node
        goal_i, goal_j = self.GRID_SIZE - 1, self.GRID_SIZE
        edges_remaining = abs(goal_i - curr_i) + abs(goal_j - curr_j)
        return remaining_on_lane + edges_remaining * lane_len
    
    def off_road_update(self):
        """Classify the ego position and return the penalty
        (0 if still on lane). Also updates the flags + episode stats.
        Intersection-aware near corners so turns aren't spuriously flagged.
        """
        cfg = self.config
        # closest *current* lane index
        lane_idx, _ = self._closest_lane_index()
        self.vehicle.lane_index = lane_idx  # update lane index in the vehicle

        # union-of-lanes edge distance near intersection
        edge_d = self._edge_distance_intersection_aware(lane_idx, lookahead=10.0)

        if edge_d >= 0.0:
            self.off_road_hard = self.off_road_soft = False
            return 0.0
        if edge_d >= cfg["soft_offroad_margin"]:
            # soft off road
            self.off_road_soft = True
            return cfg["soft_offroad_step"]
        self.off_road_hard = True
        return cfg["off_road_penalty"]

    def _reset(self):
        self._create_road(self.config)
        self._create_vehicles(self.config)
        self._last_speed = self._last_acc = self._last_jerk = self._last_steer = 0.0
        self.off_road_hard = self.off_road_soft = False
        # initialize potential-based shaping (turn-friendly)
        lane_idx, _ = self._closest_lane_index()
        lane = self.road.network.get_lane(lane_idx)
        long, lat = lane.local_coordinates(self.vehicle.position)
        w = lane.width_at(long)
        lat_error = lat / (0.5 * w)
        dist_goal = self._network_distance_to_goal(self.vehicle.position, lane_idx)
        wp = self._next_waypoint()
        desired_heading = np.arctan2((wp - self.vehicle.position)[1], (wp - self.vehicle.position)[0])
        heading_error = wrap_to_pi(self.vehicle.heading - desired_heading)
        self._phi_prev = (
            self.config["alpha"] * dist_goal +
            self.config["beta"]  * abs(lat_error) +
            self.config["gamma"] * abs(heading_error)
        )
        self._ep_stats = {
            "total_reward": 0.0,
            "shape": 0.0,
            "ttc": 0.0,
            "step_cost": 0.0,
            "jerk": 0.0,
            "crash": 0.0,
            "goal": 0.0,
            "off_road_hard_event": 0,
            "off_road_soft_penalty": 0.0,
            "dist0" : float(dist_goal),
            "dist" : float(dist_goal),
            "dist_min" : float(dist_goal),
        }
        self.steps = 0
    
    def step(self, action):
        """One simulatior tick with counters, rewards and termination checks."""
        obs, reward, terminated, truncated, info = super().step(action)
        self.steps += 1
        return obs, reward, terminated, truncated, info

    def _reward(self, action) -> float:
        cfg = self.config
        # ---- compute geometric errors ---------------------------------
        lane_idx, _ = self._closest_lane_index()
        lane = self.road.network.get_lane(lane_idx)
        long, lat = lane.local_coordinates(self.vehicle.position)
        w = lane.width_at(long)
        lat_error = lat / (0.5 * w)
        dist_goal = self._network_distance_to_goal(self.vehicle.position, lane_idx)
        # turn-friendly heading target: next waypoint, not lane heading
        wp = self._next_waypoint()
        desired_heading = np.arctan2((wp - self.vehicle.position)[1], (wp - self.vehicle.position)[0])
        heading_error = wrap_to_pi(self.vehicle.heading - desired_heading)

        # ---- turn boost -----------------------------------------------
        remaining = max(lane.length - long, 0.0)
        turn_boost = 1.0 + 2.0 * np.exp(-(remaining / 12.0) ** 2)
        # ---- potential-based shaping ----------------------------------
        phi_now = cfg["alpha"] * dist_goal + cfg["beta"] * abs(lat_error) + (cfg["gamma"] * turn_boost) * abs(heading_error)
        r_shape = self._phi_prev - phi_now
        self._phi_prev = phi_now
        # ---- off road handling ----------------------------------------
        r_off = self.off_road_update()
        # ---- safety / comfort / sparse terms --------------------------
        r_safety = cfg["ttc_penalty_coef"] * max(0, 3 - self._front_ttc())
        r_comfort = (cfg["jerk_cost"] * abs(self._last_jerk) +
                     cfg["steer_smoothness_coef"] * abs(action[1] - self._last_steer))
        self._last_steer = action[1]
        r_terminal = cfg["goal_reward"] if self._goal_reached() else 0.0
        r_terminal += cfg["crash_penalty"] if self.vehicle.crashed else 0.0
        if self.off_road_hard:
            self._ep_stats["off_road_hard_event"] += 1
        if self.off_road_soft:
            self._ep_stats["off_road_soft_penalty"] += r_off
        # ---- step cost
        r_step = cfg["step_cost"]
        # ---- aggregate -----------------------------------------------
        r = r_shape - r_safety - r_comfort + r_terminal + r_off + r_step
        self._ep_stats["shape"] += r_shape
        self._ep_stats["ttc"] += -r_safety
        self._ep_stats["jerk"] += cfg["jerk_cost"] * abs(self._last_jerk)
        self._ep_stats["step_cost"] += r_step
        self._ep_stats["crash"] += cfg["crash_penalty"] if self.vehicle.crashed else 0.0
        self._ep_stats["goal"] += cfg["goal_reward"] if self._goal_reached() else 0.0
        self._ep_stats["total_reward"] += r
        self._ep_stats["dist"] = float(dist_goal)
        self._ep_stats["dist_min"] = float(min(self._ep_stats["dist_min"], dist_goal))
        return float(r)

    def _info(self, obs, action):
        if self._is_terminated() or self._is_truncated():
            return self._ep_stats.copy()
        return {}

    def _goal_reached(self) -> bool:
        return np.linalg.norm(self.vehicle.position - self.goal_position) < 5.0

    def _is_terminated(self):
        # going off-road terminates the episode
        if self.vehicle.crashed or self._goal_reached() or self.off_road_hard:
            return True
        return False

    def _is_truncated(self):
        return self.steps >= self.config["duration"]

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
# Gymnasium registration                                                      #
# --------------------------------------------------------------------------- #
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
