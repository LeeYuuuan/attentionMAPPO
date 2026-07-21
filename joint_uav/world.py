from __future__ import annotations

from dataclasses import dataclass, field
from enum import IntEnum
from typing import Any, Mapping

import numpy as np


class UAVStatus(IntEnum):
    SERVING = 0
    WAITING = 1
    CHARGING = 2
    DEAD = 3


def _section(cfg: Mapping[str, Any], name: str) -> Mapping[str, Any]:
    value = cfg.get(name)
    if not isinstance(value, Mapping):
        raise KeyError(f"Missing config section: {name}")
    return value


@dataclass
class World:
    """Array-based world state for sensors, UAVs, packet arrivals, and movement."""

    world_size: float = 400.0
    airship_pos: np.ndarray = field(default_factory=lambda: np.array([200.0, 200.0], dtype=np.float32))

    map_rng: np.random.Generator = field(default_factory=np.random.default_rng)
    dyn_rng: np.random.Generator = field(default_factory=np.random.default_rng)

    user_pos: np.ndarray = field(default_factory=lambda: np.zeros((0, 2), dtype=np.float32))
    user_lam: np.ndarray = field(default_factory=lambda: np.zeros((0,), dtype=np.float32))
    user_pkts: np.ndarray = field(default_factory=lambda: np.zeros((0,), dtype=np.float32))
    user_last_visit: np.ndarray = field(default_factory=lambda: np.zeros((0,), dtype=np.float32))

    uav_pos: np.ndarray = field(default_factory=lambda: np.zeros((0, 2), dtype=np.float32))
    uav_prev_pos: np.ndarray = field(default_factory=lambda: np.zeros((0, 2), dtype=np.float32))
    uav_status: np.ndarray = field(default_factory=lambda: np.zeros((0,), dtype=np.int8))
    uav_battery: np.ndarray = field(default_factory=lambda: np.ones((0,), dtype=np.float32))

    now: float = 0.0

    _fixed_user_pos: np.ndarray | None = None
    _packet_lam: float = 0.1
    _cold_start_dt: float = 60.0
    _init_battery: float = 1.0

    def build_map_once(self, cfg: Mapping[str, Any]) -> None:
        self._validate_cfg(cfg)

        seeds = _section(cfg, "seeds")
        world_cfg = _section(cfg, "world")
        users_cfg = _section(cfg, "users")
        uav_cfg = _section(cfg, "uav")

        self.world_size = float(world_cfg["size"])
        self.airship_pos = np.asarray(world_cfg["airship_pos"], dtype=np.float32)

        self.map_rng = np.random.default_rng(int(seeds["map_seed"]))
        self.dyn_rng = np.random.default_rng(int(seeds["world_dyn_seed"]))

        self._packet_lam = float(users_cfg["lam"])
        self._cold_start_dt = float(users_cfg.get("cold_start_dt", 60.0))
        self._init_battery = float(uav_cfg.get("init_battery", 1.0))

        self.user_pos = self._generate_user_positions(cfg)
        self._fixed_user_pos = self.user_pos.copy()

        num_users = self.user_pos.shape[0]
        self.user_lam = np.full(num_users, self._packet_lam, dtype=np.float32)
        self.user_pkts = np.zeros(num_users, dtype=np.float32)
        self.user_last_visit = np.zeros(num_users, dtype=np.float32)

        max_uavs = int(uav_cfg["max_uavs"])
        self.uav_pos = np.repeat(self.airship_pos[None, :], max_uavs, axis=0).astype(np.float32)
        self.uav_prev_pos = self.uav_pos.copy()
        self.uav_status = np.full(max_uavs, int(UAVStatus.WAITING), dtype=np.int8)
        self.uav_battery = np.full(max_uavs, self._init_battery, dtype=np.float32)

        self.reset_episode(reset_users=True, reset_uavs=True)

    def reset_episode(self, *, reset_users: bool = True, reset_uavs: bool = True) -> None:
        self.now = 0.0

        if self._fixed_user_pos is not None:
            self.user_pos = self._fixed_user_pos.copy()

        if reset_users and self.user_pkts.size > 0:
            self.user_pkts[:] = 0.0
            self.user_last_visit[:] = 0.0
            self.user_lam[:] = self._packet_lam

        if reset_uavs and self.uav_pos.size > 0:
            self.uav_pos[:] = self.airship_pos[None, :]
            self.uav_prev_pos[:] = self.airship_pos[None, :]
            self.uav_status[:] = int(UAVStatus.WAITING)
            self.uav_battery[:] = self._init_battery

        if self._cold_start_dt > 0.0:
            self.update_packets(self._cold_start_dt)

    def reset_dyn_rng(self, seed: int) -> None:
        self.dyn_rng = np.random.default_rng(int(seed))

    @property
    def max_uavs(self) -> int:
        return int(self.uav_pos.shape[0])

    def serving_ids(self) -> np.ndarray:
        return np.flatnonzero(self.uav_status == int(UAVStatus.SERVING)).astype(np.int64)

    def waiting_ids(self) -> np.ndarray:
        return np.flatnonzero(self.uav_status == int(UAVStatus.WAITING)).astype(np.int64)

    def charging_ids(self) -> np.ndarray:
        return np.flatnonzero(self.uav_status == int(UAVStatus.CHARGING)).astype(np.int64)

    def dead_ids(self) -> np.ndarray:
        return np.flatnonzero(self.uav_status == int(UAVStatus.DEAD)).astype(np.int64)

    def set_uav_status(self, status: np.ndarray) -> None:
        status = np.asarray(status, dtype=np.int8)
        if status.shape != (self.max_uavs,):
            raise ValueError(f"status must have shape ({self.max_uavs},), got {status.shape}")

        valid = {int(UAVStatus.SERVING), int(UAVStatus.WAITING), int(UAVStatus.CHARGING), int(UAVStatus.DEAD)}
        if not set(status.tolist()).issubset(valid):
            raise ValueError(f"unknown UAV status in {status.tolist()}")

        self.uav_status[:] = status

        back_to_airship = np.flatnonzero(
            (self.uav_status != int(UAVStatus.SERVING)) & (self.uav_status != int(UAVStatus.DEAD))
        )
        if back_to_airship.size > 0:
            self.uav_prev_pos[back_to_airship] = self.uav_pos[back_to_airship]
            self.uav_pos[back_to_airship] = self.airship_pos[None, :]

    def update_packets(self, dt: float) -> None:
        dt = float(dt)
        if dt <= 0.0:
            return

        if self.user_pkts.size > 0:
            increments = self.dyn_rng.poisson(self.user_lam * dt).astype(np.float32)
            self.user_pkts += increments
            self.user_last_visit += dt

        self.now += dt

    def move_serving_uavs(self, actions: np.ndarray) -> dict[str, np.ndarray | int | float]:
        actions = np.asarray(actions, dtype=np.float32)
        if actions.shape != (self.max_uavs, 2):
            raise ValueError(f"actions must have shape ({self.max_uavs}, 2), got {actions.shape}")

        ids = self.serving_ids()
        if ids.size == 0:
            return self._empty_move_info()

        self.uav_prev_pos[ids] = self.uav_pos[ids]
        proposed = self.uav_pos[ids] + actions[ids]
        clipped = self.clip_positions(proposed.copy())

        overflow = proposed - clipped
        oob_mask = (np.abs(overflow) > 1e-6).any(axis=1)
        overflow_dist = float(np.linalg.norm(overflow[oob_mask], axis=1).sum()) if oob_mask.any() else 0.0

        self.uav_pos[ids] = clipped

        return {
            "ids": ids,
            "proposed": proposed.astype(np.float32, copy=False),
            "clipped": clipped.astype(np.float32, copy=False),
            "oob_mask": oob_mask,
            "oob_count": int(oob_mask.sum()),
            "overflow_dist": overflow_dist,
        }

    def apply_full_clear_service(self, owner_user_to_uav: np.ndarray, pkts_before: np.ndarray) -> np.ndarray:
        owner = np.asarray(owner_user_to_uav, dtype=np.int64)
        pkts_before = np.asarray(pkts_before, dtype=np.float32)

        collected = np.zeros((self.max_uavs, self.user_pkts.size), dtype=np.float32)
        covered = owner >= 0
        if not covered.any():
            return collected

        user_ids = np.flatnonzero(covered)
        uav_ids = owner[user_ids]
        collected[uav_ids, user_ids] = pkts_before[user_ids]

        self.user_pkts[user_ids] = 0.0
        self.user_last_visit[user_ids] = 0.0
        return collected

    def apply_rate_service(
        self,
        owner_user_to_uav: np.ndarray,
        pkts_before: np.ndarray,
        service_amount_per_user: float,
        *,
        reset_last_visit_on_cover: bool = True,
    ) -> np.ndarray:
        owner = np.asarray(owner_user_to_uav, dtype=np.int64)
        pkts_before = np.asarray(pkts_before, dtype=np.float32)
        amount = float(service_amount_per_user)

        collected = np.zeros((self.max_uavs, self.user_pkts.size), dtype=np.float32)
        if amount <= 0.0:
            return collected

        covered = owner >= 0
        if not covered.any():
            return collected

        user_ids = np.flatnonzero(covered)
        uav_ids = owner[user_ids]
        served = np.minimum(pkts_before[user_ids], amount)

        collected[uav_ids, user_ids] = served
        self.user_pkts[user_ids] -= served

        if reset_last_visit_on_cover:
            self.user_last_visit[user_ids] = 0.0

        return collected

    def clip_positions(self, xy: np.ndarray) -> np.ndarray:
        xy = np.asarray(xy, dtype=np.float32)
        np.clip(xy[:, 0], 0.0, self.world_size, out=xy[:, 0])
        np.clip(xy[:, 1], 0.0, self.world_size, out=xy[:, 1])
        return xy

    def users_pos(self) -> np.ndarray:
        return self.user_pos.astype(np.float32, copy=False)

    def uavs_pos(self) -> np.ndarray:
        return self.uav_pos.astype(np.float32, copy=False)

    def pkts_num_array(self) -> np.ndarray:
        return self.user_pkts.astype(np.float32, copy=False)

    def last_visit_array(self) -> np.ndarray:
        return self.user_last_visit.astype(np.float32, copy=False)

    def lam_array(self) -> np.ndarray:
        return self.user_lam.astype(np.float32, copy=False)

    @staticmethod
    def _validate_cfg(cfg: Mapping[str, Any]) -> None:
        seeds = _section(cfg, "seeds")
        world = _section(cfg, "world")
        users = _section(cfg, "users")
        uav = _section(cfg, "uav")

        for key in ("map_seed", "world_dyn_seed"):
            if key not in seeds:
                raise KeyError(f"Missing seeds.{key}")

        if float(world["size"]) <= 0.0:
            raise ValueError("world.size must be positive")
        if "airship_pos" not in world:
            raise KeyError("Missing world.airship_pos")

        if int(users["num_users"]) < 0:
            raise ValueError("users.num_users must be non-negative")
        if int(users["n_clusters"]) <= 0:
            raise ValueError("users.n_clusters must be positive")
        if not 0.0 <= float(users["cluster_ratio"]) <= 1.0:
            raise ValueError("users.cluster_ratio must be in [0, 1]")
        if float(users["cluster_std"]) < 0.0:
            raise ValueError("users.cluster_std must be non-negative")
        if float(users["lam"]) < 0.0:
            raise ValueError("users.lam must be non-negative")

        if int(uav["max_uavs"]) <= 0:
            raise ValueError("uav.max_uavs must be positive")
        if float(uav.get("init_battery", 1.0)) < 0.0:
            raise ValueError("uav.init_battery must be non-negative")

    def _generate_user_positions(self, cfg: Mapping[str, Any]) -> np.ndarray:
        users = _section(cfg, "users")
        world = _section(cfg, "world")

        num_users = int(users["num_users"])
        world_size = float(world["size"])

        if num_users <= 0:
            return np.zeros((0, 2), dtype=np.float32)

        mode = str(users.get("distribution_mode", "fixed_cluster_uniform")).lower()
        if mode not in {"fixed_cluster_uniform", "cluster_uniform"}:
            raise ValueError(f"Unsupported users.distribution_mode: {mode}")

        n_clustered = int(round(float(users["cluster_ratio"]) * num_users))
        n_clustered = max(0, min(num_users, n_clustered))
        n_uniform = num_users - n_clustered

        points: list[np.ndarray] = []

        if n_clustered > 0:
            n_clusters = max(1, int(users["n_clusters"]))
            cluster_std = float(users["cluster_std"])

            centers = np.column_stack(
                [
                    self.map_rng.uniform(50.0, world_size - 50.0, n_clusters),
                    self.map_rng.uniform(50.0, world_size - 50.0, n_clusters),
                ]
            ).astype(np.float32)

            counts = np.full(n_clusters, n_clustered // n_clusters, dtype=int)
            counts[: n_clustered % n_clusters] += 1

            for cluster_id, count in enumerate(counts):
                count = int(count)
                if count <= 0:
                    continue
                cx, cy = centers[cluster_id]
                x = self.map_rng.normal(cx, cluster_std, count)
                y = self.map_rng.normal(cy, cluster_std, count)
                points.append(np.column_stack([x, y]))

        if n_uniform > 0:
            x = self.map_rng.uniform(0.0, world_size, n_uniform)
            y = self.map_rng.uniform(0.0, world_size, n_uniform)
            points.append(np.column_stack([x, y]))

        pos = np.vstack(points).astype(np.float32) if points else np.zeros((0, 2), dtype=np.float32)
        pos[:, 0] = np.clip(pos[:, 0], 0.0, world_size)
        pos[:, 1] = np.clip(pos[:, 1], 0.0, world_size)
        return pos

    @staticmethod
    def _empty_move_info() -> dict[str, np.ndarray | int | float]:
        empty_pos = np.zeros((0, 2), dtype=np.float32)
        return {
            "ids": np.zeros((0,), dtype=np.int64),
            "proposed": empty_pos,
            "clipped": empty_pos,
            "oob_mask": np.zeros((0,), dtype=bool),
            "oob_count": 0,
            "overflow_dist": 0.0,
        }
