from __future__ import annotations

from dataclasses import dataclass
from typing import Any, Mapping

import numpy as np

try:
    from .world import UAVStatus, World
except ImportError:
    from world import UAVStatus, World


def _section(cfg: Mapping[str, Any], name: str) -> Mapping[str, Any]:
    value = cfg.get(name)
    if not isinstance(value, Mapping):
        raise KeyError(f"Missing config section: {name}")
    return value


@dataclass
class StepMetrics:
    owner_user_to_uav: np.ndarray
    covered_any: np.ndarray
    pkts_before_collect: np.ndarray
    pkts_after_collect: np.ndarray
    collected_by_uav: np.ndarray
    per_uav_collected_sum: np.ndarray
    per_uav_collected_max: np.ndarray
    team_max_sum: float
    max_before: float
    max_after: float


class VariableUAVEnv:
    def __init__(self, cfg: Mapping[str, Any]):
        self.cfg = cfg
        self.world = World()
        self.world.build_map_once(cfg)

        self.max_uavs = self.world.max_uavs
        self.num_users = self.world.user_pos.shape[0]

        time_cfg = _section(cfg, "time")
        self.high_steps = int(time_cfg["high_steps"])
        self.low_steps_per_high = int(time_cfg["low_steps_per_high"])
        self.move_sec = float(time_cfg["move_sec"])
        self.collect_sec = float(time_cfg["collect_sec"])
        self.total_low_steps = self.high_steps * self.low_steps_per_high

        uav_cfg = _section(cfg, "uav")
        self.move_limit = float(uav_cfg["move_limit"])
        self.coverage_radius = float(uav_cfg["coverage_radius"])

        self.low_step = 0
        self.high_step = 0

        self.traj: list[list[np.ndarray]] = []
        self.status_history: list[np.ndarray] = []
        self.hist_max_pkt: list[float] = []
        self.hist_cov_max_pkt: list[float] = []
        self.hist_reward: list[float] = []

    def reset(
        self,
        *,
        initial_status: np.ndarray | None = None,
        reset_dyn_rng: bool = False,
    ) -> dict[str, np.ndarray]:
        if reset_dyn_rng:
            seed = int(_section(self.cfg, "seeds")["world_dyn_seed"])
            self.world.reset_dyn_rng(seed)

        self.world.reset_episode(reset_users=True, reset_uavs=True)

        if initial_status is None:
            initial_status = self._default_initial_status()
        self.world.set_uav_status(initial_status)

        self.low_step = 0
        self.high_step = 0

        self.traj = [[self.world.uav_pos[i].copy()] for i in range(self.max_uavs)]
        self.status_history = [self.world.uav_status.copy()]
        self.hist_max_pkt = []
        self.hist_cov_max_pkt = []
        self.hist_reward = []

        return self._build_obs()

    def step(
        self,
        actions: np.ndarray,
        target_status: np.ndarray | None = None,
    ) -> tuple[dict[str, np.ndarray], float, bool, dict[str, Any]]:
        if target_status is not None:
            self.world.set_uav_status(target_status)

        actions = self._clip_actions_box(actions)
        move_info = self.world.move_serving_uavs(actions)

        self.world.update_packets(self.move_sec)
        self.world.update_packets(self.collect_sec)

        metrics = self._collect_data()
        reward = self._compute_reward(metrics, move_info)

        self.low_step += 1
        self.high_step = self.low_step // self.low_steps_per_high

        self._record_step(metrics, reward)

        done = self.low_step >= self.total_low_steps
        obs = self._build_obs()
        info = self._build_info(metrics, move_info, reward)

        return obs, float(reward), bool(done), info

    @property
    def is_high_boundary(self) -> bool:
        return self.low_step % self.low_steps_per_high == 0

    def set_uav_status(self, status: np.ndarray) -> dict[str, np.ndarray]:
        self.world.set_uav_status(status)
        return self._build_obs()

    def _default_initial_status(self) -> np.ndarray:
        env_cfg = self.cfg.get("env", {})
        ids = env_cfg.get("initial_serving_ids", [0, 1])
        status = np.full(self.max_uavs, int(UAVStatus.WAITING), dtype=np.int8)
        status[np.asarray(ids, dtype=np.int64)] = int(UAVStatus.SERVING)
        return status

    def _clip_actions_box(self, actions: np.ndarray) -> np.ndarray:
        actions = np.asarray(actions, dtype=np.float32)
        if actions.shape != (self.max_uavs, 2):
            raise ValueError(f"actions must have shape ({self.max_uavs}, 2), got {actions.shape}")
        return np.clip(actions, -self.move_limit, self.move_limit).astype(np.float32)

    def _collect_data(self) -> StepMetrics:
        owner = self._assign_users_to_serving_uavs()
        covered_any = owner >= 0
        pkts_before = self.world.user_pkts.copy()

        service_cfg = _section(self.cfg, "service")
        mode = str(service_cfg["mode"]).lower()

        if mode == "full_clear":
            collected = self.world.apply_full_clear_service(owner, pkts_before)
        elif mode == "rate":
            collected = self.world.apply_rate_service(
                owner,
                pkts_before,
                service_amount_per_user=float(service_cfg["service_amount_per_user"]),
                reset_last_visit_on_cover=bool(service_cfg.get("reset_last_visit_on_cover", True)),
            )
        else:
            raise ValueError(f"Unsupported service.mode: {mode}")

        pkts_after = self.world.user_pkts.copy()
        per_uav_sum = collected.sum(axis=1).astype(np.float32)
        per_uav_max = (
            collected.max(axis=1).astype(np.float32)
            if collected.size
            else np.zeros(self.max_uavs, dtype=np.float32)
        )

        return StepMetrics(
            owner_user_to_uav=owner,
            covered_any=covered_any,
            pkts_before_collect=pkts_before,
            pkts_after_collect=pkts_after,
            collected_by_uav=collected,
            per_uav_collected_sum=per_uav_sum,
            per_uav_collected_max=per_uav_max,
            team_max_sum=float(per_uav_max.sum()),
            max_before=float(pkts_before.max()) if pkts_before.size else 0.0,
            max_after=float(pkts_after.max()) if pkts_after.size else 0.0,
        )

    def _assign_users_to_serving_uavs(self) -> np.ndarray:
        users = self.world.user_pos
        serving_ids = self.world.serving_ids()
        owner = np.full(users.shape[0], -1, dtype=np.int64)

        if users.shape[0] == 0 or serving_ids.size == 0:
            return owner

        uav_pos = self.world.uav_pos[serving_ids]
        diff = users[:, None, :] - uav_pos[None, :, :]
        dist2 = np.sum(diff * diff, axis=-1)

        covered = dist2 <= self.coverage_radius ** 2
        covered_any = covered.any(axis=1)

        if not covered_any.any():
            return owner

        dist2_masked = np.where(covered, dist2, np.inf)
        nearest_local = dist2_masked.argmin(axis=1)
        owner[covered_any] = serving_ids[nearest_local[covered_any]]
        return owner

    def _compute_reward(
        self,
        metrics: StepMetrics,
        move_info: dict[str, np.ndarray | int | float],
    ) -> float:
        reward_cfg = _section(self.cfg, "reward")

        collect_weight = float(reward_cfg.get("collect_weight", 1.0))
        backlog_weight = float(reward_cfg.get("backlog_weight", 1.0))
        oob_penalty = float(reward_cfg.get("oob_penalty", 500.0))
        oob_penalty_scale = float(reward_cfg.get("oob_penalty_scale", 5.0))

        oob_count = int(move_info["oob_count"])
        overflow_dist = float(move_info["overflow_dist"])

        return (
            collect_weight * metrics.team_max_sum
            - backlog_weight * metrics.max_after
            - oob_penalty * oob_count
            - oob_penalty_scale * overflow_dist
        )

    def _build_obs(self) -> dict[str, np.ndarray]:
        obs_cfg = self.cfg.get("obs", {})

        last_visit = self.world.user_last_visit.astype(np.float32, copy=True)
        if bool(obs_cfg.get("normalize_last_visit", False)):
            last_visit = last_visit / float(obs_cfg.get("last_visit_norm", 1000.0))

        serving_ids = self.world.serving_ids()
        pos = self.world.uav_pos[serving_ids].astype(np.float32, copy=True)

        if bool(obs_cfg.get("normalize_positions", False)):
            pos = pos / float(self.world.world_size)

        if bool(obs_cfg.get("include_uav_id", True)):
            denom = max(1, self.max_uavs - 1)
            id_col = (serving_ids.astype(np.float32) / float(denom))[:, None]
            active_uav_obs = np.concatenate([pos, id_col], axis=1).astype(np.float32)
        else:
            active_uav_obs = pos.astype(np.float32)

        return {
            "global_obs": last_visit.astype(np.float32),
            "active_uav_obs": active_uav_obs.astype(np.float32),
            "active_uav_ids": serving_ids.astype(np.int64),
            "uav_status": self.world.uav_status.copy(),
        }

    def _record_step(self, metrics: StepMetrics, reward: float) -> None:
        for i in range(self.max_uavs):
            self.traj[i].append(self.world.uav_pos[i].copy())

        self.status_history.append(self.world.uav_status.copy())
        self.hist_max_pkt.append(float(metrics.max_after))
        self.hist_cov_max_pkt.append(
            float(metrics.per_uav_collected_max.max()) if metrics.per_uav_collected_max.size else 0.0
        )
        self.hist_reward.append(float(reward))

    def _build_info(
        self,
        metrics: StepMetrics,
        move_info: dict[str, np.ndarray | int | float],
        reward: float,
    ) -> dict[str, Any]:
        return {
            "time": float(self.world.now),
            "low_step": int(self.low_step),
            "high_step": int(self.high_step),
            "serving_ids": self.world.serving_ids().copy(),
            "uav_status": self.world.uav_status.copy(),
            "covered_count": int(metrics.covered_any.sum()),
            "owner_user_to_uav": metrics.owner_user_to_uav.copy(),
            "pkts_before_collect": metrics.pkts_before_collect.copy(),
            "pkts_after_collect": metrics.pkts_after_collect.copy(),
            "collected_by_uav": metrics.collected_by_uav.copy(),
            "per_uav_collected_sum": metrics.per_uav_collected_sum.copy(),
            "per_uav_collected_max": metrics.per_uav_collected_max.copy(),
            "team_max_sum": float(metrics.team_max_sum),
            "max_before": float(metrics.max_before),
            "max_after": float(metrics.max_after),
            "oob_count": int(move_info["oob_count"]),
            "oob_overflow_dist": float(move_info["overflow_dist"]),
            "reward": float(reward),
        }
