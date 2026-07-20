from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
from typing import Any

import numpy as np

from .attention_sac import AttentionSACAgent
from .low_env import VariableUAVEnv
from .replay import BalancedVariableReplayBuffer
from .world import UAVStatus


SERVING = int(UAVStatus.SERVING)
WAITING = int(UAVStatus.WAITING)
CHARGING = int(UAVStatus.CHARGING)
DEAD = int(UAVStatus.DEAD)


class ChargingUAV:
    def __init__(self, battery_frac: float) -> None:
        self.battery = float(battery_frac)
        self.status = SERVING
        self.want_leave_next = False
        self.full_pending = False


@dataclass
class EnergyModelConfig:
    P_hover: float = 326.0
    P_horiz: float = 257.0
    v_horiz: float = 10.0
    battery_capacity_Wh: float = 125.0
    airship_pos: tuple[float, float] = (200.0, 200.0)
    airship_alt: float = 50.0
    uav_alt: float = 0.0
    P_climb: float | None = None
    return_table_path: str = ""


class EnergyModel:
    def __init__(self, cfg: EnergyModelConfig) -> None:
        self.cfg = cfg
        self.dist: np.ndarray | None = None
        self.frac: np.ndarray | None = None
        if cfg.return_table_path:
            table_path = Path(cfg.return_table_path)
            if not table_path.exists():
                raise FileNotFoundError(f"return-energy table not found: {table_path}")
            data = np.load(table_path, allow_pickle=True)
            if "dist" not in data or "frac" not in data:
                raise ValueError("return-energy table must contain dist and frac")
            order = np.argsort(data["dist"])
            self.dist = np.asarray(data["dist"], np.float32)[order]
            self.frac = np.asarray(data["frac"], np.float32)[order]
            if (
                self.dist.ndim != 1 or self.frac.ndim != 1
                or self.dist.size != self.frac.size or self.dist.size < 2
                or not np.isfinite(self.dist).all() or not np.isfinite(self.frac).all()
            ):
                raise ValueError("invalid dist/frac arrays in return-energy table")

    def block_energy_frac(self, traj_xy: np.ndarray, dt_min: float, block_steps: int) -> float:
        traj = np.asarray(traj_xy, np.float32)
        if len(traj) < 2:
            return 0.0
        dt_h = (float(dt_min) / 60.0) / float(block_steps)
        energy = 0.0
        for t in range(1, len(traj)):
            moving = np.linalg.norm(traj[t] - traj[t - 1]) > 1e-6
            energy += (self.cfg.P_horiz if moving else self.cfg.P_hover) * dt_h
        return float(energy / self.cfg.battery_capacity_Wh)

    def outbound_energy_frac(self, start_pos_xy: np.ndarray) -> float:
        d = float(np.linalg.norm(np.asarray(start_pos_xy) - np.asarray(self.cfg.airship_pos)))
        return float((self.cfg.P_horiz * (d / self.cfg.v_horiz) / 3600.0) / self.cfg.battery_capacity_Wh)

    def return_energy_frac(self, pos_xy: np.ndarray) -> float:
        d_xy = float(np.linalg.norm(np.asarray(pos_xy) - np.asarray(self.cfg.airship_pos)))
        # At horizontal radius zero the supplied table still contains the
        # non-zero vertical flight from UAV altitude to airship altitude.
        if self.dist is not None and self.frac is not None:
            return float(np.interp(d_xy, self.dist, self.frac))
        dh = self.cfg.airship_alt - self.cfg.uav_alt
        distance = float(np.sqrt(d_xy * d_xy + dh * dh))
        power = self.cfg.P_climb if dh > 0 and self.cfg.P_climb is not None else self.cfg.P_horiz
        return float((power * (distance / self.cfg.v_horiz) / 3600.0) / self.cfg.battery_capacity_Wh)


@dataclass
class JointEnvConfig:
    episode_len_steps: int = 100
    timestep_min: float = 10.0
    block_steps: int = 10
    max_slots: int = 2
    max_wait: int = 2
    end_if_any_dead: bool = False
    serving_weight: float = 3.0
    waiting_weight: float = 1.0
    buffer_weight: float = 1.0
    death_penalty: float = 10.0
    buf_ref: float = 800.0
    low_random_steps: int = 5000
    low_random_seed: int = 1043

    @classmethod
    def from_dict(cls, x: dict[str, Any]) -> "JointEnvConfig":
        return cls(**{k: x[k] for k in cls.__dataclass_fields__ if k in x})


class AttentionChargingEnv:
    """
    One continuous packet world + the original charging/FIFO state machine.

    Charging action meanings are intentionally context dependent:
      SERVING: 0 request charge, 1 keep serving
      WAITING: 0 keep waiting, 1 leave and serve
      CHARGING: 0 keep charging, 1 leave and serve
    """

    def __init__(
        self,
        low_env: VariableUAVEnv,
        low_agent: AttentionSACAgent,
        energy_model: EnergyModel,
        cfg: JointEnvConfig,
        replay: BalancedVariableReplayBuffer | None = None,
    ) -> None:
        self.low_env, self.low_agent = low_env, low_agent
        self.energy_model, self.cfg, self.replay = energy_model, cfg, replay
        self.n_uav = low_env.max_uavs
        if cfg.block_steps != low_env.low_steps_per_high:
            raise ValueError("joint.block_steps must equal time.low_steps_per_high")
        if cfg.episode_len_steps != low_env.high_steps:
            raise ValueError("joint.episode_len_steps must equal time.high_steps")
        self.buf_clip = 3.0 * cfg.buf_ref
        self.t = 0
        self.uavs: list[ChargingUAV] = []
        self.wait_queue: list[int] = []
        self.history: dict[str, list] = {}
        self.mission_pos_bank = np.zeros((self.n_uav, 2), np.float32)
        self.return_energy_due = np.zeros(self.n_uav, np.float32)
        self.last_buf_metric = 0.0
        self.low_action_steps = 0
        self.low_action_rng = np.random.default_rng(cfg.low_random_seed)

    def _free_slots(self) -> int:
        return self.cfg.max_slots - sum(u.status == CHARGING for u in self.uavs)

    def _fill_slots_from_queue(self) -> None:
        while self._free_slots() > 0 and self.wait_queue:
            idx = self.wait_queue.pop(0)
            if self.uavs[idx].status == WAITING:
                self.uavs[idx].status = CHARGING

    def _status(self) -> np.ndarray:
        return np.asarray([u.status for u in self.uavs], dtype=np.int8)

    def _battery(self) -> np.ndarray:
        return np.asarray([u.battery for u in self.uavs], dtype=np.float32)

    def reset(self, *, reset_dyn_rng: bool = False) -> dict[str, np.ndarray]:
        self.t, self.wait_queue = 0, []
        self.uavs = [ChargingUAV(1.0) for _ in range(self.n_uav)]
        status = np.full(self.n_uav, SERVING, dtype=np.int8)
        self.low_env.reset(initial_status=status, reset_dyn_rng=reset_dyn_rng)
        air = self.low_env.world.airship_pos.astype(np.float32)
        self.mission_pos_bank[:] = air[None, :]
        self.return_energy_due[:] = 0.0
        self.last_buf_metric = float(self.low_env.world.user_pkts.max()) if self.low_env.num_users else 0.0
        self._sync_world_state(prev_status=status)
        self.history = {
            "battery": [self._battery().copy()], "status": [self._status().copy()],
            "buf_metric": [], "upper_reward": [], "low_reward": [], "serving_count": [],
            "queue_length": [], "newly_dead_count": [],
            "reward_serving": [], "reward_waiting": [], "reward_buffer": [], "reward_death": [],
            "trajectory_blocks": [],
            "low_max_before": [], "low_max_after": [], "low_team_max_sum": [],
            "low_covered_count": [], "low_oob_count": [], "low_serving_count": [],
            "low_per_uav_collected_sum": [], "low_per_uav_collected_max": [],
        }
        return self.get_obs_all()

    def _queue_rank(self, idx: int) -> float:
        if idx not in self.wait_queue or self.cfg.max_wait <= 0:
            return 0.0
        return float(self.wait_queue.index(idx) + 1) / float(self.cfg.max_wait)

    def _backlog_features(self) -> tuple[float, float]:
        pkts = self.low_env.world.user_pkts
        if pkts.size == 0:
            return 0.0, 0.0
        return min(float(pkts.max()), self.buf_clip) / self.cfg.buf_ref, min(float(pkts.mean()), self.buf_clip) / self.cfg.buf_ref

    def _distance_norm(self, idx: int) -> float:
        air = self.low_env.world.airship_pos
        dmax = np.sqrt(2.0) * self.low_env.world.world_size
        return float(np.linalg.norm(self.mission_pos_bank[idx] - air) / max(dmax, 1e-6))

    def _obs_agent(self, idx: int) -> np.ndarray:
        u = self.uavs[idx]
        onehot = np.zeros(4, np.float32); onehot[u.status] = 1.0
        n_chg = sum(x.status == CHARGING for x in self.uavs)
        slots_r = n_chg / max(1, self.cfg.max_slots)
        queue_r = len(self.wait_queue) / max(1, self.cfg.max_wait)
        id_r = idx / max(1, self.n_uav - 1)
        max_b, mean_b = self._backlog_features()
        return np.asarray([
            u.battery, *onehot, slots_r, queue_r, id_r, *self._battery().tolist(),
            self._queue_rank(idx), max_b, mean_b, self._distance_norm(idx),
        ], np.float32)

    def _obs_global(self) -> np.ndarray:
        feat: list[float] = []
        for u in self.uavs:
            onehot = np.zeros(4, np.float32); onehot[u.status] = 1.0
            feat.extend([u.battery, *onehot])
        feat.extend([
            sum(u.status == CHARGING for u in self.uavs) / max(1, self.cfg.max_slots),
            len(self.wait_queue) / max(1, self.cfg.max_wait),
        ])
        feat.extend(self._distance_norm(i) for i in range(self.n_uav))
        feat.extend(self._backlog_features())
        feat.append(sum(u.status == SERVING for u in self.uavs) / float(self.n_uav))
        return np.asarray(feat, np.float32)

    def get_obs_all(self) -> dict[str, np.ndarray]:
        return {
            "actor_obs": np.stack([self._obs_agent(i) for i in range(self.n_uav)]),
            "critic_obs": self._obs_global(),
            "agent_active": self._status() != DEAD,
        }

    def _sync_world_state(self, prev_status: np.ndarray) -> None:
        status = self._status()
        self.low_env.world.set_uav_status(status)
        newly_serving = np.flatnonzero(
            (status == SERVING) & np.isin(prev_status, [WAITING, CHARGING])
        )
        if newly_serving.size:
            self.low_env.world.uav_pos[newly_serving] = self.mission_pos_bank[newly_serving]
            self.low_env.world.uav_prev_pos[newly_serving] = self.mission_pos_bank[newly_serving]
        self.low_env.world.uav_battery[:] = self._battery()

    def _apply_original_charging_transitions(self, actions: np.ndarray) -> np.ndarray:
        actions = np.asarray(actions, dtype=np.int64)
        if actions.shape != (self.n_uav,) or not np.isin(actions, [0, 1]).all():
            raise ValueError(f"upper actions must be binary with shape ({self.n_uav},)")
        prev_status = self._status().copy()
        want_charge, want_leave = actions == 0, actions == 1

        # The following order is intentionally identical to the old environment.
        for i, u in enumerate(self.uavs):
            if u.status == CHARGING and want_leave[i]:
                u.want_leave_next = True
        for u in self.uavs:
            if u.status == CHARGING and (u.full_pending or u.want_leave_next):
                u.status, u.full_pending, u.want_leave_next = SERVING, False, False
        for i, u in enumerate(self.uavs):
            if u.status == WAITING and want_leave[i]:
                u.status = SERVING
        self.wait_queue = [i for i in self.wait_queue if self.uavs[i].status == WAITING]
        self._fill_slots_from_queue()
        free = self._free_slots()
        for i, u in enumerate(self.uavs):
            if u.status != SERVING or not want_charge[i]:
                continue
            if free > 0:
                u.status, free = CHARGING, free - 1
            elif len(self.wait_queue) < self.cfg.max_wait:
                u.status = WAITING
                self.wait_queue.append(i)

        # Preserve the original rule: return energy is paid on entry to CHARGING.
        for i, (u, old) in enumerate(zip(self.uavs, prev_status)):
            if old != CHARGING and u.status == CHARGING:
                u.battery -= float(self.return_energy_due[i])
                self.return_energy_due[i] = 0.0
                if u.battery <= 0.0:
                    u.battery, u.status = 0.0, DEAD
        self._sync_world_state(prev_status)
        return prev_status

    def _run_low_block(
        self, *, deterministic: bool, collect_replay: bool
    ) -> tuple[float, float, dict[int, np.ndarray]]:
        serving = self.low_env.world.serving_ids()
        tracks = {int(i): [self.low_env.world.uav_pos[i].copy()] for i in serving}
        total_reward, max_after = 0.0, []
        zero = np.zeros((self.n_uav, 2), np.float32)

        for k in range(self.cfg.block_steps):
            obs = self.low_env._build_obs()
            if serving.size:
                use_safe_random = (
                    collect_replay and self.low_action_steps < self.cfg.low_random_steps
                )
                if use_safe_random:
                    active_actions = self._safe_random_active_actions(obs)
                    ids = np.asarray(obs["active_uav_ids"], np.int64)
                else:
                    action_t, ids_t = self.low_agent.select_action(obs, deterministic=deterministic)
                    active_actions = action_t.detach().cpu().numpy().astype(np.float32)
                    ids = ids_t.detach().cpu().numpy()
                full_actions = zero.copy()
                full_actions[ids] = active_actions
            else:
                active_actions = np.zeros((0, 2), np.float32)
                full_actions = zero.copy()
            next_obs, reward, env_done, info = self.low_env.step(full_actions)
            if collect_replay:
                self.low_action_steps += 1
            boundary = k == self.cfg.block_steps - 1
            if collect_replay and self.replay is not None and serving.size:
                self.replay.add(obs, active_actions, reward, next_obs, done=boundary or env_done)
            total_reward += float(reward)
            max_after.append(float(info["max_after"]))
            self.history["low_max_before"].append(float(info["max_before"]))
            self.history["low_max_after"].append(float(info["max_after"]))
            self.history["low_team_max_sum"].append(float(info["team_max_sum"]))
            self.history["low_covered_count"].append(int(info["covered_count"]))
            self.history["low_oob_count"].append(int(info["oob_count"]))
            self.history["low_serving_count"].append(int(len(info["serving_ids"])))
            self.history["low_per_uav_collected_sum"].append(
                np.asarray(info["per_uav_collected_sum"], np.float32).copy()
            )
            self.history["low_per_uav_collected_max"].append(
                np.asarray(info["per_uav_collected_max"], np.float32).copy()
            )
            for i in serving:
                tracks[int(i)].append(self.low_env.world.uav_pos[i].copy())

        return total_reward, float(np.mean(max_after)), {
            i: np.asarray(x, np.float32) for i, x in tracks.items()
        }

    def _safe_random_active_actions(self, obs: dict[str, np.ndarray]) -> np.ndarray:
        ids = np.asarray(obs["active_uav_ids"], np.int64)
        actions = np.zeros((len(ids), 2), np.float32)
        for k, uid in enumerate(ids):
            pos = self.low_env.world.uav_pos[int(uid)]
            low = np.maximum(-self.low_env.move_limit, -pos)
            high = np.minimum(
                self.low_env.move_limit, self.low_env.world.world_size - pos
            )
            actions[k] = self.low_action_rng.uniform(low, high).astype(np.float32)
        return actions

    def _calc_reward(
        self, buf_metric: float, newly_dead_count: int
    ) -> tuple[float, float, float, float, float]:
        num_serving = sum(u.status == SERVING for u in self.uavs)
        num_waiting = sum(u.status == WAITING for u in self.uavs)
        serving_term = self.cfg.serving_weight * num_serving / float(self.n_uav)
        waiting_term = self.cfg.waiting_weight * num_waiting / float(self.n_uav)
        buffer_term = -self.cfg.buffer_weight * min(float(buf_metric), self.buf_clip) / self.cfg.buf_ref
        death_term = -self.cfg.death_penalty * int(newly_dead_count)
        reward = serving_term + waiting_term + buffer_term + death_term
        return (
            float(reward), float(serving_term), float(waiting_term),
            float(buffer_term), float(death_term),
        )

    def step(
        self, actions: np.ndarray, *, deterministic_low: bool = False,
        collect_low_replay: bool = True
    ) -> tuple[dict[str, np.ndarray], float, bool, dict[str, Any]]:
        status_at_step_start = self._status().copy()
        prev_status = self._status().copy()
        prev_serving = np.flatnonzero(prev_status == SERVING)
        if prev_serving.size:
            self.mission_pos_bank[prev_serving] = self.low_env.world.uav_pos[prev_serving]
        prev_status = self._apply_original_charging_transitions(actions)

        current_serving = self.low_env.world.serving_ids()
        outbound = np.zeros(self.n_uav, np.float32)
        for i in current_serving:
            if prev_status[i] in (CHARGING, WAITING):
                outbound[i] = self.energy_model.outbound_energy_frac(self.mission_pos_bank[i])

        low_reward, buf_metric, tracks = self._run_low_block(
            deterministic=deterministic_low, collect_replay=collect_low_replay
        )
        self.history["trajectory_blocks"].append(
            {int(i): np.asarray(traj, np.float32).copy() for i, traj in tracks.items()}
        )
        block_energy = np.zeros(self.n_uav, np.float32)
        for i, traj in tracks.items():
            self.mission_pos_bank[i] = traj[-1]
            block_energy[i] = self.energy_model.block_energy_frac(
                traj, self.cfg.timestep_min, self.cfg.block_steps
            )
            self.return_energy_due[i] = self.energy_model.return_energy_frac(traj[-1])

        for i, (u, old) in enumerate(zip(self.uavs, prev_status)):
            if u.status == SERVING and old in (CHARGING, WAITING):
                u.battery -= float(outbound[i])
            if u.status == SERVING:
                u.battery -= float(block_energy[i])
            if u.battery <= 0.0:
                u.battery, u.status = 0.0, DEAD

        n_chg = sum(u.status == CHARGING for u in self.uavs)
        p_chg = 600.0 if n_chg == 1 else (500.0 if n_chg >= 2 else 0.0)
        if p_chg:
            delta = (p_chg * (self.cfg.timestep_min / 60.0)) / self.energy_model.cfg.battery_capacity_Wh
            for u in self.uavs:
                if u.status == CHARGING:
                    u.battery += delta
                    if u.battery >= 1.0:
                        u.battery, u.full_pending = 1.0, True

        self._sync_world_state(prev_status=self._status())
        self.t += 1
        self.last_buf_metric = buf_metric
        status_at_step_end = self._status()
        newly_dead_count = int(
            ((status_at_step_start != DEAD) & (status_at_step_end == DEAD)).sum()
        )
        reward, serving_term, waiting_term, buffer_term, death_term = self._calc_reward(
            buf_metric, newly_dead_count
        )
        done = self.t >= self.cfg.episode_len_steps or (
            self.cfg.end_if_any_dead and any(u.status == DEAD for u in self.uavs)
        )
        self.history["battery"].append(self._battery().copy())
        self.history["status"].append(self._status().copy())
        self.history["buf_metric"].append(buf_metric)
        self.history["upper_reward"].append(reward)
        self.history["low_reward"].append(low_reward)
        self.history["serving_count"].append(sum(u.status == SERVING for u in self.uavs))
        self.history["queue_length"].append(len(self.wait_queue))
        self.history["newly_dead_count"].append(newly_dead_count)
        self.history["reward_serving"].append(serving_term)
        self.history["reward_waiting"].append(waiting_term)
        self.history["reward_buffer"].append(buffer_term)
        self.history["reward_death"].append(death_term)
        return self.get_obs_all(), reward, bool(done), {
            "buf_metric": buf_metric, "low_reward": low_reward,
            "newly_dead_count": newly_dead_count,
            "serving_ids": self.low_env.world.serving_ids().copy(),
            "wait_queue": self.wait_queue.copy(),
        }
