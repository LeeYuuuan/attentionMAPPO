from __future__ import annotations

from dataclasses import dataclass

import numpy as np
import torch


@dataclass
class Transition:
    global_obs: np.ndarray
    active_uav_obs: np.ndarray
    active_actions: np.ndarray
    reward: float
    next_global_obs: np.ndarray
    next_active_uav_obs: np.ndarray
    done: bool


class BalancedVariableReplayBuffer:
    """Variable-length replay with optional active-count-balanced sampling."""

    def __init__(self, capacity: int, max_uavs: int, seed: int = 0) -> None:
        self.capacity = int(capacity)
        self.max_uavs = int(max_uavs)
        self.rng = np.random.default_rng(seed)
        self.storage: list[Transition] = []
        self.pos = 0

    def __len__(self) -> int:
        return len(self.storage)

    def clear(self) -> None:
        self.storage.clear()
        self.pos = 0

    def add(
        self,
        obs: dict[str, np.ndarray],
        active_actions: np.ndarray,
        reward: float,
        next_obs: dict[str, np.ndarray],
        done: bool,
    ) -> None:
        # N=0 still advances the world, but there is no low-level decision to learn.
        if len(obs["active_uav_ids"]) == 0:
            return
        tr = Transition(
            global_obs=np.asarray(obs["global_obs"], np.float32).copy(),
            active_uav_obs=np.asarray(obs["active_uav_obs"], np.float32).copy(),
            active_actions=np.asarray(active_actions, np.float32).copy(),
            reward=float(reward),
            next_global_obs=np.asarray(next_obs["global_obs"], np.float32).copy(),
            next_active_uav_obs=np.asarray(next_obs["active_uav_obs"], np.float32).copy(),
            done=bool(done),
        )
        if len(self.storage) < self.capacity:
            self.storage.append(tr)
        else:
            self.storage[self.pos] = tr
            self.pos = (self.pos + 1) % self.capacity

    def sample(
        self, batch_size: int, device: torch.device | str, balanced_fraction: float = 0.5
    ) -> dict[str, torch.Tensor]:
        if len(self.storage) < batch_size:
            raise ValueError(f"buffer has {len(self.storage)} items; need {batch_size}")

        all_idx = np.arange(len(self.storage))
        balanced_n = int(round(batch_size * float(np.clip(balanced_fraction, 0.0, 1.0))))
        natural_n = batch_size - balanced_n
        selected: list[int] = []

        groups: dict[int, list[int]] = {}
        for i, tr in enumerate(self.storage):
            groups.setdefault(tr.active_uav_obs.shape[0], []).append(i)
        counts = sorted(groups)
        for j in range(balanced_n):
            n = counts[j % len(counts)]
            selected.append(int(self.rng.choice(groups[n])))
        if natural_n:
            selected.extend(self.rng.choice(all_idx, size=natural_n, replace=True).tolist())
        self.rng.shuffle(selected)
        return collate([self.storage[i] for i in selected], device)


def collate(batch: list[Transition], device: torch.device | str) -> dict[str, torch.Tensor]:
    bsz = len(batch)
    max_n = max(x.active_uav_obs.shape[0] for x in batch)
    next_max_n = max(x.next_active_uav_obs.shape[0] for x in batch)
    # Block boundaries are terminal for low SAC, so an empty next set is safe.
    next_max_n = max(1, next_max_n)
    gdim = batch[0].global_obs.shape[0]
    udim = batch[0].active_uav_obs.shape[1]
    adim = batch[0].active_actions.shape[1]

    g = np.zeros((bsz, gdim), np.float32)
    u = np.zeros((bsz, max_n, udim), np.float32)
    a = np.zeros((bsz, max_n, adim), np.float32)
    m = np.zeros((bsz, max_n), bool)
    ng = np.zeros((bsz, gdim), np.float32)
    nu = np.zeros((bsz, next_max_n, udim), np.float32)
    nm = np.zeros((bsz, next_max_n), bool)
    r = np.zeros((bsz, 1), np.float32)
    d = np.zeros((bsz, 1), np.float32)

    for i, x in enumerate(batch):
        n, nn = x.active_uav_obs.shape[0], x.next_active_uav_obs.shape[0]
        g[i], u[i, :n], a[i, :n], m[i, :n] = x.global_obs, x.active_uav_obs, x.active_actions, True
        ng[i], nu[i, :nn], nm[i, :nn] = x.next_global_obs, x.next_active_uav_obs, True
        r[i, 0], d[i, 0] = x.reward, float(x.done)

    tensor = lambda x, dtype: torch.as_tensor(x, dtype=dtype, device=device)
    return {
        "global_obs": tensor(g, torch.float32),
        "active_uav_obs": tensor(u, torch.float32),
        "active_actions": tensor(a, torch.float32),
        "mask": tensor(m, torch.bool),
        "reward": tensor(r, torch.float32),
        "done": tensor(d, torch.float32),
        "next_global_obs": tensor(ng, torch.float32),
        "next_active_uav_obs": tensor(nu, torch.float32),
        "next_mask": tensor(nm, torch.bool),
    }
