from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
from typing import Any

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F


class RunningMeanStd:
    def __init__(self, epsilon: float = 1e-4) -> None:
        self.mean = 0.0
        self.var = 1.0
        self.count = float(epsilon)

    def update(self, x: np.ndarray) -> None:
        x = np.asarray(x, dtype=np.float64)
        if x.size == 0:
            return
        batch_mean, batch_var, batch_count = float(x.mean()), float(x.var()), x.size
        delta = batch_mean - self.mean
        total = self.count + batch_count
        self.mean += delta * batch_count / total
        m2 = self.var * self.count + batch_var * batch_count
        m2 += delta * delta * self.count * batch_count / total
        self.var = m2 / total
        self.count = total

    @property
    def std(self) -> float:
        return float(np.sqrt(self.var + 1e-8))

    def state_dict(self) -> dict[str, float]:
        return {"mean": self.mean, "var": self.var, "count": self.count}

    def load_state_dict(self, state: dict[str, float]) -> None:
        self.mean, self.var, self.count = state["mean"], state["var"], state["count"]


class SharedCategoricalActor(nn.Module):
    def __init__(self, obs_dim: int, hidden: int = 256) -> None:
        super().__init__()
        self.net = nn.Sequential(
            nn.Linear(obs_dim, hidden), nn.Tanh(),
            nn.Linear(hidden, hidden), nn.Tanh(),
            nn.Linear(hidden, 2),
        )

    def dist(self, obs: torch.Tensor) -> torch.distributions.Categorical:
        return torch.distributions.Categorical(logits=self.net(obs))


class CentralizedCritic(nn.Module):
    def __init__(self, obs_dim: int, hidden: int = 256) -> None:
        super().__init__()
        self.net = nn.Sequential(
            nn.Linear(obs_dim, hidden), nn.Tanh(),
            nn.Linear(hidden, hidden), nn.Tanh(),
            nn.Linear(hidden, 1),
        )

    def forward(self, obs: torch.Tensor) -> torch.Tensor:
        return self.net(obs).squeeze(-1)


@dataclass
class MAPPOConfig:
    gamma: float = 0.99
    gae_lambda: float = 0.95
    actor_lr: float = 1e-4
    critic_lr: float = 1e-4
    clip_range: float = 0.10
    value_clip_range: float = 0.10
    entropy_start: float = 0.01
    entropy_end: float = 0.001
    epochs: int = 5
    minibatches: int = 1
    max_grad_norm: float = 0.5
    target_kl: float = 0.015
    hidden_dim: int = 256
    value_normalization: bool = True

    @classmethod
    def from_dict(cls, x: dict[str, Any]) -> "MAPPOConfig":
        keys = cls.__dataclass_fields__.keys()
        return cls(**{k: x[k] for k in keys if k in x})


class EpisodeRollout:
    def __init__(self) -> None:
        self.actor_obs: list[np.ndarray] = []
        self.critic_obs: list[np.ndarray] = []
        self.actions: list[np.ndarray] = []
        self.logp: list[np.ndarray] = []
        self.agent_active: list[np.ndarray] = []
        self.rewards: list[float] = []
        self.dones: list[bool] = []
        self.values: list[float] = []

    def add(
        self, actor_obs: np.ndarray, critic_obs: np.ndarray, actions: np.ndarray,
        logp: np.ndarray, reward: float, done: bool, value: float,
        agent_active: np.ndarray | None = None,
    ) -> None:
        self.actor_obs.append(np.asarray(actor_obs, np.float32).copy())
        self.critic_obs.append(np.asarray(critic_obs, np.float32).copy())
        self.actions.append(np.asarray(actions, np.int64).copy())
        self.logp.append(np.asarray(logp, np.float32).copy())
        if agent_active is None:
            agent_active = np.ones(np.asarray(actions).shape, dtype=bool)
        self.agent_active.append(np.asarray(agent_active, bool).copy())
        self.rewards.append(float(reward))
        self.dones.append(bool(done))
        self.values.append(float(value))

    def finish(
        self, *, last_value: float, gamma: float, gae_lambda: float
    ) -> dict[str, np.ndarray]:
        rewards = np.asarray(self.rewards, np.float32)
        dones = np.asarray(self.dones, np.float32)
        values = np.asarray(self.values, np.float32)
        adv = np.zeros_like(rewards)
        gae = 0.0
        for t in reversed(range(len(rewards))):
            next_value = last_value if t == len(rewards) - 1 else values[t + 1]
            nonterminal = 1.0 - dones[t]
            delta = rewards[t] + gamma * nonterminal * next_value - values[t]
            gae = delta + gamma * gae_lambda * nonterminal * gae
            adv[t] = gae
        return {
            "actor_obs": np.stack(self.actor_obs),
            "critic_obs": np.stack(self.critic_obs),
            "actions": np.stack(self.actions),
            "logp": np.stack(self.logp),
            "agent_active": np.stack(self.agent_active),
            "values": values,
            "advantages": adv,
            "returns": adv + values,
        }


class MAPPO:
    """Parameter-shared MAPPO actor with one centralized team-value critic."""

    def __init__(
        self, actor_obs_dim: int, critic_obs_dim: int, n_agents: int,
        cfg: MAPPOConfig, device: torch.device | str
    ) -> None:
        self.cfg, self.n_agents, self.device = cfg, int(n_agents), torch.device(device)
        self.actor = SharedCategoricalActor(actor_obs_dim, cfg.hidden_dim).to(self.device)
        self.critic = CentralizedCritic(critic_obs_dim, cfg.hidden_dim).to(self.device)
        self.actor_opt = torch.optim.Adam(self.actor.parameters(), lr=cfg.actor_lr, eps=1e-5)
        self.critic_opt = torch.optim.Adam(self.critic.parameters(), lr=cfg.critic_lr, eps=1e-5)
        self.value_rms = RunningMeanStd()

    def _normalize_value(self, x: torch.Tensor) -> torch.Tensor:
        if not self.cfg.value_normalization:
            return x
        return (x - self.value_rms.mean) / self.value_rms.std

    def _denormalize_value(self, x: torch.Tensor) -> torch.Tensor:
        if not self.cfg.value_normalization:
            return x
        return x * self.value_rms.std + self.value_rms.mean

    @torch.no_grad()
    def act(
        self, actor_obs: np.ndarray, critic_obs: np.ndarray, deterministic: bool = False
    ) -> tuple[np.ndarray, np.ndarray, float]:
        ao = torch.as_tensor(actor_obs, dtype=torch.float32, device=self.device)
        co = torch.as_tensor(critic_obs, dtype=torch.float32, device=self.device).unsqueeze(0)
        dist = self.actor.dist(ao)
        actions = dist.probs.argmax(-1) if deterministic else dist.sample()
        logp = dist.log_prob(actions)
        value_raw = self._denormalize_value(self.critic(co)).item()
        return (
            actions.cpu().numpy().astype(np.int64),
            logp.cpu().numpy().astype(np.float32),
            float(value_raw),
        )

    @torch.no_grad()
    def value(self, critic_obs: np.ndarray) -> float:
        co = torch.as_tensor(critic_obs, dtype=torch.float32, device=self.device).unsqueeze(0)
        return float(self._denormalize_value(self.critic(co)).item())

    def update(
        self, episodes: list[dict[str, np.ndarray]], *, progress: float
    ) -> dict[str, float]:
        data = {k: np.concatenate([ep[k] for ep in episodes], axis=0) for k in episodes[0]}
        returns_np = data["returns"]
        self.value_rms.update(returns_np)

        # Actor samples are per-agent; critic samples are per joint timestep.
        t, n = data["actions"].shape
        actor_obs = torch.as_tensor(data["actor_obs"].reshape(t * n, -1), dtype=torch.float32, device=self.device)
        actions = torch.as_tensor(data["actions"].reshape(-1), dtype=torch.long, device=self.device)
        old_logp = torch.as_tensor(data["logp"].reshape(-1), dtype=torch.float32, device=self.device)
        adv = torch.as_tensor(data["advantages"], dtype=torch.float32, device=self.device)
        adv = (adv - adv.mean()) / (adv.std(unbiased=False) + 1e-8)
        adv_actor = adv[:, None].expand(-1, n).reshape(-1)

        critic_obs = torch.as_tensor(data["critic_obs"], dtype=torch.float32, device=self.device)
        old_values = torch.as_tensor(data["values"], dtype=torch.float32, device=self.device)
        returns = torch.as_tensor(returns_np, dtype=torch.float32, device=self.device)
        old_values_n = self._normalize_value(old_values)
        returns_n = self._normalize_value(returns)

        entropy_coef = self.cfg.entropy_start + np.clip(progress, 0.0, 1.0) * (
            self.cfg.entropy_end - self.cfg.entropy_start
        )
        lr_scale = 1.0 - 0.9 * np.clip(progress, 0.0, 1.0)
        self.actor_opt.param_groups[0]["lr"] = self.cfg.actor_lr * lr_scale
        self.critic_opt.param_groups[0]["lr"] = self.cfg.critic_lr * lr_scale

        actor_idx = np.flatnonzero(data["agent_active"].reshape(-1))
        critic_idx = np.arange(t)
        actor_chunks = max(1, int(self.cfg.minibatches))
        stats = {k: [] for k in ("actor_loss", "critic_loss", "entropy", "kl", "clipfrac")}
        stop_actor = False

        for _ in range(self.cfg.epochs):
            np.random.shuffle(actor_idx)
            np.random.shuffle(critic_idx)
            for mb in np.array_split(actor_idx, actor_chunks):
                if stop_actor or mb.size == 0:
                    continue
                j = torch.as_tensor(mb, dtype=torch.long, device=self.device)
                dist = self.actor.dist(actor_obs[j])
                new_logp = dist.log_prob(actions[j])
                log_ratio = new_logp - old_logp[j]
                ratio = log_ratio.exp()
                s1 = ratio * adv_actor[j]
                s2 = ratio.clamp(1.0 - self.cfg.clip_range, 1.0 + self.cfg.clip_range) * adv_actor[j]
                entropy = dist.entropy().mean()
                actor_loss = -torch.minimum(s1, s2).mean() - entropy_coef * entropy
                self.actor_opt.zero_grad(set_to_none=True)
                actor_loss.backward()
                nn.utils.clip_grad_norm_(self.actor.parameters(), self.cfg.max_grad_norm)
                self.actor_opt.step()
                with torch.no_grad():
                    approx_kl = ((ratio - 1.0) - log_ratio).mean()
                    clipfrac = ((ratio - 1.0).abs() > self.cfg.clip_range).float().mean()
                stats["actor_loss"].append(float(actor_loss.item()))
                stats["entropy"].append(float(entropy.item()))
                stats["kl"].append(float(approx_kl.item()))
                stats["clipfrac"].append(float(clipfrac.item()))
                if approx_kl.item() > self.cfg.target_kl:
                    stop_actor = True

            for mb in np.array_split(critic_idx, actor_chunks):
                if mb.size == 0:
                    continue
                j = torch.as_tensor(mb, dtype=torch.long, device=self.device)
                v = self.critic(critic_obs[j])
                v_clip = old_values_n[j] + (v - old_values_n[j]).clamp(
                    -self.cfg.value_clip_range, self.cfg.value_clip_range
                )
                loss_unclip = F.smooth_l1_loss(v, returns_n[j], reduction="none")
                loss_clip = F.smooth_l1_loss(v_clip, returns_n[j], reduction="none")
                critic_loss = torch.maximum(loss_unclip, loss_clip).mean()
                self.critic_opt.zero_grad(set_to_none=True)
                critic_loss.backward()
                nn.utils.clip_grad_norm_(self.critic.parameters(), self.cfg.max_grad_norm)
                self.critic_opt.step()
                stats["critic_loss"].append(float(critic_loss.item()))

        out = {k: float(np.mean(v)) if v else 0.0 for k, v in stats.items()}
        out.update(entropy_coef=float(entropy_coef), samples=float(t), early_stop=float(stop_actor))
        return out

    def state_dict(self) -> dict[str, Any]:
        return {
            "actor": self.actor.state_dict(), "critic": self.critic.state_dict(),
            "actor_opt": self.actor_opt.state_dict(), "critic_opt": self.critic_opt.state_dict(),
            "value_rms": self.value_rms.state_dict(), "cfg": self.cfg.__dict__,
        }

    def save(self, path: str | Path, *, extra: dict[str, Any] | None = None) -> None:
        payload = {"mappo": self.state_dict(), **(extra or {})}
        torch.save(payload, path)
