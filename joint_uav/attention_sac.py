from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
from typing import Any

import torch
import torch.nn.functional as F

from .models.attention_actor import AttentionActor
from .models.attention_critic import TwinAttentionCritic


@dataclass
class SACUpdateInfo:
    critic_loss: float
    actor_loss: float
    alpha_loss: float
    alpha: float
    q_mean: float
    target_q_mean: float
    active_count_mean: float


class AttentionSACAgent:
    """One SAC policy shared by every non-empty serving-UAV set."""

    def __init__(
        self,
        *,
        global_dim: int,
        uav_obs_dim: int,
        action_dim: int,
        move_limit: float,
        model_cfg: dict[str, Any],
        sac_cfg: dict[str, Any],
        device: torch.device | str = "cpu",
    ) -> None:
        self.device = torch.device(device)
        self.action_dim = int(action_dim)
        self.gamma = float(sac_cfg.get("gamma", 0.99))
        self.tau = float(sac_cfg.get("tau", 0.005))
        self.reward_scale = float(sac_cfg.get("reward_scale", 0.01))
        self.max_grad_norm = float(sac_cfg.get("max_grad_norm", 1.0))
        self.auto_alpha = bool(sac_cfg.get("auto_alpha", True))

        net_args = dict(
            global_dim=int(global_dim),
            uav_obs_dim=int(uav_obs_dim),
            action_dim=self.action_dim,
            hidden_dim=int(model_cfg.get("hidden_dim", 128)),
            n_heads=int(model_cfg.get("n_heads", 4)),
            n_layers=int(model_cfg.get("n_layers", 2)),
            move_limit=float(move_limit),
        )
        self.actor = AttentionActor(**net_args).to(self.device)
        self.critic = TwinAttentionCritic(**net_args).to(self.device)
        self.target_critic = TwinAttentionCritic(**net_args).to(self.device)
        self.target_critic.load_state_dict(self.critic.state_dict())
        self.target_critic.requires_grad_(False)

        self.actor_opt = torch.optim.Adam(
            self.actor.parameters(), lr=float(sac_cfg.get("actor_lr", 3e-5))
        )
        self.critic_opt = torch.optim.Adam(
            self.critic.parameters(), lr=float(sac_cfg.get("critic_lr", 1e-4))
        )

        init_alpha = float(sac_cfg.get("init_alpha", sac_cfg.get("alpha", 0.2)))
        self.log_alpha = torch.tensor(
            float(torch.log(torch.tensor(init_alpha))), device=self.device, requires_grad=True
        )
        self.alpha_opt = torch.optim.Adam(
            [self.log_alpha], lr=float(sac_cfg.get("alpha_lr", 3e-5))
        )

    @property
    def alpha(self) -> torch.Tensor:
        return self.log_alpha.exp()

    @torch.no_grad()
    def select_action(
        self, obs: dict[str, Any], deterministic: bool = False
    ) -> tuple[torch.Tensor, torch.Tensor]:
        uav_obs = torch.as_tensor(
            obs["active_uav_obs"], dtype=torch.float32, device=self.device
        )
        ids = torch.as_tensor(obs["active_uav_ids"], dtype=torch.long, device=self.device)
        if uav_obs.shape[0] == 0:
            return torch.zeros((0, self.action_dim), device=self.device), ids

        global_obs = torch.as_tensor(
            obs["global_obs"], dtype=torch.float32, device=self.device
        )
        mask = torch.ones(uav_obs.shape[0], dtype=torch.bool, device=self.device)
        action, _, mean_action = self.actor(
            global_obs, uav_obs, mask, deterministic=deterministic
        )
        return (mean_action if deterministic else action).squeeze(0), ids

    def update(self, batch: dict[str, torch.Tensor]) -> SACUpdateInfo:
        global_obs = batch["global_obs"]
        uav_obs = batch["active_uav_obs"]
        actions = batch["active_actions"]
        mask = batch["mask"]
        reward = batch["reward"] * self.reward_scale
        done = batch["done"]
        next_global_obs = batch["next_global_obs"]
        next_uav_obs = batch["next_active_uav_obs"]
        next_mask = batch["next_mask"]

        with torch.no_grad():
            next_actions, next_logp, _ = self.actor(
                next_global_obs, next_uav_obs, next_mask, deterministic=False
            )
            next_logp_sum = next_logp.sum(dim=1, keepdim=True)
            tq1, tq2 = self.target_critic(
                next_global_obs, next_uav_obs, next_actions, next_mask
            )
            target_q = torch.minimum(tq1, tq2) - self.alpha.detach() * next_logp_sum
            backup = reward + (1.0 - done) * self.gamma * target_q
            backup = torch.nan_to_num(backup, nan=0.0, posinf=1e6, neginf=-1e6)

        q1, q2 = self.critic(global_obs, uav_obs, actions, mask)
        critic_loss = F.smooth_l1_loss(q1, backup) + F.smooth_l1_loss(q2, backup)
        self.critic_opt.zero_grad(set_to_none=True)
        critic_loss.backward()
        torch.nn.utils.clip_grad_norm_(self.critic.parameters(), self.max_grad_norm)
        self.critic_opt.step()

        new_actions, logp, _ = self.actor(global_obs, uav_obs, mask, deterministic=False)
        logp_sum = logp.sum(dim=1, keepdim=True)
        q1_pi, q2_pi = self.critic(global_obs, uav_obs, new_actions, mask)
        q_pi = torch.minimum(q1_pi, q2_pi)
        actor_loss = (self.alpha.detach() * logp_sum - q_pi).mean()
        self.actor_opt.zero_grad(set_to_none=True)
        actor_loss.backward()
        torch.nn.utils.clip_grad_norm_(self.actor.parameters(), self.max_grad_norm)
        self.actor_opt.step()

        counts = mask.sum(dim=1, keepdim=True).float()
        target_entropy = -float(self.action_dim) * counts
        if self.auto_alpha:
            alpha_loss = -(self.log_alpha * (logp_sum + target_entropy).detach()).mean()
            self.alpha_opt.zero_grad(set_to_none=True)
            alpha_loss.backward()
            self.alpha_opt.step()
        else:
            alpha_loss = torch.zeros((), device=self.device)

        self.soft_update_target()
        return SACUpdateInfo(
            critic_loss=float(critic_loss.item()),
            actor_loss=float(actor_loss.item()),
            alpha_loss=float(alpha_loss.item()),
            alpha=float(self.alpha.detach().item()),
            q_mean=float(torch.minimum(q1, q2).mean().item()),
            target_q_mean=float(backup.mean().item()),
            active_count_mean=float(counts.mean().item()),
        )

    @torch.no_grad()
    def soft_update_target(self) -> None:
        for source, target in zip(self.critic.parameters(), self.target_critic.parameters()):
            target.mul_(1.0 - self.tau).add_(self.tau * source)

    def set_learning_rates(self, *, actor: float, critic: float, alpha: float) -> None:
        self.actor_opt.param_groups[0]["lr"] = float(actor)
        self.critic_opt.param_groups[0]["lr"] = float(critic)
        self.alpha_opt.param_groups[0]["lr"] = float(alpha)

    def load_pretrained(self, path: str | Path, *, load_critics: bool = True) -> None:
        payload = torch.load(path, map_location=self.device, weights_only=False)
        self.actor.load_state_dict(payload["actor"])
        if load_critics and "critic" in payload:
            self.critic.load_state_dict(payload["critic"])
        if load_critics and "target_critic" in payload:
            self.target_critic.load_state_dict(payload["target_critic"])
        elif load_critics:
            self.target_critic.load_state_dict(self.critic.state_dict())

    def state_dict(self) -> dict[str, Any]:
        return {
            "actor": self.actor.state_dict(),
            "critic": self.critic.state_dict(),
            "target_critic": self.target_critic.state_dict(),
            "actor_opt": self.actor_opt.state_dict(),
            "critic_opt": self.critic_opt.state_dict(),
            "log_alpha": float(self.log_alpha.detach().item()),
            "alpha_opt": self.alpha_opt.state_dict(),
        }
