from __future__ import annotations

from typing import Optional

import torch
import torch.nn as nn
import torch.nn.functional as F


class AttentionActor(nn.Module):
    """
    Attention actor for variable-number active UAVs.

    Inputs:
        global_obs:     [B, global_dim]
        active_uav_obs: [B, N, uav_obs_dim]
        mask:           [B, N], True for valid active UAV tokens

    Outputs:
        action:          [B, N, action_dim], scaled to [-move_limit, move_limit]
        log_prob:        [B, N], per-UAV log probability, or None in deterministic mode
        mean_action:     [B, N, action_dim], deterministic tanh(mean) action
    """

    def __init__(
        self,
        global_dim: int,
        uav_obs_dim: int,
        action_dim: int = 2,
        hidden_dim: int = 128,
        n_heads: int = 4,
        n_layers: int = 2,
        move_limit: float = 50.0,
        log_std_min: float = -5.0,
        log_std_max: float = 2.0,
    ):
        super().__init__()

        self.global_dim = int(global_dim)
        self.uav_obs_dim = int(uav_obs_dim)
        self.action_dim = int(action_dim)
        self.hidden_dim = int(hidden_dim)
        self.move_limit = float(move_limit)
        self.log_std_min = float(log_std_min)
        self.log_std_max = float(log_std_max)

        self.global_proj = nn.Sequential(
            nn.Linear(self.global_dim, self.hidden_dim),
            nn.ReLU(),
            nn.Linear(self.hidden_dim, self.hidden_dim),
        )

        self.token_proj = nn.Sequential(
            nn.Linear(self.uav_obs_dim + self.hidden_dim, self.hidden_dim),
            nn.ReLU(),
            nn.Linear(self.hidden_dim, self.hidden_dim),
        )

        enc_layer = nn.TransformerEncoderLayer(
            d_model=self.hidden_dim,
            nhead=n_heads,
            dim_feedforward=4 * self.hidden_dim,
            dropout=0.0,
            activation="relu",
            batch_first=True,
            norm_first=False,
        )
        self.encoder = nn.TransformerEncoder(enc_layer, num_layers=n_layers)

        self.mean_head = nn.Linear(self.hidden_dim, self.action_dim)
        self.log_std_head = nn.Linear(self.hidden_dim, self.action_dim)

    def forward(
        self,
        global_obs: torch.Tensor,
        active_uav_obs: torch.Tensor,
        mask: Optional[torch.Tensor] = None,
        deterministic: bool = False,
    ) -> tuple[torch.Tensor, Optional[torch.Tensor], torch.Tensor]:
        global_obs, active_uav_obs, mask = self._format_inputs(global_obs, active_uav_obs, mask)

        batch_size, n_tokens, _ = active_uav_obs.shape

        global_feat = self.global_proj(global_obs)
        global_tokens = global_feat[:, None, :].expand(batch_size, n_tokens, self.hidden_dim)

        x = torch.cat([active_uav_obs, global_tokens], dim=-1)
        x = self.token_proj(x)

        key_padding_mask = ~mask
        h = self.encoder(x, src_key_padding_mask=key_padding_mask)

        mean = self.mean_head(h)
        log_std = self.log_std_head(h).clamp(self.log_std_min, self.log_std_max)
        std = torch.exp(log_std)

        mean_action = torch.tanh(mean) * self.move_limit

        if deterministic:
            action = mean_action
            log_prob = None
        else:
            normal = torch.distributions.Normal(mean, std)
            z = normal.rsample()
            tanh_z = torch.tanh(z)
            action = tanh_z * self.move_limit

            log_prob = normal.log_prob(z) - torch.log(1.0 - tanh_z.pow(2) + 1e-6)
            log_prob = log_prob.sum(dim=-1)
            log_prob = log_prob.masked_fill(~mask, 0.0)

        action = action.masked_fill(~mask[:, :, None], 0.0)
        mean_action = mean_action.masked_fill(~mask[:, :, None], 0.0)

        return action, log_prob, mean_action

    @staticmethod
    def _format_inputs(
        global_obs: torch.Tensor,
        active_uav_obs: torch.Tensor,
        mask: Optional[torch.Tensor],
    ) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        if global_obs.ndim == 1:
            global_obs = global_obs[None, :]
        if active_uav_obs.ndim == 2:
            active_uav_obs = active_uav_obs[None, :, :]

        if global_obs.ndim != 2:
            raise ValueError(f"global_obs must have shape [B, G], got {global_obs.shape}")
        if active_uav_obs.ndim != 3:
            raise ValueError(f"active_uav_obs must have shape [B, N, D], got {active_uav_obs.shape}")

        if mask is None:
            mask = torch.ones(
                active_uav_obs.shape[:2],
                dtype=torch.bool,
                device=active_uav_obs.device,
            )
        else:
            if mask.ndim == 1:
                mask = mask[None, :]
            mask = mask.to(device=active_uav_obs.device, dtype=torch.bool)

        return global_obs.float(), active_uav_obs.float(), mask
