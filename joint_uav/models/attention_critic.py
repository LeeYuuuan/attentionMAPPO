from __future__ import annotations

from typing import Optional

import torch
import torch.nn as nn


class AttentionCritic(nn.Module):
    """
    Attention critic for variable-number active UAVs.

    Inputs:
        global_obs:      [B, global_dim]
        active_uav_obs:  [B, N, uav_obs_dim]
        active_actions:  [B, N, action_dim]
        mask:            [B, N]

    Output:
        q:               [B, 1]
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
    ):
        super().__init__()

        self.global_dim = int(global_dim)
        self.uav_obs_dim = int(uav_obs_dim)
        self.action_dim = int(action_dim)
        self.hidden_dim = int(hidden_dim)
        self.move_limit = float(move_limit)

        self.global_proj = nn.Sequential(
            nn.Linear(self.global_dim, self.hidden_dim),
            nn.ReLU(),
            nn.Linear(self.hidden_dim, self.hidden_dim),
        )

        self.token_proj = nn.Sequential(
            nn.Linear(self.uav_obs_dim + self.action_dim + self.hidden_dim, self.hidden_dim),
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

        self.q_head = nn.Sequential(
            nn.Linear(self.hidden_dim + self.hidden_dim, self.hidden_dim),
            nn.ReLU(),
            nn.Linear(self.hidden_dim, self.hidden_dim),
            nn.ReLU(),
            nn.Linear(self.hidden_dim, 1),
        )

    def forward(
        self,
        global_obs: torch.Tensor,
        active_uav_obs: torch.Tensor,
        active_actions: torch.Tensor,
        mask: Optional[torch.Tensor] = None,
    ) -> torch.Tensor:
        global_obs, active_uav_obs, active_actions, mask = self._format_inputs(
            global_obs,
            active_uav_obs,
            active_actions,
            mask,
        )

        batch_size, n_tokens, _ = active_uav_obs.shape

        global_feat = self.global_proj(global_obs)
        global_tokens = global_feat[:, None, :].expand(batch_size, n_tokens, self.hidden_dim)

        action_norm = active_actions / self.move_limit
        x = torch.cat([active_uav_obs, action_norm, global_tokens], dim=-1)
        x = self.token_proj(x)

        key_padding_mask = ~mask
        h = self.encoder(x, src_key_padding_mask=key_padding_mask)

        mask_f = mask[:, :, None].float()
        h_sum = (h * mask_f).sum(dim=1)
        denom = mask_f.sum(dim=1).clamp_min(1.0)
        pooled = h_sum / denom

        q_in = torch.cat([pooled, global_feat], dim=-1)
        return self.q_head(q_in)

    @staticmethod
    def _format_inputs(
        global_obs: torch.Tensor,
        active_uav_obs: torch.Tensor,
        active_actions: torch.Tensor,
        mask: Optional[torch.Tensor],
    ) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor]:
        if global_obs.ndim == 1:
            global_obs = global_obs[None, :]
        if active_uav_obs.ndim == 2:
            active_uav_obs = active_uav_obs[None, :, :]
        if active_actions.ndim == 2:
            active_actions = active_actions[None, :, :]

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

        return global_obs.float(), active_uav_obs.float(), active_actions.float(), mask


class TwinAttentionCritic(nn.Module):
    """Twin Q networks for SAC."""

    def __init__(self, **kwargs):
        super().__init__()
        self.q1 = AttentionCritic(**kwargs)
        self.q2 = AttentionCritic(**kwargs)

    def forward(
        self,
        global_obs: torch.Tensor,
        active_uav_obs: torch.Tensor,
        active_actions: torch.Tensor,
        mask: Optional[torch.Tensor] = None,
    ) -> tuple[torch.Tensor, torch.Tensor]:
        q1 = self.q1(global_obs, active_uav_obs, active_actions, mask)
        q2 = self.q2(global_obs, active_uav_obs, active_actions, mask)
        return q1, q2
