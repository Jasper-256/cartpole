from __future__ import annotations

import torch
from torch import nn

import pufferlib.pytorch


class CartPolePolicy(nn.Module):
    """PufferLib-style actor/critic policy for cart force control."""

    is_continuous = False

    def __init__(
        self,
        observation_size: int,
        action_size: int = 3,
        hidden_size: int = 128,
    ) -> None:
        super().__init__()
        self.hidden_size = int(hidden_size)
        self.encoder = nn.Sequential(
            pufferlib.pytorch.layer_init(nn.Linear(observation_size, hidden_size)),
            nn.GELU(),
            pufferlib.pytorch.layer_init(nn.Linear(hidden_size, hidden_size)),
            nn.GELU(),
        )
        self.actor = pufferlib.pytorch.layer_init(
            nn.Linear(hidden_size, action_size),
            std=0.01,
        )
        self.value_fn = pufferlib.pytorch.layer_init(nn.Linear(hidden_size, 1), std=1.0)

    def encode_observations(
        self,
        observations: torch.Tensor,
        state: dict | None = None,
    ) -> torch.Tensor:
        del state
        return self.encoder(observations.float())

    def decode_actions(self, hidden: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor]:
        return self.actor(hidden), self.value_fn(hidden)

    def forward_eval(
        self,
        observations: torch.Tensor,
        state: dict | None = None,
    ) -> tuple[torch.Tensor, torch.Tensor]:
        hidden = self.encode_observations(observations, state)
        return self.decode_actions(hidden)

    def forward(
        self,
        observations: torch.Tensor,
        state: dict | None = None,
    ) -> tuple[torch.Tensor, torch.Tensor]:
        return self.forward_eval(observations, state)
