from __future__ import annotations

import torch
from torch import nn


class CartPolePolicy(nn.Module):
    """Linear force actor used by the native Metal ES trainer."""

    is_continuous = True

    def __init__(
        self,
        observation_size: int,
    ) -> None:
        super().__init__()
        self.observation_size = int(observation_size)
        self.actor = nn.Linear(self.observation_size, 1)
        self.reset_parameters()

    def reset_parameters(self) -> None:
        nn.init.normal_(self.actor.weight, mean=0.0, std=0.01)
        nn.init.zeros_(self.actor.bias)

    def forward(
        self,
        observations: torch.Tensor,
    ) -> torch.Tensor:
        return self.actor(observations)
