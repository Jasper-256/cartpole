from __future__ import annotations

import torch

from cartpole_multi.env import CartPoleParams


def observation_dim(num_pendulums: int) -> int:
    return 6 + 9 * int(num_pendulums)


def policy_observation_from_state(
    state: torch.Tensor,
    num_pendulums: int,
    params: CartPoleParams,
) -> torch.Tensor:
    n = int(num_pendulums)
    x = state[:, 0:1] / params.x_threshold
    x_dot = state[:, 1:2] / 5.0
    theta = state[:, 2 : 2 + n]
    theta_dot = state[:, 2 + n :]
    sin_theta = torch.sin(theta)
    cos_theta = torch.cos(theta)
    normalized_theta_dot = theta_dot / 10.0
    target_energy = 2.0 * params.gravity * params.pole_length
    link_energy = (
        0.5 * (params.pole_length * theta_dot).square()
        + params.gravity * params.pole_length * (cos_theta + 1.0)
    )
    energy_error = (link_energy - target_energy) / target_energy
    energy_pump = energy_error * normalized_theta_dot * cos_theta
    upright_gate = torch.exp(-0.5 * torch.mean((theta / 0.7).square(), dim=1, keepdim=True))
    swing_gate = 1.0 - upright_gate
    return torch.cat(
        (
            swing_gate,
            upright_gate,
            x * swing_gate,
            x_dot * swing_gate,
            x * upright_gate,
            x_dot * upright_gate,
            swing_gate * sin_theta,
            swing_gate * cos_theta,
            swing_gate * normalized_theta_dot,
            swing_gate * normalized_theta_dot * sin_theta,
            swing_gate * normalized_theta_dot * cos_theta,
            swing_gate * energy_error,
            swing_gate * energy_pump,
            upright_gate * sin_theta,
            upright_gate * normalized_theta_dot,
        ),
        dim=1,
    )
