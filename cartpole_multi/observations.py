from __future__ import annotations

import numpy as np


def encoded_observation_dim(num_pendulums: int, mode: str) -> int:
    if mode == "raw":
        return 2 + 2 * num_pendulums
    if mode == "trig":
        return 2 + 3 * num_pendulums
    raise ValueError("observation mode must be 'raw' or 'trig'")


def encode_observations(obs: np.ndarray, num_pendulums: int, mode: str = "trig") -> np.ndarray:
    obs = np.asarray(obs, dtype=np.float32)
    if mode == "raw":
        return obs.astype(np.float32, copy=False)
    if mode != "trig":
        raise ValueError("observation mode must be 'raw' or 'trig'")

    was_vector = obs.ndim == 1
    batch = obs.reshape(1, -1) if was_vector else obs
    theta = batch[:, 2 : 2 + num_pendulums]
    theta_dot = batch[:, 2 + num_pendulums : 2 + 2 * num_pendulums]
    encoded = np.concatenate(
        (
            batch[:, :2],
            np.sin(theta),
            np.cos(theta),
            theta_dot,
        ),
        axis=1,
    ).astype(np.float32, copy=False)
    return encoded[0] if was_vector else encoded
