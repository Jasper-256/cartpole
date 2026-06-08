"""Configurable multi-pendulum cartpole environment."""

from cartpole_multi.config import NUM_PENDULUMS
from cartpole_multi.env import MultiPendulumCartPoleEnv
from cartpole_multi.observations import observation_dim, policy_observation_from_state
from cartpole_multi.policy import CartPolePolicy

__all__ = [
    "CartPolePolicy",
    "MultiPendulumCartPoleEnv",
    "NUM_PENDULUMS",
    "observation_dim",
    "policy_observation_from_state",
]
