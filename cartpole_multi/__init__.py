"""Configurable multi-pendulum cartpole environment."""

from cartpole_multi.config import NUM_PENDULUMS
from cartpole_multi.env import MultiPendulumCartPoleEnv
from cartpole_multi.policy import CartPolePolicy
from cartpole_multi.torch_env import TorchMultiPendulumCartPoleEnv

__all__ = [
    "CartPolePolicy",
    "MultiPendulumCartPoleEnv",
    "NUM_PENDULUMS",
    "TorchMultiPendulumCartPoleEnv",
]
