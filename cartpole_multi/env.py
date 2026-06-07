from __future__ import annotations

from dataclasses import dataclass

import gymnasium as gym
import numpy as np
from gymnasium import spaces

from cartpole_multi.config import NUM_PENDULUMS


@dataclass(frozen=True)
class CartPoleParams:
    gravity: float = 9.8
    cart_mass: float = 1.0
    pole_mass: float = 0.08
    pole_length: float = 0.7
    force_mag: float = 14.0
    dt: float = 0.02
    cart_friction: float = 0.08
    pole_friction: float = 0.015
    x_threshold: float = 2.4
    theta_threshold: float = 1.25
    stable_x_threshold: float = 0.5
    stable_theta_threshold: float = np.deg2rad(12.0)
    stable_theta_dot_threshold: float = 1.0
    max_episode_steps: int = 500


class MultiPendulumCartPoleEnv(gym.Env):
    """A lightweight cartpole variant with N inverted pendulums on one cart.

    This is intentionally simple rather than a full rigid-body chain solver.
    Each pendulum is modeled as an inverted pendulum on the same accelerating
    cart, with aggregate horizontal reaction terms feeding back into cart
    acceleration. The flat observation and discrete action spaces make it easy
    to wrap in PufferLib's Gymnasium emulation layer.
    """

    metadata = {"render_modes": []}

    def __init__(
        self,
        num_pendulums: int = NUM_PENDULUMS,
        params: CartPoleParams | None = None,
        seed: int | None = None,
    ) -> None:
        super().__init__()
        if num_pendulums < 1:
            raise ValueError("num_pendulums must be at least 1")

        self.num_pendulums = int(num_pendulums)
        self.params = params or CartPoleParams()
        self.action_space = spaces.Discrete(3)

        obs_dim = 2 + 2 * self.num_pendulums
        high = np.full(obs_dim, np.inf, dtype=np.float32)
        self.observation_space = spaces.Box(-high, high, dtype=np.float32)

        self.state = np.zeros(obs_dim, dtype=np.float32)
        self.step_count = 0
        self._rng = np.random.default_rng(seed)

    def reset(
        self,
        *,
        seed: int | None = None,
        options: dict | None = None,
    ) -> tuple[np.ndarray, dict]:
        super().reset(seed=seed)
        if seed is not None:
            self._rng = np.random.default_rng(seed)

        spread = 0.08
        self.state[0] = self._rng.uniform(-0.03, 0.03)
        self.state[1] = self._rng.uniform(-0.03, 0.03)
        self.state[2 : 2 + self.num_pendulums] = self._rng.uniform(
            -spread, spread, size=self.num_pendulums
        )
        self.state[2 + self.num_pendulums :] = self._rng.uniform(
            -0.02, 0.02, size=self.num_pendulums
        )
        self.step_count = 0
        return self._get_obs(), {}

    def step(self, action: int) -> tuple[np.ndarray, float, bool, bool, dict]:
        action = int(action)
        if action < 0 or action >= self.action_space.n:
            raise ValueError(f"invalid action {action}")

        p = self.params
        force = (action - 1) * p.force_mag
        x = float(self.state[0])
        x_dot = float(self.state[1])
        theta = self.state[2 : 2 + self.num_pendulums].astype(np.float64)
        theta_dot = self.state[2 + self.num_pendulums :].astype(np.float64)

        sin_theta = np.sin(theta)
        cos_theta = np.cos(theta)
        pole_mass = p.pole_mass
        total_mass = p.cart_mass + self.num_pendulums * pole_mass

        centrifugal = np.sum(pole_mass * p.pole_length * theta_dot**2 * sin_theta)
        gravity_reaction = np.sum(pole_mass * p.gravity * sin_theta * cos_theta)
        x_acc = (force + centrifugal - gravity_reaction - p.cart_friction * x_dot) / total_mass

        theta_acc = (
            (p.gravity * sin_theta - x_acc * cos_theta) / p.pole_length
            - p.pole_friction * theta_dot
        )

        x_dot += p.dt * x_acc
        x += p.dt * x_dot
        theta_dot = theta_dot + p.dt * theta_acc
        theta = theta + p.dt * theta_dot
        theta = (theta + np.pi) % (2 * np.pi) - np.pi

        self.state[0] = x
        self.state[1] = x_dot
        self.state[2 : 2 + self.num_pendulums] = theta.astype(np.float32)
        self.state[2 + self.num_pendulums :] = theta_dot.astype(np.float32)
        self.step_count += 1

        terminated = bool(
            abs(x) > p.x_threshold
            or np.any(np.abs(theta) > p.theta_threshold)
            or not np.all(np.isfinite(self.state))
        )
        truncated = self.step_count >= p.max_episode_steps
        reward = self._reward(force, terminated)

        info = {
            "x": x,
            "upright": float(np.mean(np.cos(theta))),
            "stable": self._is_stable(),
            "num_pendulums": self.num_pendulums,
        }
        return self._get_obs(), reward, terminated, truncated, info

    def _reward(self, force: float, terminated: bool) -> float:
        p = self.params
        x = float(self.state[0])
        x_dot = float(self.state[1])
        theta = self.state[2 : 2 + self.num_pendulums]
        theta_dot = self.state[2 + self.num_pendulums :]

        upright = float(np.mean(np.cos(theta)))
        centered = 1.0 - min((x / p.x_threshold) ** 2, 1.0)
        velocity_cost = 0.01 * x_dot**2 + 0.002 * float(np.mean(theta_dot**2))
        action_cost = 0.0005 * (force / p.force_mag) ** 2
        reward = 1.0 + 0.5 * upright + 0.2 * centered - velocity_cost - action_cost
        if terminated:
            reward -= 2.0
        return float(reward)

    def _get_obs(self) -> np.ndarray:
        return self.state.astype(np.float32, copy=True)

    def _is_stable(self) -> bool:
        p = self.params
        x = float(self.state[0])
        theta = self.state[2 : 2 + self.num_pendulums]
        theta_dot = self.state[2 + self.num_pendulums :]
        return bool(
            abs(x) <= p.stable_x_threshold
            and np.all(np.abs(theta) <= p.stable_theta_threshold)
            and np.all(np.abs(theta_dot) <= p.stable_theta_dot_threshold)
        )
