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
    stable_x_dot_threshold: float = 0.75
    stable_theta_threshold: float = np.deg2rad(12.0)
    stable_theta_dot_threshold: float = 1.0
    max_episode_steps: int = 500
    upright_reward_weight: float = 8.0
    centered_reward_weight: float = 0.10
    energy_reward_weight: float = 0.75
    swingup_progress_reward_weight: float = 50.0
    swingup_motion_reward_weight: float = 0.05
    x_velocity_cost_weight: float = 0.01
    x_position_cost_weight: float = 2.0
    theta_velocity_cost_weight: float = 0.002
    top_theta_velocity_cost_weight: float = 0.20
    action_cost_weight: float = 0.01
    alive_reward: float = 0.0
    stable_bonus: float = 20.0
    termination_penalty: float = 150.0


class MultiPendulumCartPoleEnv(gym.Env):
    """A lightweight cartpole with an N-link pendulum chain on one cart.

    The pendulums are equal-length serial links, so ``num_pendulums=2`` means a
    double pendulum attached end-to-end rather than two rods attached to the
    cart. Angles are absolute link angles measured from upright: ``theta=0`` is
    the stabilization target and ``theta=pi`` is the natural hanging pose.
    """

    metadata = {"render_modes": []}

    def __init__(
        self,
        num_pendulums: int = NUM_PENDULUMS,
        params: CartPoleParams | None = None,
        reset_mode: str = "downward",
        seed: int | None = None,
    ) -> None:
        super().__init__()
        if num_pendulums < 1:
            raise ValueError("num_pendulums must be at least 1")

        self.num_pendulums = int(num_pendulums)
        self.params = params or CartPoleParams()
        self.reset_mode = self._validate_reset_mode(reset_mode)
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

        self.state[0] = self._rng.uniform(-0.03, 0.03)
        self.state[1] = self._sample_reset_x_dot()
        self.state[2 : 2 + self.num_pendulums] = self._sample_reset_theta(1)[0].astype(
            np.float32
        )
        self.state[2 + self.num_pendulums :] = self._sample_reset_theta_dot(1)[0]
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
        previous_theta = self.state[2 : 2 + self.num_pendulums].astype(np.float64)
        theta = previous_theta.copy()
        theta_dot = self.state[2 + self.num_pendulums :].astype(np.float64)

        q_dot = np.concatenate(([x_dot], theta_dot))
        mass_matrix = self._mass_matrix(theta)
        bias = self._bias_forces(theta, q_dot)
        generalized_force = np.zeros(self.num_pendulums + 1, dtype=np.float64)
        generalized_force[0] = force
        q_acc = np.linalg.solve(mass_matrix, generalized_force - bias)

        x_dot += p.dt * q_acc[0]
        x += p.dt * x_dot
        theta_dot = theta_dot + p.dt * q_acc[1:]
        theta = self._wrap_angles(theta + p.dt * theta_dot)

        self.state[0] = x
        self.state[1] = x_dot
        self.state[2 : 2 + self.num_pendulums] = theta.astype(np.float32)
        self.state[2 + self.num_pendulums :] = theta_dot.astype(np.float32)
        self.step_count += 1

        terminated = bool(
            abs(x) > p.x_threshold
            or not np.all(np.isfinite(self.state))
        )
        truncated = self.step_count >= p.max_episode_steps
        reward = self._reward(force, terminated, previous_theta)

        info = {
            "x": x,
            "upright": float(np.mean(np.cos(theta))),
            "downward": float(np.mean(-np.cos(theta))),
            "stable": self.is_stable(),
            "num_pendulums": self.num_pendulums,
        }
        return self._get_obs(), reward, terminated, truncated, info

    def _mass_matrix(self, theta: np.ndarray) -> np.ndarray:
        p = self.params
        n = self.num_pendulums
        size = n + 1
        matrix = np.zeros((size, size), dtype=np.float64)
        matrix[0, 0] = p.cart_mass + n * p.pole_mass

        distal_counts = np.arange(n, 0, -1, dtype=np.float64)
        cart_coupling = p.pole_mass * p.pole_length * distal_counts * np.cos(theta)
        matrix[0, 1:] = cart_coupling
        matrix[1:, 0] = cart_coupling

        for i in range(n):
            for j in range(n):
                distal_count = n - max(i, j)
                matrix[i + 1, j + 1] = (
                    p.pole_mass
                    * p.pole_length**2
                    * distal_count
                    * np.cos(theta[i] - theta[j])
                )
        return matrix

    def _bias_forces(self, theta: np.ndarray, q_dot: np.ndarray) -> np.ndarray:
        p = self.params
        n = self.num_pendulums
        size = n + 1
        d_mass = np.zeros((size, size, size), dtype=np.float64)

        for r in range(n):
            theta_idx = r + 1
            value = -p.pole_mass * p.pole_length * (n - r) * np.sin(theta[r])
            d_mass[theta_idx, 0, theta_idx] = value
            d_mass[theta_idx, theta_idx, 0] = value

        for r in range(n):
            theta_idx = r + 1
            for i in range(n):
                for j in range(n):
                    distal_count = n - max(i, j)
                    value = (
                        -p.pole_mass
                        * p.pole_length**2
                        * distal_count
                        * np.sin(theta[i] - theta[j])
                        * ((1.0 if i == r else 0.0) - (1.0 if j == r else 0.0))
                    )
                    d_mass[theta_idx, i + 1, j + 1] = value

        mass_dot_q_dot = np.einsum("rkj,r,j->k", d_mass, q_dot, q_dot)
        kinetic_gradient = 0.5 * np.einsum("rij,i,j->r", d_mass, q_dot, q_dot)

        potential_gradient = np.zeros(size, dtype=np.float64)
        distal_counts = np.arange(n, 0, -1, dtype=np.float64)
        potential_gradient[1:] = (
            -p.pole_mass * p.gravity * p.pole_length * distal_counts * np.sin(theta)
        )

        damping = np.zeros(size, dtype=np.float64)
        damping[0] = p.cart_friction * q_dot[0]
        damping[1:] = p.pole_friction * q_dot[1:]

        return mass_dot_q_dot - kinetic_gradient + potential_gradient + damping

    def _reward(
        self,
        force: float,
        terminated: bool,
        previous_theta: np.ndarray,
    ) -> float:
        p = self.params
        x = float(self.state[0])
        x_dot = float(self.state[1])
        theta = self.state[2 : 2 + self.num_pendulums]
        theta_dot = self.state[2 + self.num_pendulums :]

        link_height = (np.cos(theta) + 1.0) * 0.5
        height = float(np.mean(link_height))
        chain_height = float(np.exp(np.mean(np.log(np.clip(link_height, 1e-4, 1.0)))))
        height_reward = 0.5 * height + 0.5 * chain_height
        height_progress = float(np.mean(np.cos(theta) - np.cos(previous_theta)))
        low_motion = (1.0 - height) * height * float(np.mean(np.tanh(np.abs(theta_dot) / 2.0)))
        centered = 1.0 - min((x / p.x_threshold) ** 2, 1.0)
        velocity_cost = (
            p.x_position_cost_weight * (x / p.x_threshold) ** 2
            + p.x_velocity_cost_weight * x_dot**2
            + p.theta_velocity_cost_weight * float(np.mean(theta_dot**2))
            + p.top_theta_velocity_cost_weight
            * float(np.mean((link_height**2) * (theta_dot**2)))
        )
        action_cost = p.action_cost_weight * (force / p.force_mag) ** 2
        stable = self.is_stable()
        energy_score = self._energy_score(theta, theta_dot)
        reward = (
            p.alive_reward
            + p.upright_reward_weight * height_reward
            + p.centered_reward_weight * centered
            + p.energy_reward_weight * energy_score
            + p.swingup_progress_reward_weight * height_progress
            + p.swingup_motion_reward_weight * low_motion
            - velocity_cost
            - action_cost
        )
        if stable:
            reward += p.stable_bonus
        if terminated:
            reward -= p.termination_penalty
        return float(reward)

    def _get_obs(self) -> np.ndarray:
        return self.state.astype(np.float32, copy=True)

    @staticmethod
    def _wrap_angles(theta: np.ndarray) -> np.ndarray:
        return (theta + np.pi) % (2 * np.pi) - np.pi

    @staticmethod
    def _validate_reset_mode(reset_mode: str) -> str:
        valid_modes = {"downward", "upright", "uniform", "mixed"}
        if reset_mode not in valid_modes:
            raise ValueError(f"reset_mode must be one of {sorted(valid_modes)}")
        return reset_mode

    def _sample_reset_theta(self, count: int) -> np.ndarray:
        spread = 0.08
        mode = self.reset_mode
        if mode == "mixed":
            modes = self._rng.choice(
                np.array(["downward", "upright", "uniform"]),
                size=count,
                p=np.array([0.45, 0.25, 0.30]),
            )
        else:
            modes = np.full(count, mode)

        theta = np.empty((count, self.num_pendulums), dtype=np.float64)
        downward = modes == "downward"
        upright = modes == "upright"
        uniform = modes == "uniform"
        if np.any(downward):
            theta[downward] = np.pi + self._rng.uniform(
                -spread,
                spread,
                size=(int(np.sum(downward)), self.num_pendulums),
            )
        if np.any(upright):
            theta[upright] = self._rng.uniform(
                -2.0 * spread,
                2.0 * spread,
                size=(int(np.sum(upright)), self.num_pendulums),
            )
        if np.any(uniform):
            theta[uniform] = self._rng.uniform(
                -np.pi,
                np.pi,
                size=(int(np.sum(uniform)), self.num_pendulums),
            )
        return self._wrap_angles(theta)

    def _sample_reset_x_dot(self) -> float:
        if self.reset_mode in {"mixed", "uniform"}:
            return float(self._rng.uniform(-0.5, 0.5))
        return float(self._rng.uniform(-0.03, 0.03))

    def _sample_reset_theta_dot(self, count: int) -> np.ndarray:
        if self.reset_mode == "mixed":
            return self._rng.uniform(-4.0, 4.0, size=(count, self.num_pendulums)).astype(
                np.float32
            )
        if self.reset_mode == "uniform":
            return self._rng.uniform(-3.0, 3.0, size=(count, self.num_pendulums)).astype(
                np.float32
            )
        return self._rng.uniform(-0.02, 0.02, size=(count, self.num_pendulums)).astype(
            np.float32
        )

    def _energy_score(self, theta: np.ndarray, theta_dot: np.ndarray) -> float:
        p = self.params
        target_energy = 2.0 * p.gravity * p.pole_length
        link_energy = (
            0.5 * (p.pole_length * theta_dot) ** 2
            + p.gravity * p.pole_length * (np.cos(theta) + 1.0)
        )
        normalized_error = (link_energy - target_energy) / target_energy
        bottom_score = np.exp(-1.0)
        raw_score = np.exp(-np.mean(normalized_error**2))
        return float((raw_score - bottom_score) / (1.0 - bottom_score))

    def is_stable(self) -> bool:
        p = self.params
        x = float(self.state[0])
        theta = self.state[2 : 2 + self.num_pendulums]
        theta_dot = self.state[2 + self.num_pendulums :]
        return bool(
            abs(x) <= p.stable_x_threshold
            and abs(float(self.state[1])) <= p.stable_x_dot_threshold
            and np.all(np.abs(theta) <= p.stable_theta_threshold)
            and np.all(np.abs(theta_dot) <= p.stable_theta_dot_threshold)
        )
