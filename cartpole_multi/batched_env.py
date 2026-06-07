from __future__ import annotations

import numpy as np
from gymnasium import spaces

from cartpole_multi.config import NUM_PENDULUMS
from cartpole_multi.env import CartPoleParams, MultiPendulumCartPoleEnv


class BatchedMultiPendulumCartPoleEnv:
    """Fast NumPy vectorized version of the serial-chain cartpole environment."""

    def __init__(
        self,
        num_pendulums: int = NUM_PENDULUMS,
        num_envs: int = 1024,
        params: CartPoleParams | None = None,
        reset_mode: str = "downward",
        seed: int = 0,
    ) -> None:
        if num_pendulums < 1:
            raise ValueError("num_pendulums must be at least 1")
        if num_envs < 1:
            raise ValueError("num_envs must be at least 1")

        self.num_pendulums = int(num_pendulums)
        self.num_envs = int(num_envs)
        self.params = params or CartPoleParams()
        self.reset_mode = MultiPendulumCartPoleEnv._validate_reset_mode(reset_mode)
        self.single_action_space = spaces.Discrete(3)
        self.single_observation_space = spaces.Box(
            -np.inf,
            np.inf,
            shape=(2 + 2 * self.num_pendulums,),
            dtype=np.float32,
        )

        self.obs_dim = int(self.single_observation_space.shape[0])
        self.state = np.zeros((self.num_envs, self.obs_dim), dtype=np.float32)
        self.step_count = np.zeros(self.num_envs, dtype=np.int32)
        self._rng = np.random.default_rng(seed)

        n = self.num_pendulums
        link_ids = np.arange(n)
        self._link_eye = np.eye(n, dtype=np.float64)
        self._distal_counts = np.arange(n, 0, -1, dtype=np.float64)
        self._distal_pair_counts = (n - np.maximum.outer(link_ids, link_ids)).astype(np.float64)
        self._diff_sign = self._link_eye[:, :, None] - self._link_eye[:, None, :]

    def reset(self, seed: int | None = None) -> tuple[np.ndarray, dict]:
        if seed is not None:
            self._rng = np.random.default_rng(seed)
        self._reset_indices(np.arange(self.num_envs))
        return self.state, {}

    def step(
        self,
        actions: np.ndarray,
    ) -> tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray, dict]:
        p = self.params
        actions = np.asarray(actions, dtype=np.int64).reshape(self.num_envs)
        if np.any((actions < 0) | (actions >= self.single_action_space.n)):
            raise ValueError("actions must be in [0, 2]")

        force = (actions - 1).astype(np.float64) * p.force_mag
        x = self.state[:, 0].astype(np.float64)
        x_dot = self.state[:, 1].astype(np.float64)
        theta = self.state[:, 2 : 2 + self.num_pendulums].astype(np.float64)
        theta_dot = self.state[:, 2 + self.num_pendulums :].astype(np.float64)

        q_dot = np.empty((self.num_envs, self.num_pendulums + 1), dtype=np.float64)
        q_dot[:, 0] = x_dot
        q_dot[:, 1:] = theta_dot
        mass_matrix = self._mass_matrix(theta)
        bias = self._bias_forces(theta, q_dot)
        rhs = -bias
        rhs[:, 0] += force
        q_acc = np.linalg.solve(mass_matrix, rhs)

        x_dot = x_dot + p.dt * q_acc[:, 0]
        x = x + p.dt * x_dot
        theta_dot = theta_dot + p.dt * q_acc[:, 1:]
        theta = MultiPendulumCartPoleEnv._wrap_angles(theta + p.dt * theta_dot)

        self.state[:, 0] = x.astype(np.float32)
        self.state[:, 1] = x_dot.astype(np.float32)
        self.state[:, 2 : 2 + self.num_pendulums] = theta.astype(np.float32)
        self.state[:, 2 + self.num_pendulums :] = theta_dot.astype(np.float32)
        self.step_count += 1

        terminated = (np.abs(x) > p.x_threshold) | ~np.all(np.isfinite(self.state), axis=1)
        truncated = self.step_count >= p.max_episode_steps
        done = terminated | truncated
        reward = self._reward(force, terminated)

        done_indices = np.flatnonzero(done)
        if done_indices.size:
            self._reset_indices(done_indices)

        return self.state, reward, terminated, truncated, {}

    def close(self) -> None:
        return None

    def _reset_indices(self, indices: np.ndarray) -> None:
        count = len(indices)

        self.state[indices, 0] = self._rng.uniform(-0.03, 0.03, size=count)
        self.state[indices, 1] = self._sample_reset_x_dot(count)
        self.state[indices, 2 : 2 + self.num_pendulums] = self._sample_reset_theta(
            count
        ).astype(np.float32)
        self.state[indices, 2 + self.num_pendulums :] = self._sample_reset_theta_dot(count)
        self.step_count[indices] = 0

    def _mass_matrix(self, theta: np.ndarray) -> np.ndarray:
        p = self.params
        n = self.num_pendulums
        size = n + 1
        matrix = np.empty((self.num_envs, size, size), dtype=np.float64)
        matrix[:, :, :] = 0.0
        matrix[:, 0, 0] = p.cart_mass + n * p.pole_mass

        cart_coupling = p.pole_mass * p.pole_length * self._distal_counts * np.cos(theta)
        matrix[:, 0, 1:] = cart_coupling
        matrix[:, 1:, 0] = cart_coupling

        angle_diffs = theta[:, :, None] - theta[:, None, :]
        matrix[:, 1:, 1:] = (
            p.pole_mass
            * p.pole_length**2
            * self._distal_pair_counts[None, :, :]
            * np.cos(angle_diffs)
        )
        return matrix

    def _bias_forces(self, theta: np.ndarray, q_dot: np.ndarray) -> np.ndarray:
        p = self.params
        n = self.num_pendulums
        size = n + 1
        d_mass = np.zeros((self.num_envs, size, size, size), dtype=np.float64)

        theta_derivative = (
            -p.pole_mass * p.pole_length * self._distal_counts[None, :] * np.sin(theta)
        )
        link_indices = np.arange(n) + 1
        d_mass[:, link_indices, 0, link_indices] = theta_derivative
        d_mass[:, link_indices, link_indices, 0] = theta_derivative

        angle_diffs = theta[:, :, None] - theta[:, None, :]
        d_mass[:, 1:, 1:, 1:] = (
            -p.pole_mass
            * p.pole_length**2
            * self._distal_pair_counts[None, None, :, :]
            * np.sin(angle_diffs)[:, None, :, :]
            * self._diff_sign[None, :, :, :]
        )

        mass_dot_q_dot = np.einsum("erij,er,ej->ei", d_mass, q_dot, q_dot, optimize=True)
        kinetic_gradient = 0.5 * np.einsum(
            "erij,ei,ej->er",
            d_mass,
            q_dot,
            q_dot,
            optimize=True,
        )

        bias = mass_dot_q_dot - kinetic_gradient
        bias[:, 1:] += (
            -p.pole_mass
            * p.gravity
            * p.pole_length
            * self._distal_counts[None, :]
            * np.sin(theta)
        )
        bias[:, 0] += p.cart_friction * q_dot[:, 0]
        bias[:, 1:] += p.pole_friction * q_dot[:, 1:]
        return bias

    def _reward(self, force: np.ndarray, terminated: np.ndarray) -> np.ndarray:
        p = self.params
        x = self.state[:, 0].astype(np.float64)
        x_dot = self.state[:, 1].astype(np.float64)
        theta = self.state[:, 2 : 2 + self.num_pendulums].astype(np.float64)
        theta_dot = self.state[:, 2 + self.num_pendulums :].astype(np.float64)

        upright = np.mean(np.cos(theta), axis=1)
        centered = 1.0 - np.minimum((x / p.x_threshold) ** 2, 1.0)
        velocity_cost = (
            p.x_velocity_cost_weight * x_dot**2
            + p.theta_velocity_cost_weight * np.mean(theta_dot**2, axis=1)
        )
        action_cost = p.action_cost_weight * (force / p.force_mag) ** 2
        stable = (
            (np.abs(x) <= p.stable_x_threshold)
            & np.all(np.abs(theta) <= p.stable_theta_threshold, axis=1)
            & np.all(np.abs(theta_dot) <= p.stable_theta_dot_threshold, axis=1)
        )
        target_energy = 2.0 * p.gravity * p.pole_length
        link_energy = (
            0.5 * (p.pole_length * theta_dot) ** 2
            + p.gravity * p.pole_length * (np.cos(theta) + 1.0)
        )
        normalized_error = (link_energy - target_energy) / target_energy
        energy_score = -np.mean(normalized_error**2, axis=1)
        reward = (
            p.alive_reward
            + p.upright_reward_weight * ((upright + 1.0) * 0.5)
            + p.centered_reward_weight * centered
            + p.energy_reward_weight * energy_score
            - velocity_cost
            - action_cost
        )
        reward[stable] += p.stable_bonus
        reward = reward.astype(np.float32)
        reward[terminated] -= p.termination_penalty
        return reward

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
        return MultiPendulumCartPoleEnv._wrap_angles(theta)

    def _sample_reset_x_dot(self, count: int) -> np.ndarray:
        if self.reset_mode in {"mixed", "uniform"}:
            return self._rng.uniform(-0.5, 0.5, size=count).astype(np.float32)
        return self._rng.uniform(-0.03, 0.03, size=count).astype(np.float32)

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
