from __future__ import annotations

from dataclasses import dataclass

import numpy as np
import torch

from cartpole_multi.config import NUM_PENDULUMS
from cartpole_multi.env import CartPoleParams
from cartpole_multi.observations import observation_dim, policy_observation_from_state


RESET_MODES = ("downward", "upright", "uniform", "mixed")


@dataclass(frozen=True)
class TorchStep:
    observation: torch.Tensor
    reward: torch.Tensor
    terminated: torch.Tensor
    truncated: torch.Tensor
    info: dict[str, torch.Tensor]


def resolve_torch_device(requested: str) -> torch.device:
    if requested in {"auto", "gpu"}:
        if torch.cuda.is_available():
            return torch.device("cuda")
        if hasattr(torch.backends, "mps") and torch.backends.mps.is_available():
            return torch.device("mps")
        raise RuntimeError(
            "requested GPU execution, but this PyTorch install cannot see CUDA or MPS; "
            "run training outside the sandbox or install a GPU-enabled PyTorch build"
        )
    return torch.device(requested)


def synchronize_device(device: torch.device) -> None:
    if device.type == "cuda":
        torch.cuda.synchronize(device)
    elif device.type == "mps" and hasattr(torch, "mps"):
        torch.mps.synchronize()


class TorchMultiPendulumCartPoleEnv:
    """Batched torch cartpole environment used for GPU evaluation."""

    def __init__(
        self,
        num_envs: int,
        num_pendulums: int = NUM_PENDULUMS,
        params: CartPoleParams | None = None,
        reset_mode: str = "mixed",
        device: torch.device | str = "cpu",
        seed: int | None = None,
        autoreset: bool = True,
    ) -> None:
        if num_envs < 1:
            raise ValueError("num_envs must be at least 1")
        if num_pendulums < 1:
            raise ValueError("num_pendulums must be at least 1")
        if reset_mode not in RESET_MODES:
            raise ValueError(f"reset_mode must be one of {RESET_MODES}")

        self.num_envs = int(num_envs)
        self.num_pendulums = int(num_pendulums)
        self.params = params or CartPoleParams()
        self.reset_mode = reset_mode
        self.autoreset = bool(autoreset)
        self.device = torch.device(device)
        self.dtype = torch.float32
        self.raw_state_dim = 2 + 2 * self.num_pendulums
        self.observation_dim = observation_dim(self.num_pendulums)
        self.generator = torch.Generator(device=self.device)

        n = self.num_pendulums
        idx = torch.arange(n, dtype=self.dtype, device=self.device)
        self._distal_counts = torch.arange(n, 0, -1, dtype=self.dtype, device=self.device)
        self._distal_count_matrix = (
            n - torch.maximum(idx[:, None], idx[None, :])
        ).to(self.dtype)
        self._eye_n = torch.eye(n, dtype=self.dtype, device=self.device)
        self._base_state = torch.zeros(self.raw_state_dim, dtype=self.dtype, device=self.device)

        self.state = torch.zeros(
            (self.num_envs, self.raw_state_dim),
            dtype=self.dtype,
            device=self.device,
        )
        self.step_count = torch.zeros(self.num_envs, dtype=torch.int32, device=self.device)
        self.episode_return = torch.zeros(self.num_envs, dtype=self.dtype, device=self.device)
        self.episode_length = torch.zeros(self.num_envs, dtype=torch.int32, device=self.device)
        self.episode_stable_steps = torch.zeros(
            self.num_envs,
            dtype=torch.int32,
            device=self.device,
        )
        self.generator.manual_seed(0 if seed is None else seed)

    def reset(self, seed: int | None = None) -> torch.Tensor:
        if seed is not None:
            self.generator.manual_seed(seed)
        self.state = self._sample_reset_state(self.num_envs)
        self.step_count.zero_()
        self.episode_return.zero_()
        self.episode_length.zero_()
        self.episode_stable_steps.zero_()
        return self.observation()

    def observation(self) -> torch.Tensor:
        return policy_observation_from_state(
            self.state,
            self.num_pendulums,
            self.params,
        )

    @torch.no_grad()
    def step_force(self, force: torch.Tensor) -> TorchStep:
        if self.num_pendulums == 2:
            return self._step_two_pendulums_force(force)
        return self._step_generic_force(force)

    def _step_generic_force(self, force: torch.Tensor) -> TorchStep:
        force = force.to(device=self.device, dtype=self.dtype).view(self.num_envs)
        p = self.params
        n = self.num_pendulums

        previous_state = self.state
        x = previous_state[:, 0]
        x_dot = previous_state[:, 1]
        theta = previous_state[:, 2 : 2 + n]
        theta_dot = self.state[:, 2 + n :]
        q_dot = torch.cat((x_dot[:, None], theta_dot), dim=1)

        mass_matrix = self._mass_matrix(theta)
        bias = self._bias_forces(theta, q_dot)
        generalized_force = torch.zeros_like(bias)
        generalized_force[:, 0] = force
        q_acc = self._solve_mass_matrix(mass_matrix, generalized_force - bias)

        x_dot = x_dot + p.dt * q_acc[:, 0]
        x = x + p.dt * x_dot
        theta_dot = theta_dot + p.dt * q_acc[:, 1:]
        theta = self._wrap_angles(theta + p.dt * theta_dot)

        next_state = torch.cat((x[:, None], x_dot[:, None], theta, theta_dot), dim=1)
        finite = torch.isfinite(next_state).all(dim=1)
        terminated = (x.abs() > p.x_threshold) | ~finite
        truncated = self.step_count + 1 >= p.max_episode_steps
        stable = self._is_stable_state(next_state)
        reward = self._reward(next_state, previous_state, force, terminated, stable)
        done = terminated | truncated

        self.state = next_state
        self.step_count += 1
        self.episode_return += reward
        self.episode_length += 1
        self.episode_stable_steps += stable.to(torch.int32)

        completed_return = torch.where(done, self.episode_return, torch.zeros_like(reward))
        completed_length = torch.where(done, self.episode_length, torch.zeros_like(self.step_count))
        completed_stable = torch.where(
            done,
            self.episode_stable_steps,
            torch.zeros_like(self.episode_stable_steps),
        )

        if self.autoreset:
            reset_state = self._sample_reset_state(self.num_envs)
            done_f = done[:, None]
            self.state = torch.where(done_f, reset_state, self.state)
            self.step_count = torch.where(done, torch.zeros_like(self.step_count), self.step_count)
            self.episode_return = torch.where(
                done,
                torch.zeros_like(self.episode_return),
                self.episode_return,
            )
            self.episode_length = torch.where(
                done,
                torch.zeros_like(self.episode_length),
                self.episode_length,
            )
            self.episode_stable_steps = torch.where(
                done,
                torch.zeros_like(self.episode_stable_steps),
                self.episode_stable_steps,
            )

        link_height = (torch.cos(theta) + 1.0) * 0.5
        chain_height = torch.exp(
            torch.mean(torch.log(torch.clamp(link_height, min=1e-4, max=1.0)), dim=1)
        )
        controlled_upright = chain_height * torch.exp(
            -0.5
            * torch.mean(
                (theta / self.params.stable_theta_threshold).square()
                + (theta_dot / self.params.stable_theta_dot_threshold).square(),
                dim=1,
            )
        )
        return TorchStep(
            observation=self.observation(),
            reward=reward,
            terminated=terminated,
            truncated=truncated,
            info={
                "done": done,
                "stable": stable,
                "episode_return": completed_return,
                "episode_length": completed_length.to(self.dtype),
                "episode_stable_steps": completed_stable.to(self.dtype),
                "x": x,
                "upright": torch.mean(torch.cos(theta), dim=1),
                "height": torch.mean(link_height, dim=1),
                "controlled_upright": controlled_upright,
                "top_theta_speed": torch.mean(
                    link_height.square() * theta_dot.abs(),
                    dim=1,
                ),
            },
        )

    def _step_two_pendulums_force(self, force: torch.Tensor) -> TorchStep:
        force = force.to(device=self.device, dtype=self.dtype).view(self.num_envs)
        p = self.params
        state = self.state

        x = state[:, 0]
        x_dot = state[:, 1]
        theta_1 = state[:, 2]
        theta_2 = state[:, 3]
        theta_dot_1 = state[:, 4]
        theta_dot_2 = state[:, 5]

        previous_cos_1 = torch.cos(theta_1)
        previous_cos_2 = torch.cos(theta_2)
        sin_1 = torch.sin(theta_1)
        sin_2 = torch.sin(theta_2)
        delta = theta_1 - theta_2
        sin_delta = torch.sin(delta)
        cos_delta = torch.cos(delta)

        pole_mass_length = p.pole_mass * p.pole_length
        pole_inertia = p.pole_mass * p.pole_length**2
        matrix_00 = p.cart_mass + 2.0 * p.pole_mass
        matrix_01 = 2.0 * pole_mass_length * previous_cos_1
        matrix_02 = pole_mass_length * previous_cos_2
        matrix_11 = 2.0 * pole_inertia
        matrix_12 = pole_inertia * cos_delta
        matrix_22 = pole_inertia

        theta_dot_1_sq = theta_dot_1.square()
        theta_dot_2_sq = theta_dot_2.square()
        bias_0 = (
            -2.0 * pole_mass_length * sin_1 * theta_dot_1_sq
            - pole_mass_length * sin_2 * theta_dot_2_sq
            + p.cart_friction * x_dot
        )
        bias_1 = (
            pole_inertia * sin_delta * theta_dot_2_sq
            - 2.0 * pole_mass_length * p.gravity * sin_1
            + p.pole_friction * theta_dot_1
        )
        bias_2 = (
            -pole_inertia * sin_delta * theta_dot_1_sq
            - pole_mass_length * p.gravity * sin_2
            + p.pole_friction * theta_dot_2
        )

        rhs_0 = force - bias_0
        rhs_1 = -bias_1
        rhs_2 = -bias_2

        cofactor_00 = matrix_11 * matrix_22 - matrix_12.square()
        cofactor_01 = matrix_02 * matrix_12 - matrix_01 * matrix_22
        cofactor_02 = matrix_01 * matrix_12 - matrix_02 * matrix_11
        cofactor_11 = matrix_00 * matrix_22 - matrix_02.square()
        cofactor_12 = matrix_01 * matrix_02 - matrix_00 * matrix_12
        cofactor_22 = matrix_00 * matrix_11 - matrix_01.square()
        determinant = (
            matrix_00 * cofactor_00
            + matrix_01 * cofactor_01
            + matrix_02 * cofactor_02
        )

        q_acc_0 = (
            cofactor_00 * rhs_0
            + cofactor_01 * rhs_1
            + cofactor_02 * rhs_2
        ) / determinant
        q_acc_1 = (
            cofactor_01 * rhs_0
            + cofactor_11 * rhs_1
            + cofactor_12 * rhs_2
        ) / determinant
        q_acc_2 = (
            cofactor_02 * rhs_0
            + cofactor_12 * rhs_1
            + cofactor_22 * rhs_2
        ) / determinant

        x_dot = x_dot + p.dt * q_acc_0
        x = x + p.dt * x_dot
        theta_dot_1 = theta_dot_1 + p.dt * q_acc_1
        theta_dot_2 = theta_dot_2 + p.dt * q_acc_2
        theta_1 = self._wrap_angles(theta_1 + p.dt * theta_dot_1)
        theta_2 = self._wrap_angles(theta_2 + p.dt * theta_dot_2)

        finite = (
            torch.isfinite(x)
            & torch.isfinite(x_dot)
            & torch.isfinite(theta_1)
            & torch.isfinite(theta_2)
            & torch.isfinite(theta_dot_1)
            & torch.isfinite(theta_dot_2)
        )
        terminated = (x.abs() > p.x_threshold) | ~finite
        truncated = self.step_count + 1 >= p.max_episode_steps

        theta_1_abs = theta_1.abs()
        theta_2_abs = theta_2.abs()
        theta_dot_1_abs = theta_dot_1.abs()
        theta_dot_2_abs = theta_dot_2.abs()
        stable = (
            (x.abs() <= p.stable_x_threshold)
            & (x_dot.abs() <= p.stable_x_dot_threshold)
            & (theta_1_abs <= p.stable_theta_threshold)
            & (theta_2_abs <= p.stable_theta_threshold)
            & (theta_dot_1_abs <= p.stable_theta_dot_threshold)
            & (theta_dot_2_abs <= p.stable_theta_dot_threshold)
        )

        cos_1 = torch.cos(theta_1)
        cos_2 = torch.cos(theta_2)
        link_height_1 = (cos_1 + 1.0) * 0.5
        link_height_2 = (cos_2 + 1.0) * 0.5
        height = 0.5 * (link_height_1 + link_height_2)
        chain_height = torch.sqrt(
            torch.clamp(link_height_1, min=1e-4, max=1.0)
            * torch.clamp(link_height_2, min=1e-4, max=1.0)
        )
        height_reward = 0.5 * height + 0.5 * chain_height
        height_progress = 0.5 * (
            (cos_1 - previous_cos_1)
            + (cos_2 - previous_cos_2)
        )
        low_motion = (
            (1.0 - height)
            * height
            * 0.5
            * (
                torch.tanh(theta_dot_1_abs / 2.0)
                + torch.tanh(theta_dot_2_abs / 2.0)
            )
        )
        centered = 1.0 - torch.clamp((x / p.x_threshold).square(), 0.0, 1.0)
        top_theta_speed = 0.5 * (
            link_height_1.square() * theta_dot_1_abs
            + link_height_2.square() * theta_dot_2_abs
        )
        theta_velocity_mean = 0.5 * (theta_dot_1.square() + theta_dot_2.square())
        velocity_cost = (
            p.x_position_cost_weight * (x / p.x_threshold).square()
            + p.x_velocity_cost_weight * x_dot.square()
            + p.theta_velocity_cost_weight * theta_velocity_mean
            + p.top_theta_velocity_cost_weight
            * 0.5
            * (
                link_height_1.square() * theta_dot_1.square()
                + link_height_2.square() * theta_dot_2.square()
            )
        )
        action_cost = p.action_cost_weight * (force / p.force_mag).square()
        energy_score = self._energy_score_two(
            cos_1,
            cos_2,
            theta_dot_1,
            theta_dot_2,
        )
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
        reward = reward + p.stable_bonus * stable.to(self.dtype)
        reward = torch.where(terminated, reward - p.termination_penalty, reward)
        done = terminated | truncated

        state[:, 0] = x
        state[:, 1] = x_dot
        state[:, 2] = theta_1
        state[:, 3] = theta_2
        state[:, 4] = theta_dot_1
        state[:, 5] = theta_dot_2
        self.step_count += 1
        self.episode_return += reward
        self.episode_length += 1
        self.episode_stable_steps += stable.to(torch.int32)

        completed_return = torch.where(done, self.episode_return, torch.zeros_like(reward))
        completed_length = torch.where(done, self.episode_length, torch.zeros_like(self.step_count))
        completed_stable = torch.where(
            done,
            self.episode_stable_steps,
            torch.zeros_like(self.episode_stable_steps),
        )

        if self.autoreset:
            reset_state = self._sample_reset_state(self.num_envs)
            done_f = done[:, None]
            self.state = torch.where(done_f, reset_state, self.state)
            self.step_count = torch.where(done, torch.zeros_like(self.step_count), self.step_count)
            self.episode_return = torch.where(
                done,
                torch.zeros_like(self.episode_return),
                self.episode_return,
            )
            self.episode_length = torch.where(
                done,
                torch.zeros_like(self.episode_length),
                self.episode_length,
            )
            self.episode_stable_steps = torch.where(
                done,
                torch.zeros_like(self.episode_stable_steps),
                self.episode_stable_steps,
            )

        controlled_upright = chain_height * torch.exp(
            -0.5
            * (
                (theta_1 / p.stable_theta_threshold).square()
                + (theta_2 / p.stable_theta_threshold).square()
                + (theta_dot_1 / p.stable_theta_dot_threshold).square()
                + (theta_dot_2 / p.stable_theta_dot_threshold).square()
            )
        )
        return TorchStep(
            observation=self.observation(),
            reward=reward,
            terminated=terminated,
            truncated=truncated,
            info={
                "done": done,
                "stable": stable,
                "episode_return": completed_return,
                "episode_length": completed_length.to(self.dtype),
                "episode_stable_steps": completed_stable.to(self.dtype),
                "x": x,
                "upright": 0.5 * (cos_1 + cos_2),
                "height": height,
                "controlled_upright": controlled_upright,
                "top_theta_speed": top_theta_speed,
            },
        )

    def _sample_reset_state(self, count: int) -> torch.Tensor:
        p = self.params
        n = self.num_pendulums
        shape = (count, n)
        state = torch.empty((count, self.raw_state_dim), dtype=self.dtype, device=self.device)
        state[:, 0] = self._uniform((count,), -0.03, 0.03)
        if self.reset_mode in {"mixed", "uniform"}:
            state[:, 1] = self._uniform((count,), -0.5, 0.5)
        else:
            state[:, 1] = self._uniform((count,), -0.03, 0.03)

        spread = 0.08
        downward = np.pi + self._uniform(shape, -spread, spread)
        upright = self._uniform(shape, -2.0 * spread, 2.0 * spread)
        uniform = self._uniform(shape, -np.pi, np.pi)

        if self.reset_mode == "mixed":
            mode_draw = torch.rand(
                (count, 1),
                dtype=self.dtype,
                device=self.device,
                generator=self.generator,
            )
            theta = torch.where(
                mode_draw < 0.45,
                downward,
                torch.where(mode_draw < 0.70, upright, uniform),
            )
            theta_dot = self._uniform(shape, -4.0, 4.0)
        elif self.reset_mode == "downward":
            theta = downward
            theta_dot = self._uniform(shape, -0.02, 0.02)
        elif self.reset_mode == "upright":
            theta = upright
            theta_dot = self._uniform(shape, -0.02, 0.02)
        else:
            theta = uniform
            theta_dot = self._uniform(shape, -3.0, 3.0)

        state[:, 2 : 2 + n] = self._wrap_angles(theta)
        state[:, 2 + n :] = theta_dot
        return state

    def _uniform(self, shape: tuple[int, ...], low: float, high: float) -> torch.Tensor:
        return low + (high - low) * torch.rand(
            shape,
            dtype=self.dtype,
            device=self.device,
            generator=self.generator,
        )

    def _mass_matrix(self, theta: torch.Tensor) -> torch.Tensor:
        p = self.params
        n = self.num_pendulums
        batch = theta.shape[0]
        size = n + 1
        matrix = torch.zeros((batch, size, size), dtype=self.dtype, device=self.device)
        matrix[:, 0, 0] = p.cart_mass + n * p.pole_mass

        cart_coupling = p.pole_mass * p.pole_length * self._distal_counts * torch.cos(theta)
        matrix[:, 0, 1:] = cart_coupling
        matrix[:, 1:, 0] = cart_coupling

        angle_delta = theta[:, :, None] - theta[:, None, :]
        matrix[:, 1:, 1:] = (
            p.pole_mass
            * p.pole_length**2
            * self._distal_count_matrix
            * torch.cos(angle_delta)
        )
        return matrix

    def _bias_forces(self, theta: torch.Tensor, q_dot: torch.Tensor) -> torch.Tensor:
        p = self.params
        n = self.num_pendulums
        batch = theta.shape[0]
        size = n + 1
        d_mass = torch.zeros(
            (batch, size, size, size),
            dtype=self.dtype,
            device=self.device,
        )

        cart_derivative = (
            -p.pole_mass
            * p.pole_length
            * self._distal_counts
            * torch.sin(theta)
        )
        theta_idx = torch.arange(1, size, device=self.device)
        d_mass[:, theta_idx, 0, theta_idx] = cart_derivative
        d_mass[:, theta_idx, theta_idx, 0] = cart_derivative

        angle_delta = theta[:, :, None] - theta[:, None, :]
        base = (
            -p.pole_mass
            * p.pole_length**2
            * self._distal_count_matrix
            * torch.sin(angle_delta)
        )
        derivative_sign = self._eye_n[None, :, :, None] - self._eye_n[None, :, None, :]
        d_mass[:, 1:, 1:, 1:] = base[:, None, :, :] * derivative_sign

        mass_dot_q_dot = torch.einsum("brkj,br,bj->bk", d_mass, q_dot, q_dot)
        kinetic_gradient = 0.5 * torch.einsum("brij,bi,bj->br", d_mass, q_dot, q_dot)

        potential_gradient = torch.zeros((batch, size), dtype=self.dtype, device=self.device)
        potential_gradient[:, 1:] = (
            -p.pole_mass
            * p.gravity
            * p.pole_length
            * self._distal_counts
            * torch.sin(theta)
        )

        damping = torch.zeros((batch, size), dtype=self.dtype, device=self.device)
        damping[:, 0] = p.cart_friction * q_dot[:, 0]
        damping[:, 1:] = p.pole_friction * q_dot[:, 1:]
        return mass_dot_q_dot - kinetic_gradient + potential_gradient + damping

    def _solve_mass_matrix(self, matrix: torch.Tensor, rhs: torch.Tensor) -> torch.Tensor:
        size = matrix.shape[1]
        eps = torch.tensor(1e-7, dtype=self.dtype, device=self.device)
        lower = torch.zeros_like(matrix)

        for col in range(size):
            diagonal = matrix[:, col, col]
            if col:
                diagonal = diagonal - lower[:, col, :col].square().sum(dim=1)
            lower[:, col, col] = torch.sqrt(torch.clamp(diagonal, min=eps))

            for row in range(col + 1, size):
                value = matrix[:, row, col]
                if col:
                    value = value - (lower[:, row, :col] * lower[:, col, :col]).sum(dim=1)
                lower[:, row, col] = value / lower[:, col, col]

        y = torch.zeros_like(rhs)
        for row in range(size):
            value = rhs[:, row]
            if row:
                value = value - (lower[:, row, :row] * y[:, :row]).sum(dim=1)
            y[:, row] = value / lower[:, row, row]

        solution = torch.zeros_like(rhs)
        for row in range(size - 1, -1, -1):
            value = y[:, row]
            if row + 1 < size:
                value = value - (lower[:, row + 1 :, row] * solution[:, row + 1 :]).sum(dim=1)
            solution[:, row] = value / lower[:, row, row]

        return solution

    def _reward(
        self,
        state: torch.Tensor,
        previous_state: torch.Tensor,
        force: torch.Tensor,
        terminated: torch.Tensor,
        stable: torch.Tensor,
    ) -> torch.Tensor:
        p = self.params
        n = self.num_pendulums
        x = state[:, 0]
        x_dot = state[:, 1]
        theta = state[:, 2 : 2 + n]
        theta_dot = state[:, 2 + n :]
        previous_theta = previous_state[:, 2 : 2 + n]

        link_height = (torch.cos(theta) + 1.0) * 0.5
        height = torch.mean(link_height, dim=1)
        chain_height = torch.exp(
            torch.mean(torch.log(torch.clamp(link_height, min=1e-4, max=1.0)), dim=1)
        )
        height_reward = 0.5 * height + 0.5 * chain_height
        height_progress = torch.mean(torch.cos(theta) - torch.cos(previous_theta), dim=1)
        low_motion = (
            (1.0 - height)
            * height
            * torch.mean(torch.tanh(theta_dot.abs() / 2.0), dim=1)
        )
        centered = 1.0 - torch.clamp((x / p.x_threshold).square(), 0.0, 1.0)
        velocity_cost = (
            p.x_position_cost_weight * (x / p.x_threshold).square()
            + p.x_velocity_cost_weight * x_dot.square()
            + p.theta_velocity_cost_weight * torch.mean(theta_dot.square(), dim=1)
            + p.top_theta_velocity_cost_weight
            * torch.mean(link_height.square() * theta_dot.square(), dim=1)
        )
        action_cost = p.action_cost_weight * (force / p.force_mag).square()
        reward = (
            p.alive_reward
            + p.upright_reward_weight * height_reward
            + p.centered_reward_weight * centered
            + p.energy_reward_weight * self._energy_score(theta, theta_dot)
            + p.swingup_progress_reward_weight * height_progress
            + p.swingup_motion_reward_weight * low_motion
            - velocity_cost
            - action_cost
        )
        reward = reward + p.stable_bonus * stable.to(self.dtype)
        return torch.where(terminated, reward - p.termination_penalty, reward)

    def _energy_score(self, theta: torch.Tensor, theta_dot: torch.Tensor) -> torch.Tensor:
        p = self.params
        target_energy = 2.0 * p.gravity * p.pole_length
        link_energy = (
            0.5 * (p.pole_length * theta_dot).square()
            + p.gravity * p.pole_length * (torch.cos(theta) + 1.0)
        )
        normalized_error = (link_energy - target_energy) / target_energy
        bottom_score = np.exp(-1.0)
        raw_score = torch.exp(-torch.mean(normalized_error.square(), dim=1))
        return (raw_score - bottom_score) / (1.0 - bottom_score)

    def _energy_score_two(
        self,
        cos_1: torch.Tensor,
        cos_2: torch.Tensor,
        theta_dot_1: torch.Tensor,
        theta_dot_2: torch.Tensor,
    ) -> torch.Tensor:
        p = self.params
        target_energy = 2.0 * p.gravity * p.pole_length
        velocity_scale = 0.5 * p.pole_length**2
        potential_scale = p.gravity * p.pole_length
        error_1 = (
            velocity_scale * theta_dot_1.square()
            + potential_scale * (cos_1 + 1.0)
            - target_energy
        ) / target_energy
        error_2 = (
            velocity_scale * theta_dot_2.square()
            + potential_scale * (cos_2 + 1.0)
            - target_energy
        ) / target_energy
        bottom_score = np.exp(-1.0)
        raw_score = torch.exp(-0.5 * (error_1.square() + error_2.square()))
        return (raw_score - bottom_score) / (1.0 - bottom_score)

    def _is_stable_state(self, state: torch.Tensor) -> torch.Tensor:
        p = self.params
        n = self.num_pendulums
        theta = state[:, 2 : 2 + n]
        theta_dot = state[:, 2 + n :]
        return (
            (state[:, 0].abs() <= p.stable_x_threshold)
            & (state[:, 1].abs() <= p.stable_x_dot_threshold)
            & torch.all(theta.abs() <= p.stable_theta_threshold, dim=1)
            & torch.all(theta_dot.abs() <= p.stable_theta_dot_threshold, dim=1)
        )

    @staticmethod
    def _wrap_angles(theta: torch.Tensor) -> torch.Tensor:
        return torch.remainder(theta + np.pi, 2 * np.pi) - np.pi
