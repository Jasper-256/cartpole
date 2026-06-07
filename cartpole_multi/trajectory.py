from __future__ import annotations

import argparse
import time
from dataclasses import dataclass

import numpy as np
import torch

from cartpole_multi.env import CartPoleParams, MultiPendulumCartPoleEnv


LINEAR_FEEDBACK_GAIN = np.asarray(
    [33.14, 48.76, -352.44, 674.17, 15.41, 124.83],
    dtype=np.float64,
)


@dataclass
class TrajectoryPlan:
    actions: list[int]
    segments: np.ndarray
    states: np.ndarray
    device: str
    batch_size: int
    iterations: int
    simulated_steps: int
    best_score: float


def resolve_torch_device(requested: str) -> torch.device:
    if requested in {"auto", "gpu"}:
        if torch.cuda.is_available():
            return torch.device("cuda")
        if hasattr(torch.backends, "mps") and torch.backends.mps.is_available():
            return torch.device("mps")
        raise RuntimeError(
            "requested GPU execution, but this PyTorch install cannot see CUDA or MPS; "
            "pass --device cpu only when you intentionally want a CPU development run"
        )
    return torch.device(requested)


def default_trajectory_batch_size(device: torch.device) -> int:
    if device.type == "cuda":
        return 131_072
    if device.type == "mps":
        return 65_536
    return 16_384


def optimize_trajectory_plan(args: argparse.Namespace) -> TrajectoryPlan:
    if args.num_pendulums != 2:
        raise ValueError("trajectory optimizer currently supports --num-pendulums 2")

    device = resolve_torch_device(args.device)
    batch_size = args.trajectory_batch_size or default_trajectory_batch_size(device)
    params = CartPoleParams()

    base_env = MultiPendulumCartPoleEnv(
        num_pendulums=args.num_pendulums,
        reset_mode=args.eval_reset_mode,
        seed=args.seed,
    )
    initial_state, _info = base_env.reset()

    torch.manual_seed(args.trajectory_seed)
    probs = torch.full(
        (args.trajectory_segments, 3),
        1.0 / 3.0,
        dtype=torch.float32,
        device=device,
    )
    best_segments: torch.Tensor | None = None
    best_score = -float("inf")
    simulated_steps = 0

    for iteration in range(1, args.trajectory_iterations + 1):
        iteration_start = time.perf_counter()
        sampled = sample_segment_actions_torch(batch_size, probs)
        if best_segments is not None:
            inject_mutated_best_segments(sampled, best_segments, args.trajectory_mutation_rate)
        sample_elapsed = time.perf_counter() - iteration_start

        score_start = time.perf_counter()
        scores = score_double_action_segments_torch(
            sampled,
            np.asarray(initial_state, dtype=np.float32),
            params,
            action_repeat=args.trajectory_action_repeat,
            upright_reward=args.trajectory_upright_reward,
            theta_dot_cost=args.trajectory_theta_dot_cost,
            x_cost=args.trajectory_x_cost,
            x_dot_cost=args.trajectory_x_dot_cost,
            cost_weight=args.trajectory_cost_weight,
            stable_bonus=args.trajectory_stable_bonus,
            min_cost_weight=args.trajectory_min_cost_weight,
            final_cost_weight=args.trajectory_final_cost_weight,
            alive_bonus=args.trajectory_alive_bonus,
            terminal_penalty=args.trajectory_terminal_penalty,
            stable_x_threshold=args.stable_x_threshold,
            stable_x_dot_threshold=args.stable_x_dot_threshold,
            stable_theta_threshold=args.stable_theta_threshold,
            stable_theta_dot_threshold=args.stable_theta_dot_threshold,
        )
        score_elapsed = time.perf_counter() - score_start
        simulated_steps += batch_size * args.trajectory_segments * args.trajectory_action_repeat

        iteration_best_score, best_idx = torch.max(scores, dim=0)
        if float(iteration_best_score.item()) > best_score:
            best_score = float(iteration_best_score.item())
            best_segments = sampled[best_idx].detach().clone()

        update_start = time.perf_counter()
        probs = update_segment_probs_torch(
            sampled,
            scores,
            args.trajectory_elite_frac,
            probs,
        )
        update_elapsed = time.perf_counter() - update_start
        iteration_elapsed = max(time.perf_counter() - iteration_start, 1e-9)

        if args.log_every and iteration % args.log_every == 0:
            simulated_per_update = (
                batch_size
                * args.trajectory_segments
                * args.trajectory_action_repeat
            )
            print(
                f"trajectory_update={iteration}/{args.trajectory_iterations} "
                f"device={device.type} batch={batch_size} "
                f"best_score={float(iteration_best_score.item()):.1f} "
                f"global_best_score={best_score:.1f} "
                f"elapsed={iteration_elapsed:.3f}s "
                f"sample={sample_elapsed:.3f}s "
                f"score={score_elapsed:.3f}s "
                f"update={update_elapsed:.3f}s "
                f"sim_sps={simulated_per_update / iteration_elapsed:.0f}",
                flush=True,
            )

    if best_segments is None:
        raise RuntimeError("trajectory optimizer did not sample any action segments")

    segment_array = best_segments.cpu().numpy().astype(np.int8, copy=False)
    actions = [
        int(action)
        for action in segment_array
        for _repeat in range(args.trajectory_action_repeat)
    ]
    states = rollout_double_actions(
        np.asarray(initial_state, dtype=np.float32),
        actions,
        params,
    )
    return TrajectoryPlan(
        actions=actions,
        segments=segment_array,
        states=states,
        device=device.type,
        batch_size=batch_size,
        iterations=args.trajectory_iterations,
        simulated_steps=simulated_steps,
        best_score=best_score,
    )


def sample_segment_actions_torch(num_sequences: int, probs: torch.Tensor) -> torch.Tensor:
    uniform = torch.rand((num_sequences, probs.shape[0]), dtype=torch.float32, device=probs.device)
    cumulative = torch.cumsum(probs, dim=1)
    return (
        (uniform > cumulative[:, 0].unsqueeze(0)).to(torch.int8)
        + (uniform > cumulative[:, 1].unsqueeze(0)).to(torch.int8)
    )


def inject_mutated_best_segments(
    sampled: torch.Tensor,
    best_segments: torch.Tensor,
    mutation_rate: float,
) -> None:
    keep_count = min(max(2, sampled.shape[0] // 16), 2048)
    sampled[0] = best_segments
    if keep_count == 1:
        return
    base = best_segments.unsqueeze(0).expand(keep_count - 1, -1)
    mutations = torch.rand(base.shape, dtype=torch.float32, device=sampled.device) < mutation_rate
    replacements = torch.randint(0, 3, base.shape, dtype=torch.int8, device=sampled.device)
    sampled[1:keep_count] = torch.where(mutations, replacements, base)


def update_segment_probs_torch(
    segments: torch.Tensor,
    scores: torch.Tensor,
    elite_frac: float,
    previous_probs: torch.Tensor,
) -> torch.Tensor:
    elite_count = max(32, int(segments.shape[0] * elite_frac))
    elite_indices = torch.topk(scores, elite_count, sorted=False).indices
    elite = segments[elite_indices]
    counts = torch.stack(
        [(elite == action).to(torch.float32).mean(dim=0) for action in range(3)],
        dim=1,
    )
    probs = 0.45 * previous_probs + 0.55 * counts
    probs = 0.02 / 3.0 + 0.98 * probs
    return probs / probs.sum(dim=1, keepdim=True)


def score_double_action_segments_torch(
    segments: torch.Tensor,
    initial_state: np.ndarray,
    params: CartPoleParams,
    *,
    action_repeat: int,
    upright_reward: float,
    theta_dot_cost: float,
    x_cost: float,
    x_dot_cost: float,
    cost_weight: float,
    stable_bonus: float,
    min_cost_weight: float,
    final_cost_weight: float,
    alive_bonus: float,
    terminal_penalty: float,
    stable_x_threshold: float,
    stable_x_dot_threshold: float,
    stable_theta_threshold: float,
    stable_theta_dot_threshold: float,
) -> torch.Tensor:
    device = segments.device
    num_sequences = segments.shape[0]
    state = torch.as_tensor(initial_state, dtype=torch.float32, device=device)
    x = torch.full((num_sequences,), float(state[0]), dtype=torch.float32, device=device)
    x_dot = torch.full((num_sequences,), float(state[1]), dtype=torch.float32, device=device)
    theta_1 = torch.full((num_sequences,), float(state[2]), dtype=torch.float32, device=device)
    theta_2 = torch.full((num_sequences,), float(state[3]), dtype=torch.float32, device=device)
    theta_dot_1 = torch.full((num_sequences,), float(state[4]), dtype=torch.float32, device=device)
    theta_dot_2 = torch.full((num_sequences,), float(state[5]), dtype=torch.float32, device=device)

    alive = torch.ones(num_sequences, dtype=torch.bool, device=device)
    score = torch.zeros(num_sequences, dtype=torch.float32, device=device)
    stable_steps = torch.zeros(num_sequences, dtype=torch.float32, device=device)
    min_cost = torch.full((num_sequences,), float("inf"), dtype=torch.float32, device=device)
    final_cost = torch.zeros(num_sequences, dtype=torch.float32, device=device)
    step_count = 0

    for segment_idx in range(segments.shape[1]):
        force = (segments[:, segment_idx].to(torch.float32) - 1.0) * params.force_mag
        for _repeat in range(action_repeat):
            step_count += 1
            x, x_dot, theta_1, theta_2, theta_dot_1, theta_dot_2 = step_double_soa_torch(
                x,
                x_dot,
                theta_1,
                theta_2,
                theta_dot_1,
                theta_dot_2,
                force,
                params,
            )
            angle_cost = 0.5 * (theta_1.square() + theta_2.square())
            velocity_cost = 0.5 * (theta_dot_1.square() + theta_dot_2.square())
            cost = (
                angle_cost
                + theta_dot_cost * velocity_cost
                + x_cost * x.square()
                + x_dot_cost * x_dot.square()
            )
            stable = (
                (torch.abs(x) <= stable_x_threshold)
                & (torch.abs(x_dot) <= stable_x_dot_threshold)
                & (torch.abs(theta_1) <= stable_theta_threshold)
                & (torch.abs(theta_2) <= stable_theta_threshold)
                & (torch.abs(theta_dot_1) <= stable_theta_dot_threshold)
                & (torch.abs(theta_dot_2) <= stable_theta_dot_threshold)
                & alive
            )
            upright = 0.5 * (torch.cos(theta_1) + torch.cos(theta_2))
            stable_steps += stable.to(torch.float32)
            min_cost = torch.where(alive, torch.minimum(min_cost, cost), min_cost)
            score += alive.to(torch.float32) * (upright_reward * upright - cost_weight * cost)
            final_cost = torch.where(alive, cost, final_cost)

            finite = (
                torch.isfinite(x)
                & torch.isfinite(x_dot)
                & torch.isfinite(theta_1)
                & torch.isfinite(theta_2)
                & torch.isfinite(theta_dot_1)
                & torch.isfinite(theta_dot_2)
            )
            done = ((torch.abs(x) > params.x_threshold) | ~finite) & alive
            if step_count >= params.max_episode_steps:
                done = done | alive
            score = torch.where(done, score - terminal_penalty, score)
            alive = alive & ~done
            x = torch.where(done, state[0].expand_as(x), x)
            x_dot = torch.where(done, state[1].expand_as(x_dot), x_dot)
            theta_1 = torch.where(done, state[2].expand_as(theta_1), theta_1)
            theta_2 = torch.where(done, state[3].expand_as(theta_2), theta_2)
            theta_dot_1 = torch.where(done, state[4].expand_as(theta_dot_1), theta_dot_1)
            theta_dot_2 = torch.where(done, state[5].expand_as(theta_dot_2), theta_dot_2)

    score += (
        stable_bonus * stable_steps
        - min_cost_weight * min_cost
        - final_cost_weight * final_cost
        + alive_bonus * alive.to(torch.float32)
    )
    return score


def step_double_soa_torch(
    x: torch.Tensor,
    x_dot: torch.Tensor,
    theta_1: torch.Tensor,
    theta_2: torch.Tensor,
    theta_dot_1: torch.Tensor,
    theta_dot_2: torch.Tensor,
    force: torch.Tensor,
    params: CartPoleParams,
) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor]:
    pole_mass_length = params.pole_mass * params.pole_length
    pole_inertia = params.pole_mass * params.pole_length**2
    sin_1 = torch.sin(theta_1)
    sin_2 = torch.sin(theta_2)
    cos_1 = torch.cos(theta_1)
    cos_2 = torch.cos(theta_2)
    delta = theta_1 - theta_2
    sin_delta = torch.sin(delta)
    cos_delta = torch.cos(delta)

    matrix_00 = params.cart_mass + 2.0 * params.pole_mass
    matrix_01 = 2.0 * pole_mass_length * cos_1
    matrix_02 = pole_mass_length * cos_2
    matrix_11 = 2.0 * pole_inertia
    matrix_12 = pole_inertia * cos_delta
    matrix_22 = pole_inertia

    bias_0 = (
        -2.0 * pole_mass_length * sin_1 * theta_dot_1.square()
        - pole_mass_length * sin_2 * theta_dot_2.square()
        + params.cart_friction * x_dot
    )
    bias_1 = (
        pole_inertia * sin_delta * theta_dot_2.square()
        - 2.0 * pole_mass_length * params.gravity * sin_1
        + params.pole_friction * theta_dot_1
    )
    bias_2 = (
        -pole_inertia * sin_delta * theta_dot_1.square()
        - pole_mass_length * params.gravity * sin_2
        + params.pole_friction * theta_dot_2
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

    x_dot = x_dot + params.dt * q_acc_0
    x = x + params.dt * x_dot
    theta_dot_1 = theta_dot_1 + params.dt * q_acc_1
    theta_dot_2 = theta_dot_2 + params.dt * q_acc_2
    theta_1 = wrap_angles_torch(theta_1 + params.dt * theta_dot_1)
    theta_2 = wrap_angles_torch(theta_2 + params.dt * theta_dot_2)
    return x, x_dot, theta_1, theta_2, theta_dot_1, theta_dot_2


def wrap_angles_torch(theta: torch.Tensor) -> torch.Tensor:
    return torch.remainder(theta + np.pi, 2 * np.pi) - np.pi


def step_double_soa_numpy(
    x: np.ndarray,
    x_dot: np.ndarray,
    theta_1: np.ndarray,
    theta_2: np.ndarray,
    theta_dot_1: np.ndarray,
    theta_dot_2: np.ndarray,
    force: np.ndarray,
    params: CartPoleParams,
) -> tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
    pole_mass_length = params.pole_mass * params.pole_length
    pole_inertia = params.pole_mass * params.pole_length**2
    sin_1 = np.sin(theta_1)
    sin_2 = np.sin(theta_2)
    cos_1 = np.cos(theta_1)
    cos_2 = np.cos(theta_2)
    delta = theta_1 - theta_2
    sin_delta = np.sin(delta)
    cos_delta = np.cos(delta)

    matrix_00 = params.cart_mass + 2.0 * params.pole_mass
    matrix_01 = 2.0 * pole_mass_length * cos_1
    matrix_02 = pole_mass_length * cos_2
    matrix_11 = 2.0 * pole_inertia
    matrix_12 = pole_inertia * cos_delta
    matrix_22 = pole_inertia

    bias_0 = (
        -2.0 * pole_mass_length * sin_1 * theta_dot_1**2
        - pole_mass_length * sin_2 * theta_dot_2**2
        + params.cart_friction * x_dot
    )
    bias_1 = (
        pole_inertia * sin_delta * theta_dot_2**2
        - 2.0 * pole_mass_length * params.gravity * sin_1
        + params.pole_friction * theta_dot_1
    )
    bias_2 = (
        -pole_inertia * sin_delta * theta_dot_1**2
        - pole_mass_length * params.gravity * sin_2
        + params.pole_friction * theta_dot_2
    )

    rhs_0 = force - bias_0
    rhs_1 = -bias_1
    rhs_2 = -bias_2

    cofactor_00 = matrix_11 * matrix_22 - matrix_12**2
    cofactor_01 = matrix_02 * matrix_12 - matrix_01 * matrix_22
    cofactor_02 = matrix_01 * matrix_12 - matrix_02 * matrix_11
    cofactor_11 = matrix_00 * matrix_22 - matrix_02**2
    cofactor_12 = matrix_01 * matrix_02 - matrix_00 * matrix_12
    cofactor_22 = matrix_00 * matrix_11 - matrix_01**2
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

    x_dot = x_dot + params.dt * q_acc_0
    x = x + params.dt * x_dot
    theta_dot_1 = theta_dot_1 + params.dt * q_acc_1
    theta_dot_2 = theta_dot_2 + params.dt * q_acc_2
    theta_1 = wrap_angles_numpy(theta_1 + params.dt * theta_dot_1)
    theta_2 = wrap_angles_numpy(theta_2 + params.dt * theta_dot_2)
    return x, x_dot, theta_1, theta_2, theta_dot_1, theta_dot_2


def step_double_state_numpy(
    state: np.ndarray,
    force: float,
    params: CartPoleParams,
) -> np.ndarray:
    x, x_dot, theta_1, theta_2, theta_dot_1, theta_dot_2 = state.astype(np.float64)
    (
        next_x,
        next_x_dot,
        next_theta_1,
        next_theta_2,
        next_theta_dot_1,
        next_theta_dot_2,
    ) = step_double_soa_numpy(
        np.asarray([x], dtype=np.float64),
        np.asarray([x_dot], dtype=np.float64),
        np.asarray([theta_1], dtype=np.float64),
        np.asarray([theta_2], dtype=np.float64),
        np.asarray([theta_dot_1], dtype=np.float64),
        np.asarray([theta_dot_2], dtype=np.float64),
        np.asarray([force], dtype=np.float64),
        params,
    )
    return np.asarray(
        [
            next_x[0],
            next_x_dot[0],
            next_theta_1[0],
            next_theta_2[0],
            next_theta_dot_1[0],
            next_theta_dot_2[0],
        ],
        dtype=np.float64,
    )


def wrap_angles_numpy(theta: np.ndarray) -> np.ndarray:
    return (theta + np.pi) % (2 * np.pi) - np.pi


def rollout_double_actions(
    initial_state: np.ndarray,
    actions: list[int],
    params: CartPoleParams,
) -> np.ndarray:
    states = np.zeros((len(actions) + 1, 6), dtype=np.float32)
    state = np.asarray(initial_state, dtype=np.float64)
    states[0] = state.astype(np.float32)
    for idx, action in enumerate(actions, start=1):
        force = (int(action) - 1) * params.force_mag
        state = step_double_state_numpy(state, force, params)
        states[idx] = state.astype(np.float32)
    return states


def feedback_action(
    obs: np.ndarray,
    step: int,
    plan: TrajectoryPlan,
    args: argparse.Namespace,
    params: CartPoleParams,
    feedback_active: bool,
) -> int:
    state = np.asarray(obs, dtype=np.float64).copy()
    state[2:4] = wrap_angles_numpy(state[2:4])

    if feedback_active or step >= len(plan.actions):
        force = -float(LINEAR_FEEDBACK_GAIN @ state)
        return quantize_force(force, params.force_mag, args.feedback_force_deadzone)

    action = plan.actions[step]
    if not args.feedback_track_plan:
        return int(action)

    ref = plan.states[min(step, len(plan.states) - 1)].astype(np.float64)
    error = state - ref
    error[2:4] = wrap_angles_numpy(error[2:4])
    planned_force = (int(action) - 1) * params.force_mag
    tracking_gain = np.asarray(args.feedback_tracking_gain, dtype=np.float64)
    force = planned_force - float(tracking_gain @ error)
    return quantize_force(force, params.force_mag, args.feedback_force_deadzone)


def should_activate_feedback(obs: np.ndarray, args: argparse.Namespace) -> bool:
    state = np.asarray(obs, dtype=np.float64)
    theta = wrap_angles_numpy(state[2:4])
    return bool(
        abs(float(state[0])) <= args.feedback_switch_x_threshold
        and abs(float(state[1])) <= args.feedback_switch_x_dot_threshold
        and np.all(np.abs(theta) <= args.feedback_switch_theta_threshold)
        and np.all(np.abs(state[4:6]) <= args.feedback_switch_theta_dot_threshold)
    )


def quantize_force(force: float, force_mag: float, deadzone: float) -> int:
    force = float(np.clip(force, -force_mag, force_mag))
    if force > deadzone:
        return 2
    if force < -deadzone:
        return 0
    return 1
