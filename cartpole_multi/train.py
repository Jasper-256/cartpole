from __future__ import annotations

import argparse
import math
import time
from dataclasses import dataclass

import numpy as np
import torch
from torch import nn
from torch.distributions.categorical import Categorical

from cartpole_multi.batched_env import BatchedMultiPendulumCartPoleEnv
from cartpole_multi.config import NUM_PENDULUMS
from cartpole_multi.env import MultiPendulumCartPoleEnv
from cartpole_multi.observations import encode_observations, encoded_observation_dim
from cartpole_multi.puffer_env import make_vec_env
from cartpole_multi.video import open_video, record_action_video, record_policy_video


@dataclass
class TrainResult:
    num_pendulums: int
    total_timesteps: int
    updates: int
    steps_per_second: float
    last_mean_reward: float
    last_mean_length: float
    stable_timesteps: int
    stable_rate: float
    eval_mean_reward: float
    eval_mean_length: float
    eval_stable_timesteps: int
    eval_stable_rate: float
    video_path: str | None


@dataclass
class CemRun:
    actions: list[int]
    steps: int
    stable_steps: int
    max_consecutive_stable_steps: int
    first_stable_step: int | None
    total_return: float
    terminated: bool
    truncated: bool
    solved: bool


class ActorCritic(nn.Module):
    def __init__(self, obs_dim: int, action_dim: int) -> None:
        super().__init__()
        self.net = nn.Sequential(
            nn.Linear(obs_dim, 64),
            nn.Tanh(),
            nn.Linear(64, 64),
            nn.Tanh(),
        )
        self.actor = nn.Linear(64, action_dim)
        self.critic = nn.Linear(64, 1)

    def forward(self, obs: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor]:
        hidden = self.net(obs)
        return self.actor(hidden), self.critic(hidden).squeeze(-1)

    def get_action_and_value(
        self,
        obs: torch.Tensor,
        action: torch.Tensor | None = None,
    ) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor]:
        logits, value = self(obs)
        dist = Categorical(logits=logits)
        if action is None:
            action = dist.sample()
        return action, dist.log_prob(action), dist.entropy(), value


def train(args: argparse.Namespace) -> TrainResult:
    if args.optimizer == "cem":
        return train_cem(args)

    torch.manual_seed(args.seed)
    np.random.seed(args.seed)

    device = torch.device(args.device)
    envs = make_vec_env(
        num_pendulums=args.num_pendulums,
        num_envs=args.num_envs,
        num_workers=args.num_workers,
        backend=args.backend,
        reset_mode=args.reset_mode,
        seed=args.seed,
    )
    obs, _infos = envs.reset()
    obs = np.asarray(obs, dtype=np.float32)

    obs_dim = encoded_observation_dim(args.num_pendulums, args.observation_mode)
    action_dim = 3
    model = ActorCritic(obs_dim, action_dim).to(device)
    optimizer = torch.optim.Adam(model.parameters(), lr=args.learning_rate, eps=1e-5)

    rollout_steps = min(
        args.rollout_steps,
        max(1, math.ceil(args.total_timesteps / args.num_envs)),
    )
    rollout_size = args.num_envs * rollout_steps
    updates = max(1, math.ceil(args.total_timesteps / rollout_size))
    minibatch_size = min(args.minibatch_size, rollout_size)

    obs_buf = torch.zeros((rollout_steps, args.num_envs, obs_dim), device=device)
    actions_buf = torch.zeros((rollout_steps, args.num_envs), device=device, dtype=torch.long)
    logprobs_buf = torch.zeros((rollout_steps, args.num_envs), device=device)
    rewards_buf = torch.zeros((rollout_steps, args.num_envs), device=device)
    dones_buf = torch.zeros((rollout_steps, args.num_envs), device=device)
    values_buf = torch.zeros((rollout_steps, args.num_envs), device=device)

    episode_returns = np.zeros(args.num_envs, dtype=np.float32)
    episode_lengths = np.zeros(args.num_envs, dtype=np.int32)
    recent_returns: list[float] = []
    recent_lengths: list[int] = []
    stable_timesteps = 0
    recent_stable_timesteps = 0
    recent_total_timesteps = 0
    start = time.time()
    global_step = 0

    for update in range(1, updates + 1):
        update_stable_timesteps = 0
        for step in range(rollout_steps):
            global_step += args.num_envs
            policy_obs = encode_observations(
                obs,
                args.num_pendulums,
                args.observation_mode,
            )
            obs_tensor = torch.as_tensor(policy_obs.reshape(args.num_envs, obs_dim), device=device)
            obs_buf[step] = obs_tensor
            with torch.no_grad():
                action, logprob, _entropy, value = model.get_action_and_value(obs_tensor)

            next_obs, reward, terminated, truncated, _infos = envs.step(action.cpu().numpy())
            done = np.logical_or(terminated, truncated)

            actions_buf[step] = action
            logprobs_buf[step] = logprob
            rewards_buf[step] = torch.as_tensor(reward, dtype=torch.float32, device=device)
            dones_buf[step] = torch.as_tensor(done, dtype=torch.float32, device=device)
            values_buf[step] = value

            episode_returns += np.asarray(reward, dtype=np.float32)
            episode_lengths += 1
            stable_mask = stable_observations(
                np.asarray(next_obs, dtype=np.float32),
                args.num_pendulums,
                args.stable_x_threshold,
                args.stable_theta_threshold,
                args.stable_theta_dot_threshold,
            )
            stable_count = int(np.sum(stable_mask))
            stable_timesteps += stable_count
            update_stable_timesteps += stable_count
            for idx in np.flatnonzero(done):
                recent_returns.append(float(episode_returns[idx]))
                recent_lengths.append(int(episode_lengths[idx]))
                episode_returns[idx] = 0.0
                episode_lengths[idx] = 0
            recent_returns = recent_returns[-100:]
            recent_lengths = recent_lengths[-100:]
            obs = np.asarray(next_obs, dtype=np.float32)

        with torch.no_grad():
            policy_obs = encode_observations(
                obs,
                args.num_pendulums,
                args.observation_mode,
            )
            next_obs_tensor = torch.as_tensor(
                policy_obs.reshape(args.num_envs, obs_dim),
                device=device,
            )
            next_value = model(next_obs_tensor)[1]
            advantages = torch.zeros_like(rewards_buf, device=device)
            lastgaelam = torch.zeros(args.num_envs, device=device)
            for t in reversed(range(rollout_steps)):
                if t == rollout_steps - 1:
                    next_non_terminal = 1.0 - dones_buf[t]
                    next_values = next_value
                else:
                    next_non_terminal = 1.0 - dones_buf[t + 1]
                    next_values = values_buf[t + 1]
                delta = (
                    rewards_buf[t]
                    + args.gamma * next_values * next_non_terminal
                    - values_buf[t]
                )
                lastgaelam = delta + args.gamma * args.gae_lambda * next_non_terminal * lastgaelam
                advantages[t] = lastgaelam
            returns = advantages + values_buf

        batch_obs = obs_buf.reshape((-1, obs_dim))
        batch_actions = actions_buf.reshape(-1)
        batch_logprobs = logprobs_buf.reshape(-1)
        batch_advantages = advantages.reshape(-1)
        batch_returns = returns.reshape(-1)
        batch_values = values_buf.reshape(-1)

        batch_advantages = (batch_advantages - batch_advantages.mean()) / (
            batch_advantages.std() + 1e-8
        )

        indices = np.arange(rollout_size)
        for _epoch in range(args.update_epochs):
            np.random.shuffle(indices)
            for start_idx in range(0, rollout_size, minibatch_size):
                mb_idx = indices[start_idx : start_idx + minibatch_size]
                _, new_logprob, entropy, new_value = model.get_action_and_value(
                    batch_obs[mb_idx], batch_actions[mb_idx]
                )
                logratio = new_logprob - batch_logprobs[mb_idx]
                ratio = logratio.exp()
                mb_advantages = batch_advantages[mb_idx]
                policy_loss_1 = -mb_advantages * ratio
                policy_loss_2 = -mb_advantages * torch.clamp(
                    ratio, 1 - args.clip_coef, 1 + args.clip_coef
                )
                policy_loss = torch.max(policy_loss_1, policy_loss_2).mean()

                value_loss_unclipped = (new_value - batch_returns[mb_idx]) ** 2
                value_clipped = batch_values[mb_idx] + torch.clamp(
                    new_value - batch_values[mb_idx],
                    -args.clip_coef,
                    args.clip_coef,
                )
                value_loss_clipped = (value_clipped - batch_returns[mb_idx]) ** 2
                value_loss = 0.5 * torch.max(value_loss_unclipped, value_loss_clipped).mean()
                entropy_loss = entropy.mean()

                loss = policy_loss - args.ent_coef * entropy_loss + args.vf_coef * value_loss
                optimizer.zero_grad()
                loss.backward()
                nn.utils.clip_grad_norm_(model.parameters(), args.max_grad_norm)
                optimizer.step()

        recent_stable_timesteps += update_stable_timesteps
        recent_total_timesteps += rollout_size
        if recent_total_timesteps > args.stable_window_timesteps:
            recent_stable_timesteps = update_stable_timesteps
            recent_total_timesteps = rollout_size

        if args.log_every and update % args.log_every == 0:
            elapsed = max(time.time() - start, 1e-9)
            sps = int(global_step / elapsed)
            mean_reward = float(np.mean(recent_returns)) if recent_returns else 0.0
            mean_length = float(np.mean(recent_lengths)) if recent_lengths else 0.0
            stable_rate = stable_timesteps / max(global_step, 1)
            recent_stable_rate = recent_stable_timesteps / max(recent_total_timesteps, 1)
            print(
                f"update={update}/{updates} pendulums={args.num_pendulums} "
                f"steps={global_step} sps={sps} "
                f"mean_return={mean_reward:.2f} mean_len={mean_length:.1f} "
                f"stable_steps={stable_timesteps} stable_rate={stable_rate:.3f} "
                f"recent_stable_rate={recent_stable_rate:.3f}"
            )

    train_elapsed = max(time.time() - start, 1e-9)

    close = getattr(envs, "close", None)
    if close is not None:
        close()

    eval_mean_reward, eval_mean_length, eval_stable_timesteps, eval_stable_rate = evaluate_policy(
        model,
        obs_dim,
        args,
    )
    print(
        "eval "
        f"reset_mode={args.eval_reset_mode} "
        f"mean_return={eval_mean_reward:.2f} "
        f"mean_len={eval_mean_length:.1f} "
        f"stable_steps={eval_stable_timesteps} "
        f"stable_rate={eval_stable_rate:.3f}"
    )

    video_path = None
    if args.video:
        video_path = record_policy_video(model, obs_dim, args)
        print(f"saved_video={video_path}")
        if args.open_video:
            open_video(video_path)

    return TrainResult(
        num_pendulums=args.num_pendulums,
        total_timesteps=global_step,
        updates=updates,
        steps_per_second=global_step / train_elapsed,
        last_mean_reward=float(np.mean(recent_returns)) if recent_returns else 0.0,
        last_mean_length=float(np.mean(recent_lengths)) if recent_lengths else 0.0,
        stable_timesteps=stable_timesteps,
        stable_rate=stable_timesteps / max(global_step, 1),
        eval_mean_reward=eval_mean_reward,
        eval_mean_length=eval_mean_length,
        eval_stable_timesteps=eval_stable_timesteps,
        eval_stable_rate=eval_stable_rate,
        video_path=video_path,
    )


class CemPlanner:
    def __init__(self, args: argparse.Namespace) -> None:
        self.args = args
        self.num_pendulums = args.num_pendulums
        self.num_envs = args.cem_planner_envs
        self.horizon = args.cem_planner_horizon
        self.iterations = args.cem_planner_iterations
        self.elite_frac = args.cem_elite_frac
        self.rng = np.random.default_rng(args.cem_planner_seed)

    def action(
        self,
        state: np.ndarray,
        previous_probs: np.ndarray | None,
    ) -> tuple[int, np.ndarray]:
        probs = (
            np.full((self.horizon, 3), 1.0 / 3.0, dtype=np.float64)
            if previous_probs is None
            else previous_probs.copy()
        )
        best_sequence: np.ndarray | None = None
        best_score = -np.inf

        for _iteration in range(self.iterations):
            sampled = sample_action_sequences(self.rng, self.num_envs, self.horizon, probs)
            scores = score_action_sequences(
                sampled,
                np.asarray(state, dtype=np.float32),
                self.args,
                terminal_penalty=8000.0,
                theta_dot_cost=self.args.cem_planner_theta_dot_cost,
                x_cost=self.args.cem_planner_x_cost,
                x_dot_cost=self.args.cem_planner_x_dot_cost,
                cost_weight=self.args.cem_planner_cost_weight,
                stable_x_threshold=self.args.cem_planner_stable_x_threshold,
                stable_x_dot_threshold=self.args.cem_planner_stable_x_dot_threshold,
            )
            best_idx = int(np.argmax(scores))
            if scores[best_idx] > best_score:
                best_score = float(scores[best_idx])
                best_sequence = sampled[best_idx].copy()
            probs = update_action_probs(sampled, scores, self.elite_frac, probs)

        if best_sequence is None:
            raise RuntimeError("CEM planner did not sample any action sequences")

        shifted_probs = np.vstack([probs[1:], np.full((1, 3), 1.0 / 3.0)])
        return int(best_sequence[0]), shifted_probs


def train_cem(args: argparse.Namespace) -> TrainResult:
    start = time.time()
    swingup_actions = optimize_open_loop_actions(args)
    run = evaluate_cem_actions(swingup_actions, args)
    elapsed = max(time.time() - start, 1e-9)

    print(
        "eval "
        f"optimizer=cem reset_mode={args.eval_reset_mode} "
        f"return={run.total_return:.2f} "
        f"steps={run.steps} "
        f"stable_steps={run.stable_steps} "
        f"max_consecutive_stable_steps={run.max_consecutive_stable_steps} "
        f"first_stable_step={run.first_stable_step} "
        f"solved={run.solved}"
    )

    video_path = None
    if args.video:
        video_path = record_action_video(run.actions, args, suffix="cem")
        print(f"saved_video={video_path}")
        if args.open_video:
            open_video(video_path)

    return TrainResult(
        num_pendulums=args.num_pendulums,
        total_timesteps=run.steps,
        updates=args.cem_iterations,
        steps_per_second=run.steps / elapsed,
        last_mean_reward=run.total_return,
        last_mean_length=float(run.steps),
        stable_timesteps=run.stable_steps,
        stable_rate=run.stable_steps / max(run.steps, 1),
        eval_mean_reward=run.total_return,
        eval_mean_length=float(run.steps),
        eval_stable_timesteps=run.stable_steps,
        eval_stable_rate=run.stable_steps / max(run.steps, 1),
        video_path=video_path,
    )


def optimize_open_loop_actions(args: argparse.Namespace) -> list[int]:
    rng = np.random.default_rng(args.cem_seed)
    base_env = MultiPendulumCartPoleEnv(
        num_pendulums=args.num_pendulums,
        reset_mode=args.eval_reset_mode,
        seed=args.seed,
    )
    initial_state, _info = base_env.reset()
    probs = np.full((args.cem_segments, 3), 1.0 / 3.0, dtype=np.float64)
    best_segments: np.ndarray | None = None
    best_score = -np.inf

    for iteration in range(1, args.cem_iterations + 1):
        sampled = sample_action_sequences(
            rng,
            args.cem_envs,
            args.cem_segments,
            probs,
        )
        repeated = np.repeat(sampled, args.cem_action_repeat, axis=1)
        scores = score_action_sequences(
            repeated,
            np.asarray(initial_state, dtype=np.float32),
            args,
            terminal_penalty=2000.0,
            theta_dot_cost=args.cem_theta_dot_cost,
            x_cost=args.cem_x_cost,
            x_dot_cost=args.cem_x_dot_cost,
            cost_weight=args.cem_cost_weight,
            stable_x_threshold=args.stable_x_threshold,
            stable_x_dot_threshold=np.inf,
        )
        best_idx = int(np.argmax(scores))
        if scores[best_idx] > best_score:
            best_score = float(scores[best_idx])
            best_segments = sampled[best_idx].copy()

        probs = update_action_probs(sampled, scores, args.cem_elite_frac, probs)

        if args.log_every and iteration % args.log_every == 0:
            print(
                f"cem_update={iteration}/{args.cem_iterations} "
                f"best_score={float(scores[best_idx]):.1f} "
                f"global_best_score={best_score:.1f}"
            )

    if best_segments is None:
        raise RuntimeError("CEM did not sample any action sequences")
    return [
        int(action)
        for action in best_segments
        for _repeat in range(args.cem_action_repeat)
    ]


def evaluate_cem_actions(initial_actions: list[int], args: argparse.Namespace) -> CemRun:
    env = MultiPendulumCartPoleEnv(
        num_pendulums=args.num_pendulums,
        reset_mode=args.eval_reset_mode,
        seed=args.seed,
    )
    obs, info = env.reset()
    planner = CemPlanner(args)
    planner_probs: np.ndarray | None = None
    actions: list[int] = []
    total_return = 0.0
    stable_steps = 0
    consecutive_stable_steps = 0
    max_consecutive_stable_steps = 0
    first_stable_step: int | None = None
    terminated = False
    truncated = False
    use_planner = False

    for step in range(args.video_steps):
        if not use_planner and step < len(initial_actions):
            action = initial_actions[step]
        else:
            use_planner = True
            action, planner_probs = planner.action(obs, planner_probs)

        obs, reward, terminated, truncated, info = env.step(action)
        actions.append(action)
        total_return += reward

        if not use_planner and should_switch_to_planner(obs, info, args):
            use_planner = True
            planner_probs = None

        if info["stable"]:
            stable_steps += 1
            consecutive_stable_steps += 1
            max_consecutive_stable_steps = max(
                max_consecutive_stable_steps,
                consecutive_stable_steps,
            )
            if first_stable_step is None:
                first_stable_step = step
        else:
            consecutive_stable_steps = 0

        if args.log_every and (
            step % args.cem_eval_log_every == 0 or terminated or truncated
        ):
            print(
                f"step={step} action={action} stable_steps={stable_steps} "
                f"x={obs[0]:+.3f} theta={obs[2:2 + args.num_pendulums]} "
                f"theta_dot={obs[2 + args.num_pendulums:]}"
            )

        if terminated or truncated:
            break

    solved = stable_steps >= args.solve_stable_steps
    return CemRun(
        actions=actions,
        steps=step + 1,
        stable_steps=stable_steps,
        max_consecutive_stable_steps=max_consecutive_stable_steps,
        first_stable_step=first_stable_step,
        total_return=total_return,
        terminated=terminated,
        truncated=truncated,
        solved=solved,
    )


def sample_action_sequences(
    rng: np.random.Generator,
    num_sequences: int,
    horizon: int,
    probs: np.ndarray,
) -> np.ndarray:
    uniform = rng.random((num_sequences, horizon))
    cumulative = np.cumsum(probs, axis=1)
    return (
        (uniform > cumulative[:, 0]).astype(np.int8)
        + (uniform > cumulative[:, 1]).astype(np.int8)
    )


def update_action_probs(
    sequences: np.ndarray,
    scores: np.ndarray,
    elite_frac: float,
    previous_probs: np.ndarray,
) -> np.ndarray:
    elite_count = max(16, int(sequences.shape[0] * elite_frac))
    elite = sequences[np.argpartition(scores, -elite_count)[-elite_count:]]
    counts = np.stack([(elite == action).mean(axis=0) for action in range(3)], axis=1)
    probs = 0.45 * previous_probs + 0.55 * counts
    probs = 0.02 / 3.0 + 0.98 * probs
    return probs / probs.sum(axis=1, keepdims=True)


def score_action_sequences(
    sequences: np.ndarray,
    initial_state: np.ndarray,
    args: argparse.Namespace,
    terminal_penalty: float,
    theta_dot_cost: float,
    x_cost: float,
    x_dot_cost: float,
    cost_weight: float,
    stable_x_threshold: float,
    stable_x_dot_threshold: float,
) -> np.ndarray:
    num_sequences, horizon = sequences.shape
    env = BatchedMultiPendulumCartPoleEnv(
        num_pendulums=args.num_pendulums,
        num_envs=num_sequences,
        reset_mode=args.eval_reset_mode,
        seed=args.seed + 999,
    )
    env.reset()
    env.state[:] = initial_state
    env.step_count[:] = 0

    alive = np.ones(num_sequences, dtype=bool)
    score = np.zeros(num_sequences, dtype=np.float64)
    stable_steps = np.zeros(num_sequences, dtype=np.int32)
    min_cost = np.full(num_sequences, np.inf, dtype=np.float64)
    final_cost = np.zeros(num_sequences, dtype=np.float64)

    for horizon_step in range(horizon):
        obs, _reward, terminated, truncated, _info = env.step(
            sequences[:, horizon_step].astype(np.int64)
        )
        theta = obs[:, 2 : 2 + args.num_pendulums].astype(np.float64)
        theta_dot = obs[:, 2 + args.num_pendulums :].astype(np.float64)
        x = obs[:, 0].astype(np.float64)
        x_dot = obs[:, 1].astype(np.float64)

        angle_cost = np.mean(theta**2, axis=1)
        velocity_cost = np.mean(theta_dot**2, axis=1)
        cost = (
            angle_cost
            + theta_dot_cost * velocity_cost
            + x_cost * x**2
            + x_dot_cost * x_dot**2
        )
        stable = (
            (np.abs(x) <= stable_x_threshold)
            & (np.abs(x_dot) <= stable_x_dot_threshold)
            & np.all(np.abs(theta) <= args.stable_theta_threshold, axis=1)
            & np.all(np.abs(theta_dot) <= args.stable_theta_dot_threshold, axis=1)
            & alive
        )

        stable_steps += stable.astype(np.int32)
        min_cost = np.minimum(min_cost, cost)
        score += alive * (
            args.cem_upright_reward * np.mean(np.cos(theta), axis=1)
            - cost_weight * cost
        )
        done = (terminated | truncated) & alive
        score[done] -= terminal_penalty
        alive &= ~done
        final_cost = cost

    score += (
        args.cem_stable_bonus * stable_steps
        - args.cem_min_cost_weight * min_cost
        - args.cem_final_cost_weight * final_cost
        + args.cem_alive_bonus * alive
    )
    return score


def should_switch_to_planner(obs: np.ndarray, info: dict, args: argparse.Namespace) -> bool:
    theta = obs[2 : 2 + args.num_pendulums]
    return bool(
        info["stable"]
        or (
            abs(float(obs[0])) < args.cem_switch_x_threshold
            and np.all(np.abs(theta) < args.cem_switch_theta_threshold)
        )
    )


def evaluate_policy(
    model: ActorCritic,
    obs_dim: int,
    args: argparse.Namespace,
) -> tuple[float, float, int, float]:
    if args.eval_episodes <= 0:
        return 0.0, 0.0, 0, 0.0

    env = MultiPendulumCartPoleEnv(
        num_pendulums=args.num_pendulums,
        reset_mode=args.eval_reset_mode,
        seed=args.seed + 20_000,
    )
    device = next(model.parameters()).device
    returns: list[float] = []
    lengths: list[int] = []
    stable_timesteps = 0
    total_timesteps = 0
    model.eval()
    with torch.no_grad():
        for episode in range(args.eval_episodes):
            obs, _info = env.reset(seed=args.seed + 20_000 + episode)
            episode_return = 0.0
            episode_length = 0
            for _step in range(args.eval_steps):
                policy_obs = encode_observations(
                    obs,
                    args.num_pendulums,
                    args.observation_mode,
                )
                obs_tensor = torch.as_tensor(
                    policy_obs.reshape(1, obs_dim),
                    dtype=torch.float32,
                    device=device,
                )
                logits, _value = model(obs_tensor)
                action = int(torch.argmax(logits, dim=-1).item())
                obs, reward, terminated, truncated, info = env.step(action)
                episode_return += float(reward)
                episode_length += 1
                stable_timesteps += int(info["stable"])
                total_timesteps += 1
                if terminated or truncated:
                    break
            returns.append(episode_return)
            lengths.append(episode_length)

    return (
        float(np.mean(returns)) if returns else 0.0,
        float(np.mean(lengths)) if lengths else 0.0,
        stable_timesteps,
        stable_timesteps / max(total_timesteps, 1),
    )


def stable_observations(
    obs: np.ndarray,
    num_pendulums: int,
    x_threshold: float,
    theta_threshold: float,
    theta_dot_threshold: float,
) -> np.ndarray:
    x = obs[:, 0]
    theta = obs[:, 2 : 2 + num_pendulums]
    theta_dot = obs[:, 2 + num_pendulums :]
    return (
        (np.abs(x) <= x_threshold)
        & np.all(np.abs(theta) <= theta_threshold, axis=1)
        & np.all(np.abs(theta_dot) <= theta_dot_threshold, axis=1)
    )


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument("--num-pendulums", type=int, default=NUM_PENDULUMS)
    parser.add_argument("--total-timesteps", type=int, default=131_072)
    parser.add_argument("--num-envs", type=int, default=1024)
    parser.add_argument("--num-workers", type=int, default=4)
    parser.add_argument(
        "--backend",
        choices=["numpy", "serial", "multiprocessing"],
        default="numpy",
    )
    parser.add_argument("--optimizer", choices=["cem", "ppo"], default="cem")
    parser.add_argument(
        "--reset-mode",
        choices=["downward", "upright", "uniform", "mixed"],
        default="downward",
    )
    parser.add_argument(
        "--eval-reset-mode",
        choices=["downward", "upright", "uniform", "mixed"],
        default="downward",
    )
    parser.add_argument("--eval-episodes", type=int, default=4)
    parser.add_argument("--eval-steps", type=int, default=500)
    parser.add_argument("--observation-mode", choices=["raw", "trig"], default="trig")
    parser.add_argument("--rollout-steps", type=int, default=128)
    parser.add_argument("--minibatch-size", type=int, default=16384)
    parser.add_argument("--update-epochs", type=int, default=2)
    parser.add_argument("--learning-rate", type=float, default=2.5e-4)
    parser.add_argument("--gamma", type=float, default=0.99)
    parser.add_argument("--gae-lambda", type=float, default=0.95)
    parser.add_argument("--clip-coef", type=float, default=0.2)
    parser.add_argument("--ent-coef", type=float, default=0.01)
    parser.add_argument("--vf-coef", type=float, default=0.5)
    parser.add_argument("--max-grad-norm", type=float, default=0.5)
    parser.add_argument("--seed", type=int, default=123)
    parser.add_argument("--device", default="cpu")
    parser.add_argument("--log-every", type=int, default=1)
    parser.add_argument("--stable-x-threshold", type=float, default=0.5)
    parser.add_argument("--stable-theta-threshold", type=float, default=float(np.deg2rad(12.0)))
    parser.add_argument("--stable-theta-dot-threshold", type=float, default=1.0)
    parser.add_argument("--stable-window-timesteps", type=int, default=4096)
    parser.add_argument("--solve-stable-steps", type=int, default=100)
    parser.add_argument("--cem-seed", type=int, default=42)
    parser.add_argument("--cem-envs", type=int, default=8192)
    parser.add_argument("--cem-segments", type=int, default=90)
    parser.add_argument("--cem-action-repeat", type=int, default=5)
    parser.add_argument("--cem-iterations", type=int, default=40)
    parser.add_argument("--cem-elite-frac", type=float, default=0.04)
    parser.add_argument("--cem-upright-reward", type=float, default=3.0)
    parser.add_argument("--cem-cost-weight", type=float, default=0.0)
    parser.add_argument("--cem-theta-dot-cost", type=float, default=0.08)
    parser.add_argument("--cem-x-cost", type=float, default=0.15)
    parser.add_argument("--cem-x-dot-cost", type=float, default=0.02)
    parser.add_argument("--cem-stable-bonus", type=float, default=100000.0)
    parser.add_argument("--cem-min-cost-weight", type=float, default=25000.0)
    parser.add_argument("--cem-final-cost-weight", type=float, default=5000.0)
    parser.add_argument("--cem-alive-bonus", type=float, default=500.0)
    parser.add_argument("--cem-switch-x-threshold", type=float, default=0.8)
    parser.add_argument("--cem-switch-theta-threshold", type=float, default=0.3)
    parser.add_argument("--cem-eval-log-every", type=int, default=25)
    parser.add_argument("--cem-planner-seed", type=int, default=8)
    parser.add_argument("--cem-planner-envs", type=int, default=1024)
    parser.add_argument("--cem-planner-horizon", type=int, default=70)
    parser.add_argument("--cem-planner-iterations", type=int, default=3)
    parser.add_argument("--cem-planner-stable-x-threshold", type=float, default=0.35)
    parser.add_argument("--cem-planner-stable-x-dot-threshold", type=float, default=1.2)
    parser.add_argument("--cem-planner-cost-weight", type=float, default=90.0)
    parser.add_argument("--cem-planner-theta-dot-cost", type=float, default=0.08)
    parser.add_argument("--cem-planner-x-cost", type=float, default=1.5)
    parser.add_argument("--cem-planner-x-dot-cost", type=float, default=0.35)
    parser.add_argument("--video", action=argparse.BooleanOptionalAction, default=True)
    parser.add_argument("--open-video", action=argparse.BooleanOptionalAction, default=True)
    parser.add_argument("--video-dir", default="videos")
    parser.add_argument("--video-steps", type=int, default=500)
    parser.add_argument("--video-fps", type=int, default=50)
    parser.add_argument("--video-width", type=int, default=800)
    parser.add_argument("--video-height", type=int, default=450)
    return parser.parse_args()


def main() -> None:
    result = train(parse_args())
    print(
        "finished "
        f"pendulums={result.num_pendulums} "
        f"timesteps={result.total_timesteps} "
        f"updates={result.updates} "
        f"sps={result.steps_per_second:.0f} "
        f"mean_return={result.last_mean_reward:.2f} "
        f"mean_len={result.last_mean_length:.1f} "
        f"stable_steps={result.stable_timesteps} "
        f"stable_rate={result.stable_rate:.3f} "
        f"eval_stable_steps={result.eval_stable_timesteps} "
        f"eval_stable_rate={result.eval_stable_rate:.3f} "
        f"video={result.video_path}"
    )


if __name__ == "__main__":
    main()
