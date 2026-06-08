from __future__ import annotations

import argparse
import re
import subprocess
import tempfile
import time
from dataclasses import dataclass, replace
from pathlib import Path

import numpy as np
import torch

import pufferlib.pytorch
from cartpole_multi.config import NUM_PENDULUMS
from cartpole_multi.env import CartPoleParams
from cartpole_multi.policy import CartPolePolicy
from cartpole_multi.torch_env import (
    RESET_MODES,
    TorchMultiPendulumCartPoleEnv,
    observation_dim,
    resolve_torch_device,
    synchronize_device,
)
from cartpole_multi.video import open_video, record_policy_video


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
    rollout_controlled_upright: float
    eval_mean_reward: float
    eval_mean_length: float
    eval_mean_height: float
    eval_controlled_upright: float
    eval_top_theta_speed: float
    eval_stable_timesteps: int
    eval_stable_rate: float
    checkpoint_path: str
    video_path: str | None


def train(args: argparse.Namespace) -> TrainResult:
    params = params_from_args(args)
    device = resolve_torch_device(args.device)
    if device.type == "cpu":
        raise RuntimeError(
            "Training requires CUDA or MPS. Run with --device gpu/auto/mps/cuda; "
            "CPU is only used by the post-training video renderer."
        )
    torch.set_float32_matmul_precision("high")
    torch.manual_seed(args.seed)
    if args.backend == "metal-es":
        if device.type != "mps":
            raise RuntimeError("backend=metal-es requires Apple Metal/MPS")
        return train_metal_es(args, params, device)
    if args.backend != "ppo":
        raise ValueError(f"unknown backend {args.backend}")

    env = TorchMultiPendulumCartPoleEnv(
        num_envs=args.num_envs,
        num_pendulums=args.num_pendulums,
        params=params,
        reset_mode=args.reset_mode,
        device=device,
        seed=args.seed,
        autoreset=args.autoreset,
    )
    policy = CartPolePolicy(
        observation_size=observation_dim(args.num_pendulums),
        action_size=3,
        hidden_size=args.hidden_size,
    ).to(device)
    if args.compile:
        policy = torch.compile(policy, mode=args.compile_mode)

    optimizer = make_optimizer(policy, args)
    batch_size = args.num_envs * args.rollout_steps
    num_updates = max(1, args.total_timesteps // batch_size)
    total_timesteps = num_updates * batch_size
    scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=num_updates)

    obs = env.reset(seed=args.seed)
    torch.manual_seed(args.seed)
    obs_buf = torch.zeros(
        (args.rollout_steps, args.num_envs, env.observation_dim),
        dtype=torch.float32,
        device=device,
    )
    actions_buf = torch.zeros(
        (args.rollout_steps, args.num_envs),
        dtype=torch.int64,
        device=device,
    )
    logprobs_buf = torch.zeros((args.rollout_steps, args.num_envs), device=device)
    rewards_buf = torch.zeros((args.rollout_steps, args.num_envs), device=device)
    dones_buf = torch.zeros((args.rollout_steps, args.num_envs), device=device)
    values_buf = torch.zeros((args.rollout_steps, args.num_envs), device=device)
    best_state: dict[str, torch.Tensor] | None = None
    best_score = torch.full((), -float("inf"), device=device)
    if args.select_best:
        best_state = {
            key: value.detach().clone()
            for key, value in policy.state_dict().items()
        }

    global_step = 0
    start = time.perf_counter()
    last_mean_reward = 0.0
    last_mean_length = 0.0
    last_stable_steps = 0
    last_stable_rate = 0.0
    last_controlled_upright = 0.0
    last_rollout_stats: dict[str, torch.Tensor] | None = None

    print(
        "train_start "
        f"pufferlib=3.0_style_ppo "
        f"device={device.type} "
        f"pendulums={args.num_pendulums} "
        f"num_envs={args.num_envs} "
        f"rollout_steps={args.rollout_steps} "
        f"batch_size={batch_size} "
        f"updates={num_updates} "
        f"optimizer={args.optimizer}",
        flush=True,
    )

    for update in range(1, num_updates + 1):
        rollout_stats = rollout(
            env,
            policy,
            obs,
            args,
            obs_buf,
            actions_buf,
            logprobs_buf,
            rewards_buf,
            dones_buf,
            values_buf,
        )
        obs = rollout_stats["obs"]
        last_rollout_stats = rollout_stats
        global_step += batch_size

        with torch.no_grad():
            _last_logits, last_value = policy.forward_eval(obs)
            advantages, returns = compute_gae(
                rewards_buf,
                dones_buf,
                values_buf,
                last_value.flatten(),
                gamma=args.gamma,
                gae_lambda=args.gae_lambda,
            )

        losses = update_policy(
            policy=policy,
            optimizer=optimizer,
            obs=obs_buf.reshape(-1, env.observation_dim),
            actions=actions_buf.reshape(-1),
            old_logprobs=logprobs_buf.reshape(-1),
            advantages=advantages.reshape(-1),
            returns=returns.reshape(-1),
            old_values=values_buf.reshape(-1),
            args=args,
        )
        if args.anneal_lr:
            scheduler.step()

        if args.select_best and (
            update % args.select_best_every == 0 or update == num_updates
        ):
            selection_score = score_policy(policy, args, params, device)
            better = selection_score > best_score
            best_score = torch.where(better, selection_score, best_score)
            assert best_state is not None
            state = policy.state_dict()
            for key, value in state.items():
                best_state[key].copy_(torch.where(better, value.detach(), best_state[key]))

        if args.log_every and (update % args.log_every == 0 or update == num_updates):
            synchronize_device(device)
            loss_values = tensor_dict_to_float(losses)
            elapsed = max(time.perf_counter() - start, 1e-9)
            done_count = max(int(rollout_stats["done_count"].item()), 1)
            last_mean_reward = float((rollout_stats["episode_return_sum"] / done_count).item())
            last_mean_length = float((rollout_stats["episode_length_sum"] / done_count).item())
            last_stable_steps = int(rollout_stats["stable_steps"].item())
            last_stable_rate = float(
                (rollout_stats["stable_steps"] / max(batch_size, 1)).item()
            )
            last_controlled_upright = float(
                (
                    rollout_stats["controlled_upright_sum"]
                    / torch.clamp(rollout_stats["active_steps"], min=1.0)
                ).item()
            )
            print(
                f"update={update}/{num_updates} "
                f"step={global_step} "
                f"sps={global_step / elapsed:.0f} "
                f"mean_return={last_mean_reward:.2f} "
                f"mean_len={last_mean_length:.1f} "
                f"stable_rate={last_stable_rate:.4f} "
                f"controlled_upright={last_controlled_upright:.4f} "
                f"policy_loss={loss_values['policy_loss']:.4f} "
                f"value_loss={loss_values['value_loss']:.4f} "
                f"entropy={loss_values['entropy']:.4f} "
                f"kl={loss_values['approx_kl']:.5f} "
                f"clipfrac={loss_values['clipfrac']:.3f} "
                f"lr={optimizer.param_groups[0]['lr']:.6g}",
                flush=True,
            )

    if last_rollout_stats is not None and not args.log_every:
        synchronize_device(device)
        done_count = max(int(last_rollout_stats["done_count"].item()), 1)
        last_mean_reward = float(
            (last_rollout_stats["episode_return_sum"] / done_count).item()
        )
        last_mean_length = float(
            (last_rollout_stats["episode_length_sum"] / done_count).item()
        )
        last_stable_steps = int(last_rollout_stats["stable_steps"].item())
        last_stable_rate = float(
            (last_rollout_stats["stable_steps"] / max(batch_size, 1)).item()
        )
        last_controlled_upright = float(
            (
                last_rollout_stats["controlled_upright_sum"]
                / torch.clamp(last_rollout_stats["active_steps"], min=1.0)
            ).item()
        )

    if best_state is not None:
        policy.load_state_dict(best_state)

    checkpoint_path = ""
    if args.checkpoint:
        checkpoint_path = save_checkpoint(policy, args, params, total_timesteps)
    eval_stats = evaluate_policy(policy, args, params, device)

    video_path = None
    if args.video:
        video_start = time.perf_counter()
        video_path = record_policy_video(policy, args, params, device, suffix="ppo")
        print(
            f"timing phase=video_total elapsed={time.perf_counter() - video_start:.3f}s",
            flush=True,
        )
        print(f"saved_video={video_path}", flush=True)
        if args.open_video:
            open_video(video_path)

    elapsed = max(time.perf_counter() - start, 1e-9)
    return TrainResult(
        num_pendulums=args.num_pendulums,
        total_timesteps=total_timesteps,
        updates=num_updates,
        steps_per_second=total_timesteps / elapsed,
        last_mean_reward=last_mean_reward,
        last_mean_length=last_mean_length,
        stable_timesteps=last_stable_steps,
        stable_rate=last_stable_rate,
        rollout_controlled_upright=last_controlled_upright,
        eval_mean_reward=eval_stats["mean_reward"],
        eval_mean_length=eval_stats["mean_length"],
        eval_mean_height=eval_stats["mean_height"],
        eval_controlled_upright=eval_stats["mean_controlled_upright"],
        eval_top_theta_speed=eval_stats["mean_top_theta_speed"],
        eval_stable_timesteps=int(eval_stats["stable_timesteps"]),
        eval_stable_rate=eval_stats["stable_rate"],
        checkpoint_path=checkpoint_path,
        video_path=video_path,
    )


def train_metal_es(
    args: argparse.Namespace,
    params: CartPoleParams,
    device: torch.device,
) -> TrainResult:
    if args.num_pendulums != 2:
        raise RuntimeError("backend=metal-es currently supports --num-pendulums 2")

    binary = ensure_metal_es_binary()
    with tempfile.NamedTemporaryFile(prefix="cartpole_metal_weights_", suffix=".bin") as weights_file:
        cmd = [
            str(binary),
            str(args.metal_envs),
            str(args.metal_rollout_steps),
            str(args.metal_iterations),
            str(args.metal_sigma),
            str(args.metal_learning_rate),
            weights_file.name,
        ]
        print(
            "train_start "
            f"backend=metal-es "
            f"device={device.type} "
            f"pendulums={args.num_pendulums} "
            f"num_envs={args.metal_envs} "
            f"rollout_steps={args.metal_rollout_steps} "
            f"updates={args.metal_iterations} "
            f"optimizer=es",
            flush=True,
        )
        process = subprocess.run(
            cmd,
            check=True,
            capture_output=True,
            text=True,
        )
        if process.stderr:
            print(process.stderr, end="", flush=True)
        print(process.stdout, end="", flush=True)
        stats = parse_metal_es_stdout(process.stdout)
        weights = np.fromfile(weights_file.name, dtype=np.float32)

    if weights.shape != (27,):
        raise RuntimeError(f"metal-es wrote {weights.shape[0]} weights, expected 27")

    policy = CartPolePolicy(
        observation_size=observation_dim(args.num_pendulums),
        action_size=3,
        hidden_size=0,
    ).to(device)
    load_metal_actor_weights(policy, weights, device)

    checkpoint_path = ""
    if args.checkpoint:
        checkpoint_path = save_checkpoint(policy, args, params, int(stats["steps"]))
    eval_stats = evaluate_policy(policy, args, params, device)

    video_path = None
    if args.video:
        video_start = time.perf_counter()
        video_path = record_policy_video(policy, args, params, device, suffix="metal_es")
        print(
            f"timing phase=video_total elapsed={time.perf_counter() - video_start:.3f}s",
            flush=True,
        )
        print(f"saved_video={video_path}", flush=True)
        if args.open_video:
            open_video(video_path)

    return TrainResult(
        num_pendulums=args.num_pendulums,
        total_timesteps=int(stats["steps"]),
        updates=args.metal_iterations,
        steps_per_second=stats["sps"],
        last_mean_reward=0.0,
        last_mean_length=0.0,
        stable_timesteps=0,
        stable_rate=0.0,
        rollout_controlled_upright=0.0,
        eval_mean_reward=eval_stats["mean_reward"],
        eval_mean_length=eval_stats["mean_length"],
        eval_mean_height=eval_stats["mean_height"],
        eval_controlled_upright=eval_stats["mean_controlled_upright"],
        eval_top_theta_speed=eval_stats["mean_top_theta_speed"],
        eval_stable_timesteps=int(eval_stats["stable_timesteps"]),
        eval_stable_rate=eval_stats["stable_rate"],
        checkpoint_path=checkpoint_path,
        video_path=video_path,
    )


def ensure_metal_es_binary() -> Path:
    root = Path(__file__).resolve().parents[1]
    source = root / "benchmarks" / "metal_es_train.m"
    binary = root / "benchmarks" / "metal_es_train"
    if not source.exists():
        raise RuntimeError(f"missing Metal ES source: {source}")
    needs_compile = (
        not binary.exists()
        or source.stat().st_mtime_ns > binary.stat().st_mtime_ns
    )
    if needs_compile:
        subprocess.run(
            [
                "xcrun",
                "clang",
                "-fobjc-arc",
                "-framework",
                "Foundation",
                "-framework",
                "Metal",
                str(source),
                "-o",
                str(binary),
            ],
            check=True,
        )
    return binary


def parse_metal_es_stdout(stdout: str) -> dict[str, float]:
    match = re.search(
        r"metal_es_train steps=([0-9]+) elapsed=([0-9.]+)s sps=([0-9.]+)",
        stdout,
    )
    if match is None:
        raise RuntimeError(f"could not parse metal-es output: {stdout!r}")
    return {
        "steps": float(match.group(1)),
        "elapsed": float(match.group(2)),
        "sps": float(match.group(3)),
    }


def load_metal_actor_weights(
    policy: CartPolePolicy,
    weights: np.ndarray,
    device: torch.device,
) -> None:
    actor = weights.reshape(3, 9)
    with torch.no_grad():
        policy.actor.weight.copy_(
            torch.as_tensor(actor[:, :8], dtype=torch.float32, device=device)
        )
        policy.actor.bias.copy_(
            torch.as_tensor(actor[:, 8], dtype=torch.float32, device=device)
        )
        policy.value_fn.weight.zero_()
        policy.value_fn.bias.zero_()


def rollout(
    env: TorchMultiPendulumCartPoleEnv,
    policy: torch.nn.Module,
    obs: torch.Tensor,
    args: argparse.Namespace,
    obs_buf: torch.Tensor,
    actions_buf: torch.Tensor,
    logprobs_buf: torch.Tensor,
    rewards_buf: torch.Tensor,
    dones_buf: torch.Tensor,
    values_buf: torch.Tensor,
) -> dict[str, torch.Tensor]:
    done_count = torch.zeros((), device=obs.device)
    episode_return_sum = torch.zeros((), device=obs.device)
    episode_length_sum = torch.zeros((), device=obs.device)
    stable_steps = torch.zeros((), device=obs.device)
    controlled_upright_sum = torch.zeros((), device=obs.device)
    active_steps = torch.zeros((), device=obs.device)

    for step in range(env_step_count(obs_buf)):
        obs_buf[step] = obs
        with torch.no_grad():
            logits, value = policy.forward_eval(obs)
            action, logprob, _entropy = pufferlib.pytorch.sample_logits(logits)
        env_step = env.step(action)
        done = env_step.terminated | env_step.truncated

        actions_buf[step] = action.long()
        logprobs_buf[step] = logprob
        rewards_buf[step] = env_step.reward
        dones_buf[step] = done.float()
        values_buf[step] = value.flatten()
        done_count += done.float().sum()
        episode_return_sum += env_step.info["episode_return"].sum()
        episode_length_sum += env_step.info["episode_length"].sum()
        stable_steps += env_step.info["stable"].float().sum()
        controlled_upright_sum += env_step.info["controlled_upright"].sum()
        active_steps += torch.ones_like(env_step.reward).sum()
        obs = env_step.observation

    return {
        "obs": obs,
        "done_count": done_count,
        "episode_return_sum": episode_return_sum,
        "episode_length_sum": episode_length_sum,
        "stable_steps": stable_steps,
        "controlled_upright_sum": controlled_upright_sum,
        "active_steps": active_steps,
    }


def env_step_count(obs_buf: torch.Tensor) -> int:
    return int(obs_buf.shape[0])


def deterministic_policy_action(
    policy: torch.nn.Module,
    obs: torch.Tensor,
) -> torch.Tensor:
    logits, _value = policy.forward_eval(obs)
    return torch.argmax(logits, dim=1)


def compute_gae(
    rewards: torch.Tensor,
    dones: torch.Tensor,
    values: torch.Tensor,
    last_value: torch.Tensor,
    *,
    gamma: float,
    gae_lambda: float,
) -> tuple[torch.Tensor, torch.Tensor]:
    advantages = torch.zeros_like(rewards)
    last_gae = torch.zeros_like(last_value)
    for step in reversed(range(rewards.shape[0])):
        next_nonterminal = 1.0 - dones[step]
        next_values = last_value if step == rewards.shape[0] - 1 else values[step + 1]
        delta = rewards[step] + gamma * next_values * next_nonterminal - values[step]
        last_gae = delta + gamma * gae_lambda * next_nonterminal * last_gae
        advantages[step] = last_gae
    returns = advantages + values
    return advantages, returns


def update_policy(
    *,
    policy: torch.nn.Module,
    optimizer: torch.optim.Optimizer,
    obs: torch.Tensor,
    actions: torch.Tensor,
    old_logprobs: torch.Tensor,
    advantages: torch.Tensor,
    returns: torch.Tensor,
    old_values: torch.Tensor,
    args: argparse.Namespace,
) -> dict[str, torch.Tensor]:
    batch_size = obs.shape[0]
    minibatch_size = min(args.minibatch_size, batch_size)
    loss_sums = {
        "policy_loss": torch.zeros((), device=obs.device),
        "value_loss": torch.zeros((), device=obs.device),
        "entropy": torch.zeros((), device=obs.device),
        "approx_kl": torch.zeros((), device=obs.device),
        "clipfrac": torch.zeros((), device=obs.device),
    }
    minibatches = 0

    for _epoch in range(args.update_epochs):
        indices = torch.randperm(batch_size, device=obs.device)
        for start in range(0, batch_size, minibatch_size):
            mb_idx = indices[start : start + minibatch_size]
            logits, new_values = policy(obs[mb_idx])
            _new_actions, new_logprobs, entropy = pufferlib.pytorch.sample_logits(
                logits,
                action=actions[mb_idx],
            )
            logratio = new_logprobs - old_logprobs[mb_idx]
            ratio = logratio.exp()

            mb_advantages = advantages[mb_idx]
            mb_advantages = (mb_advantages - mb_advantages.mean()) / (
                mb_advantages.std(unbiased=False) + 1e-8
            )
            policy_loss_1 = -mb_advantages * ratio
            policy_loss_2 = -mb_advantages * torch.clamp(
                ratio,
                1.0 - args.clip_coef,
                1.0 + args.clip_coef,
            )
            policy_loss = torch.max(policy_loss_1, policy_loss_2).mean()

            new_values = new_values.flatten()
            value_clipped = old_values[mb_idx] + torch.clamp(
                new_values - old_values[mb_idx],
                -args.vf_clip_coef,
                args.vf_clip_coef,
            )
            value_loss_unclipped = (new_values - returns[mb_idx]).square()
            value_loss_clipped = (value_clipped - returns[mb_idx]).square()
            value_loss = 0.5 * torch.max(value_loss_unclipped, value_loss_clipped).mean()
            entropy_loss = entropy.mean()
            loss = policy_loss + args.vf_coef * value_loss - args.ent_coef * entropy_loss

            optimizer.zero_grad(set_to_none=True)
            loss.backward()
            torch.nn.utils.clip_grad_norm_(policy.parameters(), args.max_grad_norm)
            optimizer.step()

            with torch.no_grad():
                approx_kl = ((ratio - 1.0) - logratio).mean()
                clipfrac = ((ratio - 1.0).abs() > args.clip_coef).float().mean()
            loss_sums["policy_loss"] += policy_loss.detach()
            loss_sums["value_loss"] += value_loss.detach()
            loss_sums["entropy"] += entropy_loss.detach()
            loss_sums["approx_kl"] += approx_kl.detach()
            loss_sums["clipfrac"] += clipfrac.detach()
            minibatches += 1

    divisor = max(minibatches, 1)
    return {key: value / divisor for key, value in loss_sums.items()}


def tensor_dict_to_float(metrics: dict[str, torch.Tensor]) -> dict[str, float]:
    return {key: float(value.detach().item()) for key, value in metrics.items()}


@torch.no_grad()
def score_policy(
    policy: torch.nn.Module,
    args: argparse.Namespace,
    params: CartPoleParams,
    device: torch.device,
) -> torch.Tensor:
    env = TorchMultiPendulumCartPoleEnv(
        num_envs=args.select_eval_num_envs,
        num_pendulums=args.num_pendulums,
        params=params,
        reset_mode=args.eval_reset_mode,
        device=device,
        seed=args.seed + 20_000,
    )
    obs = env.reset(seed=args.seed + 20_000)
    active = torch.ones(args.select_eval_num_envs, dtype=torch.bool, device=device)
    controlled_upright_sum = torch.zeros((), device=device)
    active_steps = torch.zeros((), device=device)

    for _step in range(args.select_eval_steps):
        action = deterministic_policy_action(policy, obs)
        env_step = env.step(action)
        obs = env_step.observation
        active_f = active.to(torch.float32)
        controlled_upright_sum += (
            env_step.info["controlled_upright"] * active_f
        ).sum()
        active_steps += active_f.sum()
        active = active & ~(env_step.terminated | env_step.truncated)

    return controlled_upright_sum / torch.clamp(active_steps, min=1.0)


@torch.no_grad()
def evaluate_policy(
    policy: torch.nn.Module,
    args: argparse.Namespace,
    params: CartPoleParams,
    device: torch.device,
) -> dict[str, float]:
    env = TorchMultiPendulumCartPoleEnv(
        num_envs=args.eval_num_envs,
        num_pendulums=args.num_pendulums,
        params=params,
        reset_mode=args.eval_reset_mode,
        device=device,
        seed=args.seed + 10_000,
    )
    obs = env.reset(seed=args.seed + 10_000)
    active = torch.ones(args.eval_num_envs, dtype=torch.bool, device=device)
    episode_reward = torch.zeros(args.eval_num_envs, device=device)
    episode_length = torch.zeros(args.eval_num_envs, device=device)
    stable_timesteps = torch.zeros((), device=device)
    upright_sum = torch.zeros((), device=device)
    controlled_upright_sum = torch.zeros((), device=device)
    top_theta_speed_sum = torch.zeros((), device=device)
    active_steps = torch.zeros((), device=device)

    for _step in range(args.eval_steps):
        action = deterministic_policy_action(policy, obs)
        env_step = env.step(action)
        obs = env_step.observation
        active_f = active.to(torch.float32)
        episode_reward += env_step.reward * active_f
        episode_length += active_f
        stable_timesteps += (env_step.info["stable"] & active).float().sum()
        upright_sum += (env_step.info["height"] * active_f).sum()
        controlled_upright_sum += (
            env_step.info["controlled_upright"] * active_f
        ).sum()
        top_theta_speed_sum += (env_step.info["top_theta_speed"] * active_f).sum()
        active_steps += active_f.sum()
        active = active & ~(env_step.terminated | env_step.truncated)

    synchronize_device(device)
    mean_reward = float(episode_reward.mean().item())
    mean_length = float(episode_length.mean().item())
    stable_count = int(stable_timesteps.item())
    active_step_count = max(float(active_steps.item()), 1.0)
    stable_rate = stable_count / active_step_count
    mean_height = float((upright_sum / active_step_count).item())
    mean_controlled_upright = float(
        (controlled_upright_sum / active_step_count).item()
    )
    mean_top_theta_speed = float((top_theta_speed_sum / active_step_count).item())
    print(
        "eval "
        f"reset_mode={args.eval_reset_mode} "
        f"mean_reward={mean_reward:.2f} "
        f"mean_len={mean_length:.1f} "
        f"mean_height={mean_height:.3f} "
        f"mean_controlled_upright={mean_controlled_upright:.3f} "
        f"mean_top_theta_speed={mean_top_theta_speed:.3f} "
        f"stable_steps={stable_count} "
        f"stable_rate={stable_rate:.4f}",
        flush=True,
    )
    return {
        "mean_reward": mean_reward,
        "mean_length": mean_length,
        "stable_timesteps": float(stable_count),
        "stable_rate": stable_rate,
        "mean_height": mean_height,
        "mean_controlled_upright": mean_controlled_upright,
        "mean_top_theta_speed": mean_top_theta_speed,
    }


def make_optimizer(policy: torch.nn.Module, args: argparse.Namespace) -> torch.optim.Optimizer:
    if args.optimizer == "adam":
        return torch.optim.Adam(
            policy.parameters(),
            lr=args.learning_rate,
            betas=(args.adam_beta1, args.adam_beta2),
            eps=args.adam_eps,
        )
    if args.optimizer == "muon":
        try:
            from heavyball import ForeachMuon
        except ImportError as exc:
            raise RuntimeError("optimizer=muon requires heavyball to be installed") from exc
        return ForeachMuon(
            policy.parameters(),
            lr=args.learning_rate,
            betas=(args.adam_beta1, args.adam_beta2),
            eps=args.adam_eps,
        )
    raise ValueError(f"unknown optimizer {args.optimizer}")


def save_checkpoint(
    policy: torch.nn.Module,
    args: argparse.Namespace,
    params: CartPoleParams,
    total_timesteps: int,
) -> str:
    checkpoint_dir = Path(args.checkpoint_dir)
    checkpoint_dir.mkdir(parents=True, exist_ok=True)
    backend = getattr(args, "backend", "ppo").replace("-", "_")
    checkpoint_path = checkpoint_dir / (
        f"cartpole_{args.num_pendulums}p_{backend}_seed{args.seed}.pt"
    )
    torch.save(
        {
            "model_state_dict": policy.state_dict(),
            "args": vars(args),
            "params": params,
            "total_timesteps": total_timesteps,
            "observation_dim": observation_dim(args.num_pendulums),
            "pufferlib_components": [
                "pufferlib.pytorch.layer_init",
                "pufferlib.pytorch.sample_logits",
                "puffer-style PPO defaults",
            ],
        },
        checkpoint_path,
    )
    print(f"saved_checkpoint={checkpoint_path}", flush=True)
    return str(checkpoint_path)


def params_from_args(args: argparse.Namespace) -> CartPoleParams:
    return replace(
        CartPoleParams(),
        stable_x_threshold=args.stable_x_threshold,
        stable_x_dot_threshold=args.stable_x_dot_threshold,
        stable_theta_threshold=args.stable_theta_threshold,
        stable_theta_dot_threshold=args.stable_theta_dot_threshold,
    )


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument("--backend", choices=["metal-es", "ppo"], default="metal-es")
    parser.add_argument("--num-pendulums", type=int, default=NUM_PENDULUMS)
    parser.add_argument("--num-envs", type=int, default=4096)
    parser.add_argument("--eval-num-envs", type=int, default=1024)
    parser.add_argument("--reset-mode", choices=RESET_MODES, default="downward")
    parser.add_argument("--eval-reset-mode", choices=RESET_MODES, default="downward")
    parser.add_argument("--autoreset", action=argparse.BooleanOptionalAction, default=True)
    parser.add_argument("--seed", type=int, default=123)
    parser.add_argument(
        "--device",
        default="gpu",
        help="Use 'gpu'/'auto' for CUDA/MPS, or pass a concrete GPU device.",
    )

    parser.add_argument("--total-timesteps", type=int, default=3_145_728)
    parser.add_argument("--rollout-steps", type=int, default=64)
    parser.add_argument("--hidden-size", type=int, default=128)
    parser.add_argument("--optimizer", choices=["adam", "muon"], default="adam")
    parser.add_argument("--learning-rate", type=float, default=3e-3)
    parser.add_argument("--gamma", type=float, default=0.99)
    parser.add_argument("--gae-lambda", type=float, default=0.90)
    parser.add_argument("--update-epochs", type=int, default=2)
    parser.add_argument("--minibatch-size", type=int, default=65536)
    parser.add_argument("--clip-coef", type=float, default=0.2)
    parser.add_argument("--vf-coef", type=float, default=2.0)
    parser.add_argument("--vf-clip-coef", type=float, default=0.2)
    parser.add_argument("--ent-coef", type=float, default=0.001)
    parser.add_argument("--max-grad-norm", type=float, default=1.5)
    parser.add_argument("--adam-beta1", type=float, default=0.95)
    parser.add_argument("--adam-beta2", type=float, default=0.999)
    parser.add_argument("--adam-eps", type=float, default=1e-12)
    parser.add_argument("--anneal-lr", action=argparse.BooleanOptionalAction, default=True)
    parser.add_argument("--compile", action=argparse.BooleanOptionalAction, default=False)
    parser.add_argument("--compile-mode", default="max-autotune-no-cudagraphs")
    parser.add_argument("--metal-envs", type=int, default=3_145_728)
    parser.add_argument("--metal-rollout-steps", type=int, default=500)
    parser.add_argument("--metal-iterations", type=int, default=1)
    parser.add_argument("--metal-sigma", type=float, default=0.05)
    parser.add_argument("--metal-learning-rate", type=float, default=0.02)

    parser.add_argument("--log-every", type=int, default=0)
    parser.add_argument("--eval-steps", type=int, default=500)
    parser.add_argument("--select-best", action=argparse.BooleanOptionalAction, default=True)
    parser.add_argument("--select-best-every", type=int, default=3)
    parser.add_argument("--select-eval-num-envs", type=int, default=128)
    parser.add_argument("--select-eval-steps", type=int, default=100)
    parser.add_argument("--stable-x-threshold", type=float, default=0.5)
    parser.add_argument("--stable-x-dot-threshold", type=float, default=0.75)
    parser.add_argument("--stable-theta-threshold", type=float, default=float(np.deg2rad(12.0)))
    parser.add_argument("--stable-theta-dot-threshold", type=float, default=1.0)

    parser.add_argument("--checkpoint-dir", default="checkpoints")
    parser.add_argument("--checkpoint", action=argparse.BooleanOptionalAction, default=True)
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
        f"rollout_controlled_upright={result.rollout_controlled_upright:.3f} "
        f"eval_mean_return={result.eval_mean_reward:.2f} "
        f"eval_mean_height={result.eval_mean_height:.3f} "
        f"eval_controlled_upright={result.eval_controlled_upright:.3f} "
        f"eval_top_theta_speed={result.eval_top_theta_speed:.3f} "
        f"eval_stable_steps={result.eval_stable_timesteps} "
        f"eval_stable_rate={result.eval_stable_rate:.3f} "
        f"checkpoint={result.checkpoint_path} "
        f"video={result.video_path}",
        flush=True,
    )


if __name__ == "__main__":
    main()
