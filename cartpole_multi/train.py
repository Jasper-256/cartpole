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

from cartpole_multi.config import NUM_PENDULUMS
from cartpole_multi.env import CartPoleParams
from cartpole_multi.observations import observation_dim
from cartpole_multi.policy import CartPolePolicy
from cartpole_multi.torch_env import (
    RESET_MODES,
    TorchMultiPendulumCartPoleEnv,
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
    train_elapsed: float
    eval_mean_reward: float
    eval_mean_length: float
    eval_mean_height: float
    eval_max_height: float
    eval_controlled_upright: float
    eval_max_controlled_upright: float
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
            "Training requires CUDA or MPS. Run outside the sandbox with "
            "--device gpu/auto/mps/cuda; CPU is only used by post-training rendering."
        )
    if device.type != "mps":
        raise RuntimeError("The native Metal ES trainer requires Apple Metal/MPS")
    if args.num_pendulums != 2:
        raise RuntimeError("The native Metal ES trainer currently supports --num-pendulums 2")

    torch.set_float32_matmul_precision("high")
    torch.manual_seed(args.seed)

    binary = ensure_metal_es_binary()
    with tempfile.NamedTemporaryFile(
        prefix="cartpole_metal_weights_",
        suffix=".bin",
    ) as weights_file:
        cmd = [
            str(binary),
            str(args.num_envs),
            str(args.rollout_steps),
            str(args.updates),
            str(args.sigma),
            str(args.learning_rate),
            weights_file.name,
        ]
        print(
            "train_start "
            "backend=metal-es "
            f"device={device.type} "
            f"pendulums={args.num_pendulums} "
            f"num_envs={args.num_envs} "
            f"rollout_steps={args.rollout_steps} "
            f"updates={args.updates} "
            "optimizer=es",
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

    obs_dim = observation_dim(args.num_pendulums)
    expected_weights = obs_dim + 1
    if weights.shape != (expected_weights,):
        raise RuntimeError(
            f"metal-es wrote {weights.shape[0]} weights, expected {expected_weights}"
        )

    policy = CartPolePolicy(observation_size=obs_dim).to(device)
    load_metal_actor_weights(policy, weights, obs_dim, device)

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
        updates=args.updates,
        steps_per_second=stats["sps"],
        train_elapsed=stats["elapsed"],
        eval_mean_reward=eval_stats["mean_reward"],
        eval_mean_length=eval_stats["mean_length"],
        eval_mean_height=eval_stats["mean_height"],
        eval_max_height=eval_stats["max_height"],
        eval_controlled_upright=eval_stats["mean_controlled_upright"],
        eval_max_controlled_upright=eval_stats["max_controlled_upright"],
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
    observation_size: int,
    device: torch.device,
) -> None:
    actor = weights.reshape(1, observation_size + 1)
    with torch.no_grad():
        policy.actor.weight.copy_(
            torch.as_tensor(actor[:, :observation_size], dtype=torch.float32, device=device)
        )
        policy.actor.bias.copy_(
            torch.as_tensor(actor[:, observation_size], dtype=torch.float32, device=device)
        )


def deterministic_policy_force(
    policy: torch.nn.Module,
    obs: torch.Tensor,
    params: CartPoleParams,
) -> torch.Tensor:
    force_logits = policy(obs)
    return params.force_mag * torch.tanh(force_logits.flatten())


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
    max_height = torch.zeros((), device=device)
    max_controlled_upright = torch.zeros((), device=device)
    active_steps = torch.zeros((), device=device)

    for _step in range(args.eval_steps):
        force = deterministic_policy_force(policy, obs, params)
        env_step = env.step_force(force)
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
        max_height = torch.maximum(max_height, (env_step.info["height"] * active_f).max())
        max_controlled_upright = torch.maximum(
            max_controlled_upright,
            (env_step.info["controlled_upright"] * active_f).max(),
        )
        active_steps += active_f.sum()
        active = active & ~(env_step.terminated | env_step.truncated)

    synchronize_device(device)
    mean_reward = float(episode_reward.mean().item())
    mean_length = float(episode_length.mean().item())
    stable_count = int(stable_timesteps.item())
    active_step_count = max(float(active_steps.item()), 1.0)
    stable_rate = stable_count / active_step_count
    mean_height = float((upright_sum / active_step_count).item())
    max_height_value = float(max_height.item())
    mean_controlled_upright = float(
        (controlled_upright_sum / active_step_count).item()
    )
    max_controlled_upright_value = float(max_controlled_upright.item())
    mean_top_theta_speed = float((top_theta_speed_sum / active_step_count).item())
    print(
        "eval "
        f"reset_mode={args.eval_reset_mode} "
        f"mean_reward={mean_reward:.2f} "
        f"mean_len={mean_length:.1f} "
        f"mean_height={mean_height:.3f} "
        f"max_height={max_height_value:.3f} "
        f"mean_controlled_upright={mean_controlled_upright:.3f} "
        f"max_controlled_upright={max_controlled_upright_value:.3f} "
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
        "max_height": max_height_value,
        "mean_controlled_upright": mean_controlled_upright,
        "max_controlled_upright": max_controlled_upright_value,
        "mean_top_theta_speed": mean_top_theta_speed,
    }


def save_checkpoint(
    policy: torch.nn.Module,
    args: argparse.Namespace,
    params: CartPoleParams,
    total_timesteps: int,
) -> str:
    checkpoint_dir = Path(args.checkpoint_dir)
    checkpoint_dir.mkdir(parents=True, exist_ok=True)
    checkpoint_path = checkpoint_dir / (
        f"cartpole_{args.num_pendulums}p_metal_es_seed{args.seed}.pt"
    )
    torch.save(
        {
            "model_state_dict": policy.state_dict(),
            "args": vars(args),
            "params": params,
            "total_timesteps": total_timesteps,
            "observation_dim": observation_dim(args.num_pendulums),
            "trainer": "native_metal_es",
            "optimizer": "evolution_strategies",
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
    parser.add_argument("--num-pendulums", type=int, default=NUM_PENDULUMS)
    parser.add_argument("--num-envs", type=int, default=2_359_296)
    parser.add_argument("--rollout-steps", type=int, default=500)
    parser.add_argument("--updates", type=int, default=14)
    parser.add_argument("--sigma", type=float, default=0.25)
    parser.add_argument("--learning-rate", type=float, default=0.3)
    parser.add_argument("--eval-num-envs", type=int, default=1024)
    parser.add_argument("--eval-steps", type=int, default=500)
    parser.add_argument("--eval-reset-mode", choices=RESET_MODES, default="downward")
    parser.add_argument("--seed", type=int, default=123)
    parser.add_argument(
        "--device",
        default="gpu",
        help="Use 'gpu'/'auto' for MPS. CPU training is intentionally rejected.",
    )
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
        f"train_elapsed={result.train_elapsed:.3f}s "
        f"sps={result.steps_per_second:.0f} "
        f"eval_mean_return={result.eval_mean_reward:.2f} "
        f"eval_mean_len={result.eval_mean_length:.1f} "
        f"eval_mean_height={result.eval_mean_height:.3f} "
        f"eval_max_height={result.eval_max_height:.3f} "
        f"eval_controlled_upright={result.eval_controlled_upright:.3f} "
        f"eval_max_controlled_upright={result.eval_max_controlled_upright:.3f} "
        f"eval_top_theta_speed={result.eval_top_theta_speed:.3f} "
        f"eval_stable_steps={result.eval_stable_timesteps} "
        f"eval_stable_rate={result.eval_stable_rate:.3f} "
        f"checkpoint={result.checkpoint_path} "
        f"video={result.video_path}",
        flush=True,
    )


if __name__ == "__main__":
    main()
