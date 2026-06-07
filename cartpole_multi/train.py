from __future__ import annotations

import argparse
import time
from dataclasses import dataclass

import numpy as np

from cartpole_multi.config import NUM_PENDULUMS
from cartpole_multi.env import CartPoleParams, MultiPendulumCartPoleEnv
from cartpole_multi.trajectory import (
    TrajectoryPlan,
    feedback_action,
    optimize_trajectory_plan,
    should_activate_feedback,
)
from cartpole_multi.video import open_video, record_action_video


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
class ControllerRun:
    actions: list[int]
    steps: int
    stable_steps: int
    max_consecutive_stable_steps: int
    first_stable_step: int | None
    total_return: float
    terminated: bool
    truncated: bool
    solved: bool


def train(args: argparse.Namespace) -> TrainResult:
    if args.num_pendulums != 2:
        raise ValueError("this training pipeline is specialized for --num-pendulums 2")

    start = time.perf_counter()
    plan_start = time.perf_counter()
    plan = optimize_trajectory_plan(args)
    plan_elapsed = max(time.perf_counter() - plan_start, 1e-9)
    print(
        f"timing phase=trajectory_search elapsed={plan_elapsed:.3f}s "
        f"device={plan.device} batch={plan.batch_size} "
        f"simulated_steps={plan.simulated_steps} "
        f"sim_sps={plan.simulated_steps / plan_elapsed:.0f} "
        f"best_score={plan.best_score:.1f} actions={len(plan.actions)}",
        flush=True,
    )

    eval_start = time.perf_counter()
    run = evaluate_plan(plan, args)
    eval_elapsed = max(time.perf_counter() - eval_start, 1e-9)
    elapsed = max(time.perf_counter() - start, 1e-9)
    print(
        f"timing phase=feedback_eval elapsed={eval_elapsed:.3f}s steps={run.steps}",
        flush=True,
    )
    print(
        "eval "
        f"reset_mode={args.eval_reset_mode} "
        f"return={run.total_return:.2f} "
        f"steps={run.steps} "
        f"stable_steps={run.stable_steps} "
        f"max_consecutive_stable_steps={run.max_consecutive_stable_steps} "
        f"first_stable_step={run.first_stable_step} "
        f"solved={run.solved}",
        flush=True,
    )

    video_path = None
    if args.video:
        video_start = time.perf_counter()
        video_path = record_action_video(run.actions, args, suffix="trajectory")
        print(
            f"timing phase=video_total elapsed={time.perf_counter() - video_start:.3f}s",
            flush=True,
        )
        print(f"saved_video={video_path}", flush=True)
        if args.open_video:
            open_video(video_path)

    return TrainResult(
        num_pendulums=args.num_pendulums,
        total_timesteps=run.steps,
        updates=plan.iterations,
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


def evaluate_plan(plan: TrajectoryPlan, args: argparse.Namespace) -> ControllerRun:
    params = CartPoleParams()
    env = MultiPendulumCartPoleEnv(
        num_pendulums=args.num_pendulums,
        reset_mode=args.eval_reset_mode,
        seed=args.seed,
    )
    obs, info = env.reset()
    actions: list[int] = []
    total_return = 0.0
    stable_steps = 0
    consecutive_stable_steps = 0
    max_consecutive_stable_steps = 0
    first_stable_step: int | None = None
    terminated = False
    truncated = False
    feedback_active = False
    rollout_start = time.perf_counter()

    for step in range(args.video_steps):
        feedback_active = feedback_active or should_activate_feedback(obs, args)
        action = feedback_action(obs, step, plan, args, params, feedback_active)
        obs, reward, terminated, truncated, info = env.step(action)
        actions.append(action)
        total_return += float(reward)

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
            step % args.eval_log_every == 0 or terminated or truncated
        ):
            print(
                f"step={step} action={action} stable_steps={stable_steps} "
                f"feedback={int(feedback_active)} "
                f"x={obs[0]:+.3f} theta={obs[2:2 + args.num_pendulums]} "
                f"theta_dot={obs[2 + args.num_pendulums:]}",
                flush=True,
            )

        if terminated or truncated:
            break

    rollout_elapsed = max(time.perf_counter() - rollout_start, 1e-9)
    print(
        f"timing phase=feedback_rollout elapsed={rollout_elapsed:.3f}s "
        f"steps={step + 1}",
        flush=True,
    )
    return ControllerRun(
        actions=actions,
        steps=step + 1,
        stable_steps=stable_steps,
        max_consecutive_stable_steps=max_consecutive_stable_steps,
        first_stable_step=first_stable_step,
        total_return=total_return,
        terminated=terminated,
        truncated=truncated,
        solved=stable_steps >= args.solve_stable_steps,
    )


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument("--num-pendulums", type=int, default=NUM_PENDULUMS)
    parser.add_argument(
        "--eval-reset-mode",
        choices=["downward", "upright", "uniform", "mixed"],
        default="downward",
    )
    parser.add_argument("--seed", type=int, default=123)
    parser.add_argument(
        "--device",
        default="gpu",
        help="Use 'gpu' for CUDA/MPS, or pass 'cpu' explicitly for development.",
    )
    parser.add_argument("--log-every", type=int, default=1)
    parser.add_argument("--eval-log-every", type=int, default=25)
    parser.add_argument("--solve-stable-steps", type=int, default=100)
    parser.add_argument("--stable-x-threshold", type=float, default=0.5)
    parser.add_argument("--stable-x-dot-threshold", type=float, default=0.75)
    parser.add_argument("--stable-theta-threshold", type=float, default=float(np.deg2rad(12.0)))
    parser.add_argument("--stable-theta-dot-threshold", type=float, default=1.0)

    parser.add_argument("--trajectory-seed", type=int, default=42)
    parser.add_argument("--trajectory-batch-size", type=int, default=None)
    parser.add_argument("--trajectory-segments", type=int, default=90)
    parser.add_argument("--trajectory-action-repeat", type=int, default=5)
    parser.add_argument("--trajectory-iterations", type=int, default=12)
    parser.add_argument("--trajectory-elite-frac", type=float, default=0.025)
    parser.add_argument("--trajectory-mutation-rate", type=float, default=0.08)
    parser.add_argument("--trajectory-upright-reward", type=float, default=3.0)
    parser.add_argument("--trajectory-cost-weight", type=float, default=0.0)
    parser.add_argument("--trajectory-theta-dot-cost", type=float, default=0.08)
    parser.add_argument("--trajectory-x-cost", type=float, default=0.15)
    parser.add_argument("--trajectory-x-dot-cost", type=float, default=0.02)
    parser.add_argument("--trajectory-stable-bonus", type=float, default=100000.0)
    parser.add_argument("--trajectory-min-cost-weight", type=float, default=25000.0)
    parser.add_argument("--trajectory-final-cost-weight", type=float, default=5000.0)
    parser.add_argument("--trajectory-alive-bonus", type=float, default=500.0)
    parser.add_argument("--trajectory-terminal-penalty", type=float, default=2000.0)

    parser.add_argument("--feedback-switch-x-threshold", type=float, default=0.5)
    parser.add_argument("--feedback-switch-x-dot-threshold", type=float, default=0.75)
    parser.add_argument(
        "--feedback-switch-theta-threshold",
        type=float,
        default=float(np.deg2rad(12.0)),
    )
    parser.add_argument("--feedback-switch-theta-dot-threshold", type=float, default=1.0)
    parser.add_argument("--feedback-force-deadzone", type=float, default=1.0)
    parser.add_argument(
        "--feedback-track-plan",
        action=argparse.BooleanOptionalAction,
        default=True,
    )
    parser.add_argument(
        "--feedback-tracking-gain",
        type=float,
        nargs=6,
        default=(2.5, 2.0, 5.0, 5.0, 0.6, 0.6),
    )
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
        f"video={result.video_path}",
        flush=True,
    )


if __name__ == "__main__":
    main()
