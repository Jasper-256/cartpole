from __future__ import annotations

import argparse
import time
from dataclasses import dataclass

import numpy as np
import torch
from torch import nn
from torch.distributions.categorical import Categorical

from cartpole_multi.config import NUM_PENDULUMS
from cartpole_multi.puffer_env import make_vec_env
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
    video_path: str | None


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
    torch.manual_seed(args.seed)
    np.random.seed(args.seed)

    device = torch.device(args.device)
    envs = make_vec_env(
        num_pendulums=args.num_pendulums,
        num_envs=args.num_envs,
        num_workers=args.num_workers,
        backend=args.backend,
        seed=args.seed,
    )
    obs, _infos = envs.reset()
    obs = np.asarray(obs, dtype=np.float32)

    obs_dim = int(np.prod(obs.shape[1:]))
    action_dim = 3
    model = ActorCritic(obs_dim, action_dim).to(device)
    optimizer = torch.optim.Adam(model.parameters(), lr=args.learning_rate, eps=1e-5)

    rollout_size = args.num_envs * args.rollout_steps
    updates = max(1, args.total_timesteps // rollout_size)
    minibatch_size = min(args.minibatch_size, rollout_size)

    obs_buf = torch.zeros((args.rollout_steps, args.num_envs, obs_dim), device=device)
    actions_buf = torch.zeros((args.rollout_steps, args.num_envs), device=device, dtype=torch.long)
    logprobs_buf = torch.zeros((args.rollout_steps, args.num_envs), device=device)
    rewards_buf = torch.zeros((args.rollout_steps, args.num_envs), device=device)
    dones_buf = torch.zeros((args.rollout_steps, args.num_envs), device=device)
    values_buf = torch.zeros((args.rollout_steps, args.num_envs), device=device)

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
        for step in range(args.rollout_steps):
            global_step += args.num_envs
            obs_tensor = torch.as_tensor(obs.reshape(args.num_envs, obs_dim), device=device)
            with torch.no_grad():
                action, logprob, _entropy, value = model.get_action_and_value(obs_tensor)

            next_obs, reward, terminated, truncated, _infos = envs.step(action.cpu().numpy())
            done = np.logical_or(terminated, truncated)

            obs_buf[step] = obs_tensor
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
            next_obs_tensor = torch.as_tensor(obs.reshape(args.num_envs, obs_dim), device=device)
            next_value = model(next_obs_tensor)[1]
            advantages = torch.zeros_like(rewards_buf, device=device)
            lastgaelam = torch.zeros(args.num_envs, device=device)
            for t in reversed(range(args.rollout_steps)):
                if t == args.rollout_steps - 1:
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

    close = getattr(envs, "close", None)
    if close is not None:
        close()

    video_path = None
    if args.video:
        video_path = record_policy_video(model, obs_dim, args)
        print(f"saved_video={video_path}")
        if args.open_video:
            open_video(video_path)

    elapsed = max(time.time() - start, 1e-9)
    return TrainResult(
        num_pendulums=args.num_pendulums,
        total_timesteps=global_step,
        updates=updates,
        steps_per_second=global_step / elapsed,
        last_mean_reward=float(np.mean(recent_returns)) if recent_returns else 0.0,
        last_mean_length=float(np.mean(recent_lengths)) if recent_lengths else 0.0,
        stable_timesteps=stable_timesteps,
        stable_rate=stable_timesteps / max(global_step, 1),
        video_path=video_path,
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
    parser.add_argument("--total-timesteps", type=int, default=4096)
    parser.add_argument("--num-envs", type=int, default=16)
    parser.add_argument("--num-workers", type=int, default=4)
    parser.add_argument("--backend", choices=["serial", "multiprocessing"], default="serial")
    parser.add_argument("--rollout-steps", type=int, default=64)
    parser.add_argument("--minibatch-size", type=int, default=256)
    parser.add_argument("--update-epochs", type=int, default=2)
    parser.add_argument("--learning-rate", type=float, default=2.5e-4)
    parser.add_argument("--gamma", type=float, default=0.99)
    parser.add_argument("--gae-lambda", type=float, default=0.95)
    parser.add_argument("--clip-coef", type=float, default=0.2)
    parser.add_argument("--ent-coef", type=float, default=0.01)
    parser.add_argument("--vf-coef", type=float, default=0.5)
    parser.add_argument("--max-grad-norm", type=float, default=0.5)
    parser.add_argument("--seed", type=int, default=1)
    parser.add_argument("--device", default="cpu")
    parser.add_argument("--log-every", type=int, default=1)
    parser.add_argument("--stable-x-threshold", type=float, default=0.5)
    parser.add_argument("--stable-theta-threshold", type=float, default=float(np.deg2rad(12.0)))
    parser.add_argument("--stable-theta-dot-threshold", type=float, default=1.0)
    parser.add_argument("--stable-window-timesteps", type=int, default=4096)
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
        f"video={result.video_path}"
    )


if __name__ == "__main__":
    main()
