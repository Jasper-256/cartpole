from __future__ import annotations

from argparse import Namespace

from cartpole_multi.train import train


def main() -> None:
    base = dict(
        total_timesteps=1024,
        num_envs=8,
        num_workers=1,
        backend="serial",
        rollout_steps=32,
        minibatch_size=128,
        update_epochs=1,
        learning_rate=2.5e-4,
        gamma=0.99,
        gae_lambda=0.95,
        clip_coef=0.2,
        ent_coef=0.01,
        vf_coef=0.5,
        max_grad_norm=0.5,
        seed=1,
        device="cpu",
        log_every=1,
    )
    for num_pendulums in (1, 2):
        print(f"\n=== smoke train: {num_pendulums} pendulum(s) ===")
        result = train(Namespace(num_pendulums=num_pendulums, **base))
        print(
            f"smoke_result pendulums={result.num_pendulums} "
            f"timesteps={result.total_timesteps} sps={result.steps_per_second:.0f} "
            f"mean_return={result.last_mean_reward:.2f} mean_len={result.last_mean_length:.1f}"
        )


if __name__ == "__main__":
    main()

