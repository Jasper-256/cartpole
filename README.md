# Multi-Pendulum CartPole Training

This project trains a neural actor/critic controller for a cart with a serial
chain of equal-length pendulums. `--num-pendulums 2` means a double pendulum
attached end-to-end, not two independent rods mounted on the cart.

The default backend is a native Metal ES trainer for the two-pendulum speed
target. It keeps actor weights, rollout scoring, gradient reduction, and the ES
weight update on the GPU, then hands the learned linear actor back to Python for
checkpointing, evaluation, and video rendering. The default run trains
`1,572,864,000` simulator steps, which is 500x the earlier PPO default step
budget.

A batched torch simulator and PufferLib-style PPO loop are still available with
`--backend ppo`. In that path rollout tensors stay on the selected torch device,
including environment state, rewards, dones, policy inference, and PPO buffers.
CPU transfers are kept out of the rollout/update path; checkpointing and video
rendering happen after training.

In PPO mode the trainer keeps the best checkpoint according to a lightweight
on-device deterministic eval of `controlled_upright`, which rewards all links
being high while angular velocity is low. This avoids selecting a later PPO
update just because it collected more transient height reward.

The default shape is tuned for two pendulums on GPU/MPS, but the simulator,
policy observation size, and trainer all derive their dimensions from
`--num-pendulums`.

## Setup

Python 3.12 is recommended.

```bash
python -m venv .venv
. .venv/bin/activate
python -m pip install -e .
```

## Run

Use the main trainer to optimize the policy, evaluate it, save a checkpoint, and
record a video:

```bash
python -m cartpole_multi.train --num-pendulums 2
```

`--device gpu` is the default and requires PyTorch to see CUDA or MPS. The
default `metal-es` backend requires Apple Metal/MPS. CPU training is
intentionally rejected; the CPU is only used by the post-training video renderer.

Videos are saved to `videos/`; checkpoints are saved to `checkpoints/`.

## Main Knobs

- `--backend`: `metal-es` by default for the high-speed native Metal trainer;
  use `ppo` for the torch/PufferLib PPO path.
- `--metal-envs`: ES population size. Defaults to `3145728`.
- `--metal-rollout-steps`: Metal rollout horizon. Defaults to `500`.
- `--metal-iterations`: ES update count. Defaults to `1`.
- `--num-envs`: number of parallel torch environments in PPO mode. Defaults to
  `4096`.
- `--rollout-steps`: rollout horizon per PPO update. Defaults to `64`.
- `--total-timesteps`: PPO training step budget. Defaults to `3145728`, which
  is 12 PPO updates with the default env count and rollout horizon.
- `--minibatch-size`: PPO minibatch size. Defaults to `65536`.
- `--gamma`, `--gae-lambda`, `--clip-coef`, `--vf-coef`, `--ent-coef`: PPO
  defaults following PufferLib/CleanRL-style settings.
- `--reset-mode`: training reset distribution. Defaults to `downward`.
- `--eval-reset-mode`: video/eval reset distribution. Defaults to `downward`.
- `--log-every`: optional update logging interval. Defaults to `0` to avoid
  training-time CPU synchronization.
- `--select-best`: keep the best on-device eval checkpoint by controlled-upright
  score. Defaults to enabled with a cheap eval every 3 updates.
