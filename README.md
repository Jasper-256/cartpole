# Multi-Pendulum CartPole PPO

This project trains a neural actor/critic controller for a cart with a serial
chain of equal-length pendulums. `--num-pendulums 2` means a double pendulum
attached end-to-end, not two independent rods mounted on the cart.

Training now uses a batched torch simulator and a PufferLib-style PPO loop. The
rollout tensors stay on the selected torch device, including the environment
state, rewards, dones, policy inference, and PPO buffers. CPU transfers are kept
out of the rollout/update path; checkpointing and video rendering happen after
training.

By default the trainer keeps the best checkpoint according to a lightweight
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

`--device gpu` is the default and requires PyTorch to see CUDA or MPS. CPU
training is intentionally rejected; the CPU is only used by the post-training
video renderer.

Videos are saved to `videos/`; checkpoints are saved to `checkpoints/`.

## Main Knobs

- `--num-envs`: number of parallel torch environments. Defaults to `4096`.
- `--rollout-steps`: rollout horizon per PPO update. Defaults to `64`.
- `--total-timesteps`: training step budget. Defaults to `3145728`, which is
  12 PPO updates with the default env count and rollout horizon.
- `--minibatch-size`: PPO minibatch size. Defaults to `65536`.
- `--gamma`, `--gae-lambda`, `--clip-coef`, `--vf-coef`, `--ent-coef`: PPO
  defaults following PufferLib/CleanRL-style settings.
- `--reset-mode`: training reset distribution. Defaults to `downward`.
- `--eval-reset-mode`: video/eval reset distribution. Defaults to `downward`.
- `--log-every`: optional update logging interval. Defaults to `0` to avoid
  training-time CPU synchronization.
- `--select-best`: keep the best on-device eval checkpoint by controlled-upright
  score. Defaults to enabled with a cheap eval every 3 updates.
