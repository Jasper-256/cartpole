# Multi-Pendulum CartPole Training

This project trains a neural actor controller for a cart with a serial chain of
equal-length pendulums. `--num-pendulums 2` means a double pendulum attached
end-to-end, not two independent rods mounted on the cart.

The trainer is a native Metal evolution-strategies loop. Actor weights, rollout
scoring, gradient reduction, and weight updates stay in Metal during training.
Python only compiles/launches the Metal helper, saves the learned linear policy,
runs a GPU evaluation pass, and renders the final video.

`--num-pendulums 2` uses the tuned closed-form Metal helper. Other positive
pendulum counts use the same ES/checkpoint/eval/video pipeline through the
generic serial-chain Metal helper.

The default run trains `16,515,072,000` simulator steps:

```bash
python -m cartpole_multi.train
```

Training requires Apple Metal/MPS and intentionally rejects CPU execution. On
this machine, run training outside the sandbox so PyTorch can see `mps:0`.

Videos are saved to `videos/`; checkpoints are saved to `checkpoints/`.

## Setup

Python 3.12 is recommended.

```bash
python -m venv .venv
. .venv/bin/activate
python -m pip install -e .
```

## Main Knobs

- `--num-envs`: ES population size. Defaults to `2359296`.
- `--num-pendulums`: serial chain length. Defaults to `2`.
- `--rollout-steps`: rollout horizon per ES sample. Defaults to `500`.
- `--updates`: ES weight update count. Defaults to `14`.
- `--sigma`: perturbation scale. Defaults to `0.25`.
- `--learning-rate`: ES update scale. Defaults to `0.3`.
- `--eval-reset-mode`: video/eval reset distribution. Defaults to `downward`.
- `--video/--no-video`: enable or skip video rendering.
- `--open-video/--no-open-video`: open the rendered video after training.
