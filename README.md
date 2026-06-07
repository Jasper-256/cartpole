# Multi-Pendulum CartPole Trajectory Optimizer

This project trains a controller for a cart with a serial chain of equal-length
pendulums. `--num-pendulums 2` means a double pendulum attached end-to-end, not
two independent rods mounted on the cart.

The training entrypoint is specialized for the double-pendulum swing-up task. It
uses a GPU batched trajectory search over compact repeated action segments, then
evaluates the resulting plan with deterministic local feedback near upright. The hot
search loop keeps state as separate float32 tensors on the selected device and
does not materialize repeated action sequences.

Episodes reset with the chain hanging downward, which is the natural resting
state. The stabilization target is upright.

## Setup

Python 3.12 is recommended.

```bash
python -m venv .venv
. .venv/bin/activate
python -m pip install -e .
```

## Run

Use the main trainer to optimize a trajectory, evaluate the feedback controller,
and record a video:

```bash
python -m cartpole_multi.train --num-pendulums 2
```

`--device gpu` is the default and requires PyTorch to see CUDA or MPS. For a
small development smoke run on a machine without visible GPU support:

```bash
python -m cartpole_multi.train --num-pendulums 2 --device cpu --no-video \
  --trajectory-batch-size 256 --trajectory-iterations 1 \
  --trajectory-segments 4 --trajectory-action-repeat 2 --video-steps 8
```

Each full run saves a post-evaluation video to `videos/` and tries to open it
when training finishes. Use `--no-open-video` to save without opening, or
`--no-video` to skip rendering.

## Main Knobs

- `--trajectory-batch-size`: number of candidate segment sequences scored per
  update. Defaults to `131072`.
- `--trajectory-iterations`: number of trajectory search updates.
- `--trajectory-segments`: number of action segments in the plan.
- `--trajectory-action-repeat`: simulator steps per action segment.
- `--feedback-switch-*`: thresholds that switch from plan tracking to local
  upright feedback.

Training logs include `stable_steps`, the count of env-timesteps where the cart
is near center and both pendulums are upright and slow. By default a timestep is
stable when `|x| <= 0.5`, every `|theta| <= 12 deg`, and every
`|theta_dot| <= 1.0` radians/sec.
