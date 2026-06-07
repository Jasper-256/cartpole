# Multi-Pendulum CartPole with PufferLib

This is a small, trainable cartpole setup where one cart balances a configurable
number of inverted pendulums. The default pendulum count lives in one place:

```python
# cartpole_multi/config.py
NUM_PENDULUMS = 1
```

The environment is a flat `gymnasium.Env` with a `Discrete(3)` action space and
`Box` observations, then wrapped with `pufferlib.emulation.GymnasiumPufferEnv`
and vectorized through `pufferlib.vector.make`.

## Setup

Python 3.12 is recommended.

```bash
python3.12 -m venv .venv
. .venv/bin/activate
python -m pip install -e .
```

## Quick Smoke Runs

Use the main trainer with a small timestep budget to check that the environment,
PufferLib wrapper, and PPO loop are working:

```bash
python -m cartpole_multi.train --num-pendulums 1 --total-timesteps 1024
python -m cartpole_multi.train --num-pendulums 2 --total-timesteps 1024
```

For a slightly longer two-pendulum check:

```bash
python -m cartpole_multi.train --num-pendulums 2 --total-timesteps 2048
```

Each run saves a post-training evaluation video to `videos/` and tries to open
it when training finishes. Use `--no-open-video` to save without opening, or
`--no-video` to skip rendering:

```bash
python -m cartpole_multi.train --num-pendulums 2 --total-timesteps 2048 --no-open-video
```

Training logs include a stabilization metric:

- `stable_steps`: count of env-timesteps where the cart is near center and all
  pendulums are upright and slow.
- `stable_rate`: `stable_steps / steps`.
- `recent_stable_rate`: same ratio over the recent rollout window.

By default a timestep is stable when `|x| <= 0.5`, every `|theta| <= 12 deg`
(`0.20944 rad`), and every `|theta_dot| <= 1.0` radians/sec. You can tune those with
`--stable-x-threshold`, `--stable-theta-threshold`, and
`--stable-theta-dot-threshold`.

Use `--backend multiprocessing --num-workers 4 --num-envs 64` for a faster
CPU-vectorized run once the basic serial quick run is working.
