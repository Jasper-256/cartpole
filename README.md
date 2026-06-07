# Multi-Pendulum CartPole with PufferLib

This is a small, trainable cartpole setup where one cart balances a configurable
serial chain of equal-length pendulums. `--num-pendulums 2` means a double
pendulum attached end-to-end, not two independent pendulums mounted on the cart.
The default pendulum count lives in one place:

```python
# cartpole_multi/config.py
NUM_PENDULUMS = 1
```

The environment is a flat `gymnasium.Env` with a `Discrete(3)` action space and
`Box` observations. Training defaults to a fast NumPy batched backend, and the
Gymnasium environment can still be wrapped with
`pufferlib.emulation.GymnasiumPufferEnv` and vectorized through
`pufferlib.vector.make` with the `serial` or `multiprocessing` backends.

Episodes reset with the chain hanging downward, which is the natural resting
state. The stabilization target is still upright.

## Setup

Python 3.12 is recommended.

```bash
python3.12 -m venv .venv
. .venv/bin/activate
python -m pip install -e .
```

## Quick Smoke Runs

Use the main trainer to optimize a controller and then record a video:

```bash
python -m cartpole_multi.train --num-pendulums 1
python -m cartpole_multi.train --num-pendulums 2
```

The default optimizer is a generic cross-entropy method (CEM) over the configured
pendulum count. It is compute-heavy, but it is the path intended for the
downward swing-up task and still uses the same `train` entrypoint. For a quick
smoke check, shrink the CEM search and skip video:

```bash
python -m cartpole_multi.train --num-pendulums 2 --no-video --video-steps 5 \
  --cem-envs 64 --cem-iterations 1 --cem-segments 4 \
  --cem-planner-envs 64 --cem-planner-iterations 1 --cem-planner-horizon 4
```

The PPO loop is still available when you want neural-policy training:

```bash
python -m cartpole_multi.train --optimizer ppo --num-pendulums 2 \
  --total-timesteps 1000000 --no-video
```

Use the PufferLib wrapper explicitly when you want to compare it:

```bash
python -m cartpole_multi.train --optimizer ppo --num-pendulums 2 \
  --backend multiprocessing --num-workers 4 --num-envs 64 --no-video
```

Each run saves a post-training evaluation video to `videos/` and tries to open
it when training finishes. Use `--no-open-video` to save without opening, or
`--no-video` to skip rendering:

```bash
python -m cartpole_multi.train --num-pendulums 2 --no-open-video
```

Training logs include an upright stabilization metric:

- `stable_steps`: count of env-timesteps where the cart is near center and all
  pendulums are upright and slow.
- `stable_rate`: `stable_steps / steps`.
- `recent_stable_rate`: same ratio over the recent rollout window.

By default a timestep is stable when `|x| <= 0.5`, every `|theta| <= 12 deg`
(`0.20944 rad`), and every `|theta_dot| <= 1.0` radians/sec. You can tune those with
`--stable-x-threshold`, `--stable-theta-threshold`, and
`--stable-theta-dot-threshold`.

For maximum PPO throughput on this small model, keep `--backend numpy` and use a
large environment batch such as `--num-envs 1024`.

Training also supports `--reset-mode {downward,upright,uniform,mixed}` and
`--observation-mode {trig,raw}`. The default policy input uses trig angle
features so the network does not have to learn across the raw angle wrap.
