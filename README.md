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

## Smoke Runs

Run short PPO smoke trainings for one and two pendulums:

```bash
python -m cartpole_multi.smoke
```

Or run one configuration directly:

```bash
python -m cartpole_multi.train --num-pendulums 2 --total-timesteps 2048
```

Use `--backend multiprocessing --num-workers 4 --num-envs 64` for a faster
CPU-vectorized run once the basic serial smoke path is working.
