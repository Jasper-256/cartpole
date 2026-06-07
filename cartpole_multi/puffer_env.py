from __future__ import annotations

from functools import partial
from typing import Any

from cartpole_multi.batched_env import BatchedMultiPendulumCartPoleEnv
from cartpole_multi.config import NUM_PENDULUMS
from cartpole_multi.env import MultiPendulumCartPoleEnv


def make_env(num_pendulums: int = NUM_PENDULUMS, seed: int | None = None):
    return MultiPendulumCartPoleEnv(num_pendulums=num_pendulums, seed=seed)


def make_puffer_env(
    num_pendulums: int = NUM_PENDULUMS,
    seed: int | None = None,
    buf=None,
):
    import pufferlib.emulation

    env_creator = partial(make_env, num_pendulums=num_pendulums, seed=seed)
    return pufferlib.emulation.GymnasiumPufferEnv(
        env_creator=env_creator,
        buf=buf,
        seed=0 if seed is None else seed,
    )


def make_vec_env(
    *,
    num_pendulums: int = NUM_PENDULUMS,
    num_envs: int = 1024,
    num_workers: int = 4,
    backend: str = "numpy",
    seed: int = 0,
    **kwargs: Any,
):
    backend_name = backend.lower()
    if backend_name in {"numpy", "batched", "fast"}:
        return BatchedMultiPendulumCartPoleEnv(
            num_pendulums=num_pendulums,
            num_envs=num_envs,
            seed=seed,
        )

    import pufferlib.vector

    if backend_name in {"serial", "debug"}:
        backend_obj = getattr(pufferlib.vector, "Serial", "Serial")
        creator = partial(make_puffer_env, num_pendulums=num_pendulums)
        return pufferlib.vector.make(
            creator,
            num_envs=num_envs,
            seed=seed,
            backend=backend_obj,
            **kwargs,
        )

    if backend_name in {"multiprocessing", "mp", "process"}:
        backend_obj = getattr(pufferlib.vector, "Multiprocessing", "Multiprocessing")
        creator = partial(make_puffer_env, num_pendulums=num_pendulums)
        return pufferlib.vector.make(
            creator,
            num_envs=num_envs,
            seed=seed,
            num_workers=num_workers,
            backend=backend_obj,
            **kwargs,
        )

    raise ValueError("backend must be 'numpy', 'serial', or 'multiprocessing'")
