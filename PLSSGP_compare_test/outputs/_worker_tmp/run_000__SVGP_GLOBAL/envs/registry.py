
from __future__ import annotations

def make_task(name: str, **kwargs):
    key = str(name).lower()
    if key in ('cartpole', 'cartpole_swingup', 'carpole', 'carpole_swingup'):
        from .cartpole_swingup.task import CartPoleSwingUpTask
        return CartPoleSwingUpTask(**kwargs)
    if key in ('mountaincar', 'mountaincar_hold', 'mountaincar_continuous_hold', 'continuous_mountaincar', 'continuous_mountaincar_hold', 'mountain_car'):
        from .mountaincar_hold.task import MountainCarHoldTask
        return MountainCarHoldTask(**kwargs)
    raise ValueError(f"Unknown task/env name: {name}")

def list_tasks():
    return ['cartpole_swingup', 'mountaincar_hold']
