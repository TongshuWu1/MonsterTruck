
from __future__ import annotations

def make_task(name: str, **kwargs):
    key = str(name).lower()
    if key in ('cartpole', 'cartpole_swingup', 'carpole', 'carpole_swingup'):
        from .cartpole_swingup.task import CartPoleSwingUpTask
        return CartPoleSwingUpTask(**kwargs)
    raise ValueError(f"Unknown task/env name: {name}")

def list_tasks():
    return ['cartpole_swingup']
