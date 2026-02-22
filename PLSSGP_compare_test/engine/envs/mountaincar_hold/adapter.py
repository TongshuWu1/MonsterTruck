# Backward-compatibility adapter. Prefer using envs/mountaincar_hold/env.py directly.
from envs.mountaincar_hold.env import *  # noqa: F401,F403
from .task import MountainCarHoldTask
