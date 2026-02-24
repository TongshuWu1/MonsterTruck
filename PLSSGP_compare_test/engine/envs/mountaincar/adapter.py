# Backward-compatibility adapter for normal MountainCarContinuous.
from envs.mountaincar.env import *  # noqa: F401,F403
from .task import MountainCarTask
