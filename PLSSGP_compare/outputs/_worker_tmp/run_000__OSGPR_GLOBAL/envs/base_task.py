
from __future__ import annotations
from dataclasses import dataclass
import numpy as np
import tensorflow as tf

DTF = tf.float64

@dataclass
class EnvSpec:
    name: str
    state_dim: int
    action_dim: int
    action_low: np.ndarray
    action_high: np.ndarray
    dt: float | None = None

class BaseTask:
    name = 'base'
    spec: EnvSpec | None = None

    def make_env(self, **kwargs):
        raise NotImplementedError

    # reward/cost used by BOTH environment step and MPPI planners
    def stage_reward_np(self, x, u, x_next=None):
        raise NotImplementedError

    def stage_reward_tf(self, x, u, x_next=None):
        raise NotImplementedError

    def stage_cost_tf(self, s, u):
        return -self.stage_reward_tf(s, u)

    def stage_cost_np(self, s, u):
        return -self.stage_reward_np(s, u)

    # success check hooks (used by evaluation loops/dashboard hold count)
    def is_success_state(self, x, xdot, th, thdot):
        return False

    def success_signal_np(self, x):
        x = np.asarray(x)
        if x.ndim == 1:
            return np.asarray([self.is_success_state(*x.tolist())], dtype=bool)
        return np.asarray([self.is_success_state(*row.tolist()) for row in x], dtype=bool)
