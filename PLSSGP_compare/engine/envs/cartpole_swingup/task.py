from __future__ import annotations

import numpy as np
import tensorflow as tf

from envs.base_task import BaseTask, EnvSpec, DTF

# ---------------------------------------------------------------------
# CartPole swing-up task definition
# These defaults are intentionally aligned with the user's notebook:
#   MPPI stage_cost_tf(s,u) = 8*(1-cos(th)) + 0.8*x^2 + 0.08*xdot^2 + 0.08*thdot^2 + 0.02*u^2
# Reward is defined as the negative of that cost so env.step reward and MPPI cost are consistent.
# ---------------------------------------------------------------------

# Exact success thresholds from notebook (used in run loops / hold logic)
SUCCESS_COS_TH_MIN = 0.98
SUCCESS_X_ABS_MAX = 0.35
SUCCESS_XDOT_ABS_MAX = 1.0
SUCCESS_THDOT_ABS_MAX = 2.5
SUCCESS_HOLD_STEPS = 100

# Exact notebook MPPI stage cost weights
W_THETA = 5.0
W_X = 2.0
W_XDOT = 0.02
W_THDOT = 0.02
W_U = 0.1


class CartPoleSwingUpTask(BaseTask):
    """
    Shared CartPole swing-up task logic used by both env.step reward and MPPI planning.

    IMPORTANT:
    - stage_cost_* is exactly the same functional form as the notebook MPPI cost.
    - stage_reward_* = -stage_cost_* so env reward and planner objective stay consistent.
    """

    def __init__(
        self,
        w_theta: float = W_THETA,
        w_x: float = W_X,
        w_xdot: float = W_XDOT,
        w_thdot: float = W_THDOT,
        w_u: float = W_U,
        success_cos_th_min: float = SUCCESS_COS_TH_MIN,
        success_x_abs_max: float = SUCCESS_X_ABS_MAX,
        success_xdot_abs_max: float = SUCCESS_XDOT_ABS_MAX,
        success_thdot_abs_max: float = SUCCESS_THDOT_ABS_MAX,
        success_hold_steps: int = SUCCESS_HOLD_STEPS,
    ):
        self.name = "cartpole_swingup"

        # cost weights (match notebook stage_cost_tf)
        self.w_theta = float(w_theta)
        self.w_x = float(w_x)
        self.w_xdot = float(w_xdot)
        self.w_thdot = float(w_thdot)
        self.w_u = float(w_u)

        # success thresholds (match notebook)
        self.success_cos_th_min = float(success_cos_th_min)
        self.success_x_abs_max = float(success_x_abs_max)
        self.success_xdot_abs_max = float(success_xdot_abs_max)
        self.success_thdot_abs_max = float(success_thdot_abs_max)
        self.success_hold_steps = int(success_hold_steps)

    def spec(self):
        return EnvSpec(
            name=self.name,
            state_dim=4,
            action_dim=1,
            action_low=np.array([-1.0], dtype=np.float64),
            action_high=np.array([+1.0], dtype=np.float64),
            dt=None,
        )

    # -------------------------
    # helpers for shapes
    # -------------------------
    def _split_np(self, x, u):
        x = np.asarray(x, dtype=np.float64)
        u = np.asarray(u, dtype=np.float64)
        if x.shape[-1] != 4:
            raise ValueError(f"x must have last dim 4, got {x.shape}")

        if u.ndim == 0:
            u0 = u
        elif u.shape[-1] == 1:
            u0 = u[..., 0]
        else:
            u0 = u

        return x[..., 0], x[..., 1], x[..., 2], x[..., 3], u0

    def _stage_cost_np_array(self, s, u):
        x, xdot, th, thdot, u0 = self._split_np(s, u)
        c = (
            self.w_theta * (1.0 - np.cos(th))
            + self.w_x * (x * x)
            + self.w_xdot * (xdot * xdot)
            + self.w_thdot * (thdot * thdot)
            + self.w_u * (u0 * u0)
        )
        return np.asarray(c, dtype=np.float64)

    # -------------------------
    # reward / cost (shared)
    # -------------------------
    def stage_cost_np(self, s, u):
        return self._stage_cost_np_array(s, u)

    def stage_reward_np(self, x, u, x_next=None):
        return -self._stage_cost_np_array(x, u)

    @tf.function
    def stage_cost_tf(self, s, u):
        s = tf.cast(s, DTF)
        u = tf.cast(u, DTF)

        x = s[..., 0]
        xdot = s[..., 1]
        th = s[..., 2]
        thdot = s[..., 3]

        # Support u shaped (...,), (...,1), or scalar
        if u.shape.rank is not None and u.shape.rank > 0 and u.shape[-1] == 1:
            u0 = u[..., 0]
        else:
            u0 = u

        return (
            tf.constant(self.w_theta, DTF) * (tf.constant(1.0, DTF) - tf.cos(th))
            + tf.constant(self.w_x, DTF) * tf.square(x)
            + tf.constant(self.w_xdot, DTF) * tf.square(xdot)
            + tf.constant(self.w_thdot, DTF) * tf.square(thdot)
            + tf.constant(self.w_u, DTF) * tf.square(u0)
        )

    @tf.function
    def stage_reward_tf(self, x, u, x_next=None):
        # Reward is exactly the negative of notebook MPPI stage cost for consistency.
        return -self.stage_cost_tf(x, u)

    # -------------------------
    # success / hold logic
    # -------------------------
    def is_success_state(self, x, xdot, th, thdot):
        return (
            (np.cos(th) >= self.success_cos_th_min)
            and (abs(x) <= self.success_x_abs_max)
            and (abs(xdot) <= self.success_xdot_abs_max)
            and (abs(thdot) <= self.success_thdot_abs_max)
        )
