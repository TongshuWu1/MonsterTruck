from __future__ import annotations

import numpy as np
import tensorflow as tf

from envs.base_task import BaseTask, EnvSpec, DTF

# ---------------------------------------------------------------------
# MountainCarContinuous hold-at-goal task (compatibility task for the current suite)
#
# Success/hold condition (used by run loops through task_is_success_state hook):
#   - position >= GOAL_POS_SUCCESS
#   - |velocity| <= GOAL_VEL_ABS_MAX_SUCCESS
#   - hold for SUCCESS_HOLD_STEPS (=100 by default in config for this env)
#
# The GP/MPPI suite is currently CartPole-shaped (4-state / 4-head dynamics model), so the env adapter
# exports a compatibility 4-state tuple: (pos, vel, 0.0, 0.0). This task interprets the first two entries only.
# ---------------------------------------------------------------------

GOAL_POS_SUCCESS = 0.45
GOAL_VEL_ABS_MAX_SUCCESS = 0.03
SUCCESS_HOLD_STEPS = 100

# Cost weights (tuneable, reward = -cost + optional hold bonus)
W_POS = 3.0
W_VEL = 0.3
W_U = 0.05
HOLD_BONUS_WHEN_HOLDING = 0.05

# Tracking target used by stage cost
TARGET_POS = 0.52
TARGET_VEL = 0.0


class MountainCarHoldTask(BaseTask):
    def __init__(
        self,
        goal_pos_success: float = GOAL_POS_SUCCESS,
        goal_vel_abs_max_success: float = GOAL_VEL_ABS_MAX_SUCCESS,
        success_hold_steps: int = SUCCESS_HOLD_STEPS,
        target_pos: float = TARGET_POS,
        target_vel: float = TARGET_VEL,
        w_pos: float = W_POS,
        w_vel: float = W_VEL,
        w_u: float = W_U,
        hold_bonus_when_holding: float = HOLD_BONUS_WHEN_HOLDING,
    ):
        self.name = "mountaincar_hold"
        self.goal_pos_success = float(goal_pos_success)
        self.goal_vel_abs_max_success = float(goal_vel_abs_max_success)
        self.success_hold_steps = int(success_hold_steps)
        self.target_pos = float(target_pos)
        self.target_vel = float(target_vel)
        self.w_pos = float(w_pos)
        self.w_vel = float(w_vel)
        self.w_u = float(w_u)
        self.hold_bonus_when_holding = float(hold_bonus_when_holding)

    def spec(self):
        return EnvSpec(
            name=self.name,
            state_dim=4,  # compatibility state used by current suite
            action_dim=1,
            action_low=np.array([-1.0], dtype=np.float64),
            action_high=np.array([+1.0], dtype=np.float64),
            dt=None,
        )

    def _split_np(self, x, u):
        x = np.asarray(x, dtype=np.float64)
        u = np.asarray(u, dtype=np.float64)
        if x.shape[-1] < 2:
            raise ValueError(f"x must have at least 2 dims (pos,vel), got {x.shape}")
        if u.ndim == 0:
            u0 = u
        elif u.shape[-1] == 1:
            u0 = u[..., 0]
        else:
            u0 = u
        pos = x[..., 0]
        vel = x[..., 1]
        return pos, vel, u0

    def _hold_mask_np(self, x):
        x = np.asarray(x, dtype=np.float64)
        pos = x[..., 0]
        vel = x[..., 1]
        return (pos >= self.goal_pos_success) & (np.abs(vel) <= self.goal_vel_abs_max_success)

    def stage_cost_np(self, s, u):
        pos, vel, u0 = self._split_np(s, u)
        c = (
            self.w_pos * np.square(pos - self.target_pos)
            + self.w_vel * np.square(vel - self.target_vel)
            + self.w_u * np.square(u0)
        )
        if self.hold_bonus_when_holding != 0.0:
            c = c - self.hold_bonus_when_holding * self._hold_mask_np(s).astype(np.float64)
        return np.asarray(c, dtype=np.float64)

    def stage_reward_np(self, x, u, x_next=None):
        return -self.stage_cost_np(x, u)

    @tf.function
    def stage_cost_tf(self, s, u):
        s = tf.cast(s, DTF)
        u = tf.cast(u, DTF)
        pos = s[..., 0]
        vel = s[..., 1]

        if u.shape.rank is not None and u.shape.rank > 0 and u.shape[-1] == 1:
            u0 = u[..., 0]
        else:
            u0 = u

        c = (
            tf.constant(self.w_pos, DTF) * tf.square(pos - tf.constant(self.target_pos, DTF))
            + tf.constant(self.w_vel, DTF) * tf.square(vel - tf.constant(self.target_vel, DTF))
            + tf.constant(self.w_u, DTF) * tf.square(u0)
        )
        if self.hold_bonus_when_holding != 0.0:
            hold_mask = tf.logical_and(
                pos >= tf.constant(self.goal_pos_success, DTF),
                tf.abs(vel) <= tf.constant(self.goal_vel_abs_max_success, DTF),
            )
            c = c - tf.constant(self.hold_bonus_when_holding, DTF) * tf.cast(hold_mask, DTF)
        return c

    @tf.function
    def stage_reward_tf(self, x, u, x_next=None):
        return -self.stage_cost_tf(x, u)

    def is_success_state(self, x, xdot, th, thdot):
        # compatibility signature; x=position, xdot=velocity for mountain car adapter
        return (float(x) >= self.goal_pos_success) and (abs(float(xdot)) <= self.goal_vel_abs_max_success)
