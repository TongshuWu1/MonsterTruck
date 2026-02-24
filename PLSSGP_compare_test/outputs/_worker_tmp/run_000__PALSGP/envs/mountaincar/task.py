from __future__ import annotations

import numpy as np
import tensorflow as tf

from envs.base_task import BaseTask, EnvSpec, DTF

# ---------------------------------------------------------------------
# MountainCarContinuous goal-reaching task (no hold requirement)
#
# Success condition:
#   - position >= GOAL_POS_SUCCESS
#   - no velocity threshold requirement
#   - no hold streak requirement (SUCCESS_HOLD_STEPS = 1)
#
# The GP/MPPI suite may use a CartPole-shaped compatibility state:
#   (pos, vel, 0.0, 0.0)
# This task interprets ONLY the first two entries: [position, velocity].
# ---------------------------------------------------------------------

GOAL_POS_SUCCESS = 0.45
SUCCESS_HOLD_STEPS = 1

# Goal-reaching dense cost (your requested simple version)
# reward = -cost
GOAL_POS_COST = 0.45

# cost = max(0, goal-pos)^2 + 0.1 * vel^2 + 0.001 * u^2
W_POSITION = 1.0
W_VELOCITY = 0.1
W_ACTION = 0.001


class MountainCarTask(BaseTask):
    def __init__(
        self,
        goal_pos_success: float = GOAL_POS_SUCCESS,
        success_hold_steps: int = SUCCESS_HOLD_STEPS,
        goal_pos_cost: float = GOAL_POS_COST,
        w_position: float = W_POSITION,
        w_velocity: float = W_VELOCITY,
        w_action: float = W_ACTION,
    ):
        self.name = "mountaincar"
        self.goal_pos_success = float(goal_pos_success)
        self.success_hold_steps = int(success_hold_steps)

        # Cost parameters
        self.goal_pos_cost = float(goal_pos_cost)
        self.w_position = float(w_position)
        self.w_velocity = float(w_velocity)
        self.w_action = float(w_action)

    def spec(self):
        return EnvSpec(
            name=self.name,
            state_dim=2,  # native Continuous MountainCar obs dim (adapter may export 4D compatibility tuple)
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

    def stage_cost_np(self, s, u):
        pos, vel, u0 = self._split_np(s, u)

        # Penalize being left of the goal (no penalty once at/above goal)
        dist_to_goal = np.maximum(0.0, self.goal_pos_cost - pos)
        cost_pos = self.w_position * np.square(dist_to_goal)

        # Penalize large velocities and control effort
        cost_vel = self.w_velocity * np.square(vel)
        cost_u = self.w_action * np.square(u0)

        c = cost_pos + cost_vel + cost_u
        return np.asarray(c, dtype=np.float64)

    def stage_reward_np(self, x, u, x_next=None):
        return -self.stage_cost_np(x, u)

    @tf.function
    def stage_cost_tf(self, s, u):
        s = tf.cast(s, DTF)
        u = tf.cast(u, DTF)

        pos = s[..., 0]
        vel = s[..., 1]

        # Robust scalar-action extraction
        if u.shape.rank == 0:
            u0 = u
        elif u.shape.rank is not None and u.shape.rank > 0 and u.shape[-1] == 1:
            u0 = u[..., 0]
        else:
            u0 = u

        goal_pos = tf.constant(self.goal_pos_cost, DTF)
        zero = tf.constant(0.0, DTF)

        dist_to_goal = tf.maximum(zero, goal_pos - pos)
        cost_pos = tf.constant(self.w_position, DTF) * tf.square(dist_to_goal)
        cost_vel = tf.constant(self.w_velocity, DTF) * tf.square(vel)
        cost_u = tf.constant(self.w_action, DTF) * tf.square(u0)

        return cost_pos + cost_vel + cost_u

    @tf.function
    def stage_reward_tf(self, x, u, x_next=None):
        return -self.stage_cost_tf(x, u)

    def is_success_state(self, x, xdot, th, thdot):
        # compatibility signature; x=position for mountain car adapter
        return float(x) >= self.goal_pos_success