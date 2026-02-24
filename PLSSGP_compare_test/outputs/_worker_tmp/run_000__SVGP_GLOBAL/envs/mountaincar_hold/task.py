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
#   - hold for SUCCESS_HOLD_STEPS (=200 by default in config for this env)
#
# The GP/MPPI suite is currently CartPole-shaped (4-state / 4-head dynamics model), so the env adapter
# exports a compatibility 4-state tuple: (pos, vel, 0.0, 0.0). This task interprets the first two entries only.
# ---------------------------------------------------------------------

GOAL_POS_SUCCESS = 0.45
GOAL_VEL_ABS_MAX_SUCCESS = 0.02
SUCCESS_HOLD_STEPS = 100

# Hold cost (STOP-style base)
# reward = -cost
GOAL_POS_COST = 0.45
HOLD_POS_TOL = 0.05
HOLD_VEL_TOL = 0.02

W_POSITION = 20.0
W_VELOCITY = 0.0          # base v^2 damping (extra damping near goal still comes from STOP_BOOST)
W_ENERGY = 0.05
W_STOP_BOOST = 200.0
W_TIME = 0.1
STEADY_BONUS_COST = -50.0  # negative cost => reward bonus when holding

# Simple distance-based shaping (keep it simple)
PROX_K = 8.0               # proximity scale sharpness: prox=exp(-PROX_K*dist)
ENERGY_FAR_SCALE = 0.2     # action penalty scale far from goal (near goal goes to 1.0)
W_VEL_DIR = 6.0            # directional velocity encouragement strength
VEL_DIR_DIST_SCALE = 0.15  # tanh((goal-pos)/VEL_DIR_DIST_SCALE)


class MountainCarHoldTask(BaseTask):
    def __init__(
        self,
        goal_pos_success: float = GOAL_POS_SUCCESS,
        goal_vel_abs_max_success: float = GOAL_VEL_ABS_MAX_SUCCESS,
        success_hold_steps: int = SUCCESS_HOLD_STEPS,
        goal_pos_cost: float = GOAL_POS_COST,
        hold_pos_tol: float = HOLD_POS_TOL,
        hold_vel_tol: float = HOLD_VEL_TOL,
        w_position: float = W_POSITION,
        w_velocity: float = W_VELOCITY,
        w_energy: float = W_ENERGY,
        w_stop_boost: float = W_STOP_BOOST,
        w_time: float = W_TIME,
        steady_bonus_cost: float = STEADY_BONUS_COST,
        prox_k: float = PROX_K,
        energy_far_scale: float = ENERGY_FAR_SCALE,
        w_vel_dir: float = W_VEL_DIR,
        vel_dir_dist_scale: float = VEL_DIR_DIST_SCALE,
    ):
        self.name = "mountaincar_hold"
        self.goal_pos_success = float(goal_pos_success)
        self.goal_vel_abs_max_success = float(goal_vel_abs_max_success)
        self.success_hold_steps = int(success_hold_steps)

        # Cost parameters
        self.goal_pos_cost = float(goal_pos_cost)
        self.hold_pos_tol = float(hold_pos_tol)
        self.hold_vel_tol = float(hold_vel_tol)
        self.w_position = float(w_position)
        self.w_velocity = float(w_velocity)
        self.w_energy = float(w_energy)
        self.w_stop_boost = float(w_stop_boost)
        self.w_time = float(w_time)
        self.steady_bonus_cost = float(steady_bonus_cost)

        # Simple distance-based shaping knobs
        self.prox_k = float(prox_k)
        self.energy_far_scale = float(energy_far_scale)
        self.w_vel_dir = float(w_vel_dir)
        self.vel_dir_dist_scale = float(vel_dir_dist_scale)

    def spec(self):
        return EnvSpec(
            name=self.name,
            state_dim=2,  # native Continuous MountainCar obs dim (adapter still exports 4D compatibility tuple)
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
        return (np.abs(pos - self.goal_pos_cost) < self.hold_pos_tol) & (np.abs(vel) < self.hold_vel_tol)

    def stage_cost_np(self, s, u):
        pos, vel, u0 = self._split_np(s, u)
        dist = np.abs(pos - self.goal_pos_cost)

        # proximity-to-goal scale: ~0 far from goal, ~1 near goal
        prox = np.exp(-self.prox_k * dist)

        # position shaping (same style)
        pos_term = self.w_position * np.square(np.tanh(dist * 2.0))

        # Velocity damping:
        #   weak far from goal, strong near goal (plus strong stop boost right near goal)
        vel_weight = (
            (0.1 * self.w_velocity)
            + (self.w_velocity * prox)
            + self.w_stop_boost * np.exp(-20.0 * dist)
        )
        vel_damp_term = vel_weight * np.square(vel)

        # Directional velocity encouragement (in COST):
        #   left of goal (pos<goal): encourage +vel  -> lowers cost
        #   at goal: no encouragement
        #   beyond goal (pos>goal): encourage -vel -> lowers cost
        dir_scale = np.tanh((self.goal_pos_cost - pos) / max(1e-8, self.vel_dir_dist_scale))

        # fade this out near the goal so it won't fight stopping/holding
        vel_dir_term = -(1.0 - prox) * self.w_vel_dir * dir_scale * vel

        vel_term = vel_damp_term + vel_dir_term

        # Action penalty scaled by distance:
        #   smaller far away (allow pumping), stronger near goal (settle/hold)
        energy_weight = self.w_energy * (self.energy_far_scale + (1.0 - self.energy_far_scale) * prox)
        energy_term = energy_weight * np.square(u0)

        time_term = self.w_time * np.ones_like(dist, dtype=np.float64)

        steady_term = np.where(
            (dist < self.hold_pos_tol) & (np.abs(vel) < self.hold_vel_tol),
            self.steady_bonus_cost,
            0.0,
        )

        c = pos_term + vel_term + energy_term + time_term + steady_term
        return np.asarray(c, dtype=np.float64)

    def stage_reward_np(self, x, u, x_next=None):
        return -self.stage_cost_np(x, u)

    @tf.function
    def stage_cost_tf(self, s, u):
        s = tf.cast(s, DTF)
        u = tf.cast(u, DTF)
        pos = s[..., 0]
        vel = s[..., 1]

        if u.shape.rank == 0:
            u0 = u
        elif u.shape.rank is not None and u.shape.rank > 0 and u.shape[-1] == 1:
            u0 = u[..., 0]
        else:
            u0 = u

        goal_pos = tf.constant(self.goal_pos_cost, DTF)
        dist = tf.abs(pos - goal_pos)

        prox = tf.exp(-tf.constant(self.prox_k, DTF) * dist)

        pos_term = tf.constant(self.w_position, DTF) * tf.square(
            tf.tanh(dist * tf.constant(2.0, DTF))
        )

        vel_weight = (
            tf.constant(0.1 * self.w_velocity, DTF)
            + tf.constant(self.w_velocity, DTF) * prox
            + tf.constant(self.w_stop_boost, DTF) * tf.exp(-tf.constant(20.0, DTF) * dist)
        )
        vel_damp_term = vel_weight * tf.square(vel)

        dir_scale = tf.tanh(
            (goal_pos - pos) / tf.constant(max(1e-8, self.vel_dir_dist_scale), DTF)
        )
        vel_dir_term = (
            -(tf.constant(1.0, DTF) - prox)
            * tf.constant(self.w_vel_dir, DTF)
            * dir_scale
            * vel
        )

        vel_term = vel_damp_term + vel_dir_term

        energy_weight = tf.constant(self.w_energy, DTF) * (
            tf.constant(self.energy_far_scale, DTF)
            + (tf.constant(1.0, DTF) - tf.constant(self.energy_far_scale, DTF)) * prox
        )
        energy_term = energy_weight * tf.square(u0)

        time_term = tf.ones_like(dist, dtype=DTF) * tf.constant(self.w_time, DTF)

        hold_mask = tf.logical_and(
            dist < tf.constant(self.hold_pos_tol, DTF),
            tf.abs(vel) < tf.constant(self.hold_vel_tol, DTF),
        )
        steady_term = tf.where(
            hold_mask,
            tf.ones_like(dist, dtype=DTF) * tf.constant(self.steady_bonus_cost, DTF),
            tf.zeros_like(dist, dtype=DTF),
        )

        return pos_term + vel_term + energy_term + time_term + steady_term

    @tf.function
    def stage_reward_tf(self, x, u, x_next=None):
        return -self.stage_cost_tf(x, u)

    def is_success_state(self, x, xdot, th, thdot):
        # compatibility signature; x=position, xdot=velocity for mountain car adapter
        return (float(x) >= self.goal_pos_success) and (abs(float(xdot)) <= self.goal_vel_abs_max_success)