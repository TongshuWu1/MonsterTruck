# Canonical CartPole swing-up environment module (moved under engine/envs/cartpole_swingup/)
# This file contains the actual implementation used by the suite.
# A backward-compatibility shim remains at engine/carpole_env.py.

# ============================
# Cell 1 — Imports + Custom Env (Continuous CartPole Swing-Up) + EDGE RESPAWN
# + TensorFlow GPU setup (for GPflow/TF compute)
#
# Notes (clean + consistent with later cells):
#   - Env observations/actions stay float32 (Gym/Gymnasium standard).
#   - GP / feature pipeline uses float64 (matches your later GPflow/TF float64 setup).
# ============================

import warnings
warnings.filterwarnings("ignore", message="pkg_resources is deprecated as an API.*", category=UserWarning)

import os
import math
import numpy as np
import gymnasium as gym
from gymnasium import spaces
from gymnasium.wrappers import TimeLimit
from gymnasium.utils import seeding

# ---- numpy bool8 compat ----
if not hasattr(np, "bool8"):
    np.bool8 = np.bool_

# ============================================================
# TensorFlow GPU setup
# ============================================================
os.environ.setdefault("TF_CPP_MIN_LOG_LEVEL", "2")  # quieter TF logs

import tensorflow as tf

gpus = tf.config.list_physical_devices("GPU")
if gpus:
    for gpu in gpus:
        tf.config.experimental.set_memory_growth(gpu, True)
    print("✅ TF version:", tf.__version__)
    print("✅ Built with CUDA:", tf.test.is_built_with_cuda())
    print("✅ GPUs:", gpus)
else:
    print("⚠️ TF version:", tf.__version__)
    print("⚠️ Built with CUDA:", tf.test.is_built_with_cuda())
    print("⚠️ GPUs: [] (TensorFlow will run on CPU)")

# ============================================================
# Dtypes
# ============================================================
DTYPE_ENV = np.float32     # env obs/action dtype
DTYPE_GP  = np.float64     # GP/feature dtype (consistent with your float64 GPflow pipeline)

# ============================================================
# Shared task logic (reward/cost/success) for consistency with MPPI
# ============================================================
try:
    from envs.registry import make_task
    TASK = make_task("cartpole_swingup")
    # Export hooks into experiments.py global namespace (controllers.py picks these up)
    task_stage_cost_tf = TASK.stage_cost_tf
    task_is_success_state = TASK.is_success_state
except Exception as _task_exc:
    TASK = None
    task_stage_cost_tf = None
    task_is_success_state = None
    print(f"⚠️ Task registry unavailable in envs/cartpole_swingup/env.py, falling back to inline reward/success. ({_task_exc})")

# ============================================================
# Angle helpers
# ============================================================
def wrap_pi(x):
    return (x + np.pi) % (2 * np.pi) - np.pi

def obs_to_state(obs):
    """
    obs = [x, x_dot, theta, theta_dot]
    Wrap theta to (-pi, pi] for stability.
    """
    x, xdot, th, thdot = float(obs[0]), float(obs[1]), float(obs[2]), float(obs[3])
    th = wrap_pi(th)
    return x, xdot, th, thdot

U_MIN, U_MAX = -1.0, 1.0

def state_to_features(x, xdot, theta, thetadot, u,
                      x_scale=2.4, v_scale=3.0, w_scale=8.0,
                      dtype=DTYPE_GP):
    """
    GP features (D=6), bounded:
        [ tanh(x/x_scale),
          tanh(xdot/v_scale),
          sin(theta),
          cos(theta),
          tanh(thetadot/w_scale),
          u ]
    Returns float64 by default (matches GPflow default float64 used later).
    """
    x_feat = np.tanh(x / x_scale)
    xdot_feat = np.tanh(xdot / v_scale)
    w_feat = np.tanh(thetadot / w_scale)
    return np.array(
        [x_feat, xdot_feat, np.sin(theta), np.cos(theta), w_feat, float(u)],
        dtype=dtype
    )



# ============================================================
# Shared reward helper (used by env.step and available for debugging)
# ============================================================
def cartpole_stage_reward(x, x_dot, theta, theta_dot, u):
    """
    Shared reward helper for env.step.

    IMPORTANT: keep this aligned with the notebook MPPI stage cost:
        c = 8*(1-cos(theta)) + 0.8*x^2 + 0.08*xdot^2 + 0.08*theta_dot^2 + 0.02*u^2
    We return reward = -c so environment reward and MPPI planning objective are consistent.
    """
    if TASK is not None and hasattr(TASK, "stage_reward_np"):
        s = np.array([x, x_dot, theta, theta_dot], dtype=np.float64)
        uu = np.array([u], dtype=np.float64)
        return float(np.asarray(TASK.stage_reward_np(s, uu)).reshape(-1)[0])

    cost = (
        8.0 * (1.0 - math.cos(theta))
        + 0.8 * (x * x)
        + 0.08 * (x_dot * x_dot)
        + 0.08 * (theta_dot * theta_dot)
        + 0.02 * (u * u)
    )
    return float(-cost)

# ============================================================
# Custom Continuous CartPole Swing-Up Env (CPU physics)
# ============================================================
class ContinuousCartPoleSwingUpEnv(gym.Env):
    metadata = {"render_modes": ["rgb_array", "human"], "render_fps": 50}

    def __init__(self, render_mode=None, start_down=True):
        super().__init__()

        # physics
        self.gravity = 9.8
        self.masscart = 1.0
        self.masspole = 0.1
        self.total_mass = self.masspole + self.masscart
        self.length = 0.5
        self.polemass_length = self.masspole * self.length

        # control
        self.force_mag = 30.0
        self.tau = 0.02
        self.min_action = -1.0
        self.max_action = 1.0

        # track limits
        self.x_threshold = 2.4

        # reset mode
        self.start_down = bool(start_down)

        # render
        self.render_mode = render_mode
        self.state = None
        self.np_random = None
        self.seed()

        # spaces
        high = np.array(
            [
                self.x_threshold * 2,
                np.finfo(np.float32).max,
                np.finfo(np.float32).max,
                np.finfo(np.float32).max,
            ],
            dtype=np.float32,
        )

        self.action_space = spaces.Box(
            low=np.array([self.min_action], dtype=np.float32),
            high=np.array([self.max_action], dtype=np.float32),
            shape=(1,),
            dtype=np.float32,
        )
        self.observation_space = spaces.Box(-high, high, dtype=np.float32)

    def seed(self, seed=None):
        self.np_random, seed = seeding.np_random(seed)
        return [seed]

    def stepPhysics(self, force):
        x, x_dot, theta, theta_dot = self.state
        costheta = math.cos(theta)
        sintheta = math.sin(theta)

        temp = (force + self.polemass_length * theta_dot * theta_dot * sintheta) / self.total_mass
        thetaacc = (self.gravity * sintheta - costheta * temp) / (
            self.length * (4.0 / 3.0 - self.masspole * costheta * costheta / self.total_mass)
        )
        xacc = temp - self.polemass_length * thetaacc * costheta / self.total_mass

        x = x + self.tau * x_dot
        x_dot = x_dot + self.tau * xacc
        theta = theta + self.tau * theta_dot
        theta_dot = theta_dot + self.tau * thetaacc
        return (x, x_dot, theta, theta_dot)

    def step(self, action):
        action = np.asarray(action, dtype=np.float32).reshape(1,)
        assert self.action_space.contains(action), f"{action} invalid"

        u = float(action[0])
        force = self.force_mag * u

        self.state = self.stepPhysics(force)
        x, x_dot, theta, theta_dot = self.state

        terminated = bool(x < -self.x_threshold or x > self.x_threshold)
        truncated = False  # TimeLimit handles truncation
        reward = cartpole_stage_reward(x, x_dot, theta, theta_dot, u)

        obs = np.array([x, x_dot, theta, theta_dot], dtype=np.float32)
        info = dict(x=x, x_dot=x_dot, theta=theta, theta_dot=theta_dot, u=u)
        return obs, float(reward), terminated, truncated, info

    def reset(self, seed=None, options=None):
        super().reset(seed=seed)
        if seed is not None:
            self.seed(seed)

        x = float(self.np_random.uniform(low=-0.05, high=0.05))
        x_dot = float(self.np_random.uniform(low=-0.05, high=0.05))
        theta_dot = float(self.np_random.uniform(low=-0.05, high=0.05))

        if self.start_down:
            theta = float(math.pi + self.np_random.uniform(low=-0.10, high=0.10))
        else:
            theta = float(self.np_random.uniform(low=-0.10, high=0.10))

        self.state = (x, x_dot, theta, theta_dot)
        obs = np.array(self.state, dtype=np.float32)
        info = {}
        return obs, info

    def render(self):
        return None

    def close(self):
        return None

# ============================================================
# Edge respawn wrapper
# ============================================================
class EdgeRespawnWrapper(gym.Wrapper):
    def __init__(
        self,
        env,
        respawn_penalty=-2.0,
        reset_seed_mode="random",
        seed=0,
        suppress_reward_bonus_after_respawn_steps=1,
        suppress_reward_after_respawn_mode="add_penalty",
    ):
        super().__init__(env)
        self.respawn_penalty = float(respawn_penalty)
        self.reset_seed_mode = str(reset_seed_mode)
        self._rng = np.random.default_rng(seed)

        # Prevent a reward spike immediately after edge respawn (centered cart x bonus).
        # Modes:
        #   - "override_penalty": reward = respawn_penalty
        #   - "add_penalty"     : reward += respawn_penalty
        #   - "none"            : no suppression
        self.suppress_reward_bonus_after_respawn_steps = int(max(0, suppress_reward_bonus_after_respawn_steps))
        self.suppress_reward_after_respawn_mode = str(suppress_reward_after_respawn_mode)
        self._post_respawn_cooldown = 0

    def reset(self, *, seed=None, options=None):
        if seed is not None:
            self._rng = np.random.default_rng(seed)
        self._post_respawn_cooldown = 0
        return self.env.reset(seed=seed, options=options)

    def step(self, action):
        obs, reward, terminated, truncated, info = self.env.step(action)
        reward = float(reward)
        info = dict(info)

        # Suppress immediate post-respawn reward "bonus" (e.g., centered x after respawn).
        if self._post_respawn_cooldown > 0:
            info["post_respawn_cooldown"] = int(self._post_respawn_cooldown)
            mode = self.suppress_reward_after_respawn_mode
            if mode == "override_penalty":
                reward = float(self.respawn_penalty)
            elif mode == "add_penalty":
                reward = reward + float(self.respawn_penalty)
            self._post_respawn_cooldown -= 1
        else:
            info["post_respawn_cooldown"] = 0

        if terminated:
            info["respawned"] = True
            reward = reward + float(self.respawn_penalty)

            seed = int(self._rng.integers(0, 10**9)) if self.reset_seed_mode == "random" else None
            obs, _ = self.env.reset(seed=seed)

            # Also suppress the first step after the respawned reset state.
            self._post_respawn_cooldown = int(self.suppress_reward_bonus_after_respawn_steps)

            terminated = False
            truncated = False
        else:
            info["respawned"] = bool(info.get("respawned", False))

        return obs, float(reward), bool(terminated), bool(truncated), info

# ============================================================
# Environment factory
# ============================================================
MAX_EPISODE_STEPS = 600

def make_env(
    render_mode=None,
    seed=0,
    max_episode_steps=MAX_EPISODE_STEPS,
    start_down=True,
    edge_respawn=True,
    respawn_penalty=-2.0,
):
    env = ContinuousCartPoleSwingUpEnv(render_mode=render_mode, start_down=start_down)
    env = TimeLimit(env, max_episode_steps=max_episode_steps)
    if edge_respawn:
        env = EdgeRespawnWrapper(env, respawn_penalty=respawn_penalty, seed=seed)
    env.reset(seed=seed)
    return env

# ============================================================
# Sanity check
# ============================================================
env = make_env(render_mode=None, seed=0, start_down=True, edge_respawn=True)
obs, _ = env.reset(seed=0)
s = obs_to_state(obs)

print("✅ Env ready (edge_respawn=True)")
print("obs:", obs)
print("state:", s)
print("action space:", env.action_space)
env.close()
