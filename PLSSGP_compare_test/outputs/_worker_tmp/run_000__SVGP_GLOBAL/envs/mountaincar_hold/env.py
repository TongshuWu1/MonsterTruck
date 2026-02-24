# MountainCarContinuous hold-at-goal environment module for the PLSSGP comparison suite.
#
# IMPORTANT COMPATIBILITY NOTE:
# The current GP/MPPI pipeline is CartPole-shaped (4-state, 4 GP heads, 6D features). To avoid rewriting the
# whole planner/dynamics code, this env adapter exports a compatibility state:
#   (pos, vel, 0.0, 0.0)
# The first two dimensions are the true MountainCar state; the last two are dummy placeholders.
#
# This lets you switch ENV_MODULE in config.py now, while keeping future refactors easier.

import os
import math
import numpy as np
try:
    import gymnasium as gym
    from gymnasium.wrappers import TimeLimit
except Exception:
    import gym as gym
    from gym.wrappers import TimeLimit

# TensorFlow GPU setup (mirrors cartpole env module style)
os.environ.setdefault("TF_CPP_MIN_LOG_LEVEL", "2")
import tensorflow as tf

if not hasattr(np, "bool8"):
    np.bool8 = np.bool_

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

DTYPE_ENV = np.float32
DTYPE_GP = np.float64

# Native Continuous_MountainCar dimensions (methods stay unchanged via compatibility adapter below)
NATIVE_OBS_DIM = 2
COMPAT_STATE_DIM = 4
ACTION_DIM = 1

try:
    from envs.registry import make_task
    TASK = make_task("mountaincar_hold")
    task_stage_cost_tf = TASK.stage_cost_tf
    task_is_success_state = TASK.is_success_state
except Exception as _task_exc:
    TASK = None
    task_stage_cost_tf = None
    task_is_success_state = None
    print(f"⚠️ Task registry unavailable in envs/mountaincar_hold/env.py ({_task_exc})")

# Compatibility action limits
U_MIN, U_MAX = -1.0, 1.0

# -----------------------------
# Compatibility helpers (same names expected by experiments.py)
# -----------------------------
def wrap_pi(x):
    # dummy angle channel stays zero; keep a generic wrap for compatibility
    return (x + np.pi) % (2 * np.pi) - np.pi

def obs_to_state(obs):
    """
    obs from MountainCarContinuous-v0 is [position, velocity].
    Return compatibility 4-tuple expected by the current suite:
      (pos, vel, 0.0, 0.0)
    """
    pos = float(obs[0]); vel = float(obs[1])
    return pos, vel, 0.0, 0.0

def state_to_features(x, xdot, theta, thetadot, u,
                      x_scale=1.2, v_scale=0.08, w_scale=1.0,
                      dtype=DTYPE_GP):
    """
    Compatibility 6D feature map matching the suite's fixed dimensionality.

    We encode mountain car state into the first two + a sinusoidal position basis:
      [ tanh(pos/x_scale), tanh(vel/v_scale), sin(pi*pos), cos(pi*pos), 0.0, u ]
    The 5th channel is a dummy 'thetadot-like' slot kept at 0 to preserve shape.
    """
    pos_feat = np.tanh(float(x) / x_scale)
    vel_feat = np.tanh(float(xdot) / v_scale)
    ang = math.pi * float(x)
    return np.array([
        pos_feat,
        vel_feat,
        math.sin(ang),
        math.cos(ang),
        0.0,
        float(u),
    ], dtype=dtype)

# -----------------------------
# Shared reward helper
# -----------------------------
def mountaincar_stage_reward(pos, vel, u):
    if TASK is not None and hasattr(TASK, "stage_reward_np"):
        s = np.array([pos, vel, 0.0, 0.0], dtype=np.float64)
        uu = np.array([u], dtype=np.float64)
        return float(np.asarray(TASK.stage_reward_np(s, uu)).reshape(-1)[0])
    # fallback simple shaping
    return float(- (3.0 * (pos - 0.52) ** 2 + 0.3 * (vel ** 2) + 0.05 * (u ** 2)))

# -----------------------------
# Optional simple PIL debug renderer for Cell 2 random collection animation
# -----------------------------
def render_state_frame_from_state(x, theta=None, x_threshold=None, W=720, H=450, **kwargs):
    """Use the *actual* Gym/Gymnasium MountainCarContinuous renderer (rgb_array).

    This is a visualization helper for experiments.py (CartPole-era API compatibility).
    Only position is guaranteed at the call site; velocity may be passed via kwargs['vel'].
    """
    global _MC_RGB_RENDER_ENV

    pos = float(x)
    # experiments.py passes theta as the 2nd arg (dummy for this env), so do not trust it as velocity
    vel = float(kwargs.get("vel", 0.0))
    # clamp to valid native state range
    pos = float(np.clip(pos, -1.2, 0.6))
    vel = float(np.clip(vel, -0.07, 0.07))

    try:
        _MC_RGB_RENDER_ENV
    except NameError:
        _MC_RGB_RENDER_ENV = None

    try:
        if _MC_RGB_RENDER_ENV is None:
            try:
                _MC_RGB_RENDER_ENV = gym.make("MountainCarContinuous-v0", render_mode="rgb_array")
            except TypeError:
                _MC_RGB_RENDER_ENV = gym.make("MountainCarContinuous-v0")
            try:
                _MC_RGB_RENDER_ENV.reset(seed=0)
            except Exception:
                pass

        base = _MC_RGB_RENDER_ENV.unwrapped if hasattr(_MC_RGB_RENDER_ENV, "unwrapped") else _MC_RGB_RENDER_ENV
        # MountainCarContinuous uses a 2D native state: [position, velocity]
        if hasattr(base, "state"):
            base.state = np.array([pos, vel], dtype=np.float32)
        else:
            # Unexpected implementation; bail out so caller can continue without crashing.
            return None

        frame = _MC_RGB_RENDER_ENV.render()
        if frame is None:
            return None
        frame = np.asarray(frame, dtype=np.uint8)

        if int(W) > 0 and int(H) > 0 and (frame.shape[1] != int(W) or frame.shape[0] != int(H)):
            try:
                from PIL import Image
                frame = np.asarray(Image.fromarray(frame).resize((int(W), int(H))))
            except Exception:
                pass
        return frame
    except Exception:
        # Last-resort fallback: return None rather than breaking training/collection.
        return None

render_cartpole_frame_from_state = render_state_frame_from_state

# -----------------------------
# Gym env wrapper with reward override (uses task reward)
# -----------------------------
class MountainCarHoldRewardWrapper(gym.Wrapper):
    def __init__(self, env):
        super().__init__(env)
        self.last_obs = None

    def reset(self, *, seed=None, options=None):
        obs, info = self.env.reset(seed=seed, options=options)
        self.last_obs = np.asarray(obs, dtype=np.float32)
        return obs, info

    def step(self, action):
        action = np.asarray(action, dtype=np.float32).reshape(1,)
        obs, _reward_native, terminated, truncated, info = self.env.step(action)
        pos, vel, _, _ = obs_to_state(obs)
        u = float(action[0])
        reward = mountaincar_stage_reward(pos, vel, u)
        self.last_obs = np.asarray(obs, dtype=np.float32)
        info = dict(info)
        info.update(dict(position=pos, velocity=vel, u=u))
        return np.asarray(obs, dtype=np.float32), float(reward), bool(terminated), bool(truncated), info

class CompatStateInfoWrapper(gym.Wrapper):
    """Optionally inject a compatibility state tuple into info for debugging."""
    def step(self, action):
        obs, reward, terminated, truncated, info = self.env.step(action)
        info = dict(info)
        info['compat_state4'] = obs_to_state(obs)
        return obs, reward, terminated, truncated, info

MAX_EPISODE_STEPS = 600

def make_env(render_mode=None, seed=0, max_episode_steps=MAX_EPISODE_STEPS,
             start_down=True, edge_respawn=True, respawn_penalty=-2.0):
    # start_down/edge_respawn/respawn_penalty kept for call-site compatibility (ignored for MountainCar)
    try:
        try:
            env = gym.make("MountainCarContinuous-v0", render_mode=render_mode)
        except TypeError:
            # Some gymnasium versions may not accept render_mode here
            env = gym.make("MountainCarContinuous-v0")
    except Exception:
        # Legacy/alternate naming fallback (older custom installs)
        try:
            env = gym.make("Continuous_MountainCar-v0", render_mode=render_mode)
        except TypeError:
            env = gym.make("Continuous_MountainCar-v0")
    env = MountainCarHoldRewardWrapper(env)
    env = CompatStateInfoWrapper(env)
    env = TimeLimit(env, max_episode_steps=max_episode_steps)
    env.reset(seed=seed)
    return env

# Sanity check
env = make_env(render_mode=None, seed=0)
obs, _ = env.reset(seed=0)
print("✅ Env ready (Continuous_MountainCar hold-at-goal; no edge_respawn)")
print("obs:", obs)
print("compat state:", obs_to_state(obs))
print("native obs dim:", int(np.asarray(obs).shape[0]), "| compat state dim:", len(obs_to_state(obs)))
print("action space:", env.action_space)
env.close()
