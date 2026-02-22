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
    obs from Continuous_MountainCar-v0 is [position, velocity].
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
    """Compatibility renderer for experiments.py random collection visualization.
    Here x is mountain-car position in roughly [-1.2, 0.6].
    """
    try:
        from PIL import Image, ImageDraw
    except Exception:
        return None

    pos = float(x)
    img = Image.new("RGB", (W, H), (245, 245, 245))
    dr = ImageDraw.Draw(img)

    # Terrain profile (stylized sine hill)
    left, right = -1.2, 0.6
    margin = 30
    xs = np.linspace(left, right, 250)
    pts = []
    for xx in xs:
        yy = np.sin(3 * xx) * 0.45 + 0.55
        px = int(margin + (xx - left) / (right - left) * (W - 2 * margin))
        py = int(0.22 * H + yy * 0.55 * H)
        pts.append((px, py))
    dr.line(pts, fill=(150, 150, 150), width=4)

    # Goal flag near right hill (Continuous MountainCar goal ~0.45)
    goal_x = 0.45
    gpx = int(margin + (goal_x - left) / (right - left) * (W - 2 * margin))
    gpy = int(0.22 * H + (np.sin(3 * goal_x) * 0.45 + 0.55) * 0.55 * H)
    dr.line([(gpx, gpy - 50), (gpx, gpy)], fill=(80, 80, 80), width=3)
    dr.polygon([(gpx, gpy - 50), (gpx + 24, gpy - 42), (gpx, gpy - 34)], fill=(220, 40, 40))

    # Car position projected onto terrain
    car_y = np.sin(3 * pos) * 0.45 + 0.55
    cx = int(margin + (pos - left) / (right - left) * (W - 2 * margin))
    cy = int(0.22 * H + car_y * 0.55 * H)
    dr.rounded_rectangle([cx - 18, cy - 12, cx + 18, cy + 12], radius=5, fill=(60, 90, 160), outline=(20, 20, 20))
    dr.ellipse([cx - 14, cy + 8, cx - 4, cy + 18], fill=(30, 30, 30))
    dr.ellipse([cx + 4, cy + 8, cx + 14, cy + 18], fill=(30, 30, 30))

    return np.asarray(img, dtype=np.uint8)

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
        env = gym.make("Continuous_MountainCar-v0", render_mode=render_mode)
    except TypeError:
        # Some gymnasium versions may not accept render_mode here
        env = gym.make("Continuous_MountainCar-v0")
    env = MountainCarHoldRewardWrapper(env)
    env = CompatStateInfoWrapper(env)
    env = TimeLimit(env, max_episode_steps=max_episode_steps)
    env.reset(seed=seed)
    return env

# Sanity check
env = make_env(render_mode=None, seed=0)
obs, _ = env.reset(seed=0)
print("✅ Env ready (mountaincar_hold; no edge_respawn)")
print("obs:", obs)
print("compat state:", obs_to_state(obs))
print("action space:", env.action_space)
env.close()
