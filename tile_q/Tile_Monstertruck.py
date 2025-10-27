# ===============================================================
# TileQLearning_Monstertruck.py
# ===============================================================
# Tile-coded Q-learning for the MonsterTruck flip environment
# ---------------------------------------------------------------
# State: [roll, roll_rate, last_throttle]
# Action: discrete throttle (-1.0 to 1.0)
# Reward:
#   + Linear angular velocity reward (max upside-down, decays near upright)
#   - Linear angle penalty (further from upright = more negative)
#   - Jerk and energy/time penalties
# Success:
#   - When upright (<3° or >357°) and roll_rate ≈ 0
# ===============================================================

import os, math, time
import numpy as np
import matplotlib.pyplot as plt
import mujoco
from mujoco.glfw import glfw


# ===============================================================
# Utility: Quaternion → roll (radians)
# ===============================================================
def quat_to_rp(q):
    """Convert quaternion to (roll, pitch)"""
    w, x, y, z = q
    t0 = +2.0 * (w * x + y * z)
    t1 = +1.0 - 2.0 * (x * x + y * y)
    roll = math.atan2(t0, t1)
    t2 = +2.0 * (w * y - z * x)
    t2 = +1.0 if t2 > +1.0 else (-1.0 if t2 < -1.0 else t2)
    pitch = math.asin(t2)
    return roll, pitch


# ===============================================================
# MonsterTruck Flip Environment
# ===============================================================
class MonsterTruckFlipEnvYPR:
    def __init__(self, xml_path="monstertruck.xml",
                 frame_skip=10, max_steps=2000,
                 render=False, realtime=False):
        if not os.path.exists(xml_path):
            raise FileNotFoundError(f"Cannot find {xml_path}")
        self.model = mujoco.MjModel.from_xml_path(xml_path)
        self.data = mujoco.MjData(self.model)
        self.frame_skip = frame_skip
        self.max_steps = max_steps
        self.render_enabled = render
        self.realtime = realtime
        self.dt = self.model.opt.timestep

        self.actions = np.linspace(-1.0, 1.0, 9)
        self.last_throttle = 0.0
        self.step_count = 0
        self.body_id = mujoco.mj_name2id(self.model, mujoco.mjtObj.mjOBJ_BODY, "chassis")

        # Reward weights
        self.R = dict(
            angvel=8.0,
            angle=1.0,
            energy=0.2,
            jerk=0.2,
            time=0.4,
            success=800.0,
        )

        self._viewer_ready = False
        if render:
            self._init_viewer()

    # ---------------------- Rendering ----------------------
    def _init_viewer(self):
        if self._viewer_ready:
            return
        if not glfw.init():
            raise RuntimeError("GLFW init failed")
        self.window = glfw.create_window(1000, 800, "MonsterTruck Flip", None, None)
        glfw.make_context_current(self.window)
        self.cam = mujoco.MjvCamera()
        self.opt = mujoco.MjvOption()
        self.scene = mujoco.MjvScene(self.model, maxgeom=20000)
        self.context = mujoco.MjrContext(self.model, mujoco.mjtFontScale.mjFONTSCALE_150)
        self.cam.distance, self.cam.azimuth, self.cam.elevation = 3.0, 90, -25
        self._viewer_ready = True

    def _render(self):
        if not self.render_enabled:
            return
        if not self._viewer_ready:
            self._init_viewer()
        if glfw.window_should_close(self.window):
            glfw.destroy_window(self.window)
            glfw.terminate()
            self._viewer_ready = False
            self.render_enabled = False
            return
        self.cam.lookat[:] = self.data.xpos[self.body_id]
        mujoco.mjv_updateScene(self.model, self.data, self.opt, None, self.cam,
                               mujoco.mjtCatBit.mjCAT_ALL, self.scene)
        w, h = glfw.get_framebuffer_size(self.window)
        mujoco.mjr_render(mujoco.MjrRect(0, 0, w, h), self.scene, self.context)
        glfw.swap_buffers(self.window)
        glfw.poll_events()

    # ---------------------- Core API ----------------------
    def reset(self):
        mujoco.mj_resetData(self.model, self.data)
        self.data.qpos[:3] = np.array([0, 0, 0.6]) + 0.02 * np.random.randn(3)
        self.data.qvel[:] = 0.0
        self.data.qpos[3:7] = np.array([0, 1, 0, 0])  # upside-down
        mujoco.mj_forward(self.model, self.data)
        self.step_count = 0
        self.last_throttle = 0.0
        return self._get_obs()

    def _get_rp(self):
        q = np.copy(self.data.qpos[3:7])
        return quat_to_rp(q)

    def _get_obs(self):
        roll, _ = self._get_rp()
        roll_rate = float(self.data.cvel[self.body_id][3])
        return np.array([roll, roll_rate, self.last_throttle], dtype=np.float32)

    def step(self, a_idx):
        throttle = float(self.actions[a_idx])
        done, success = False, False

        for _ in range(self.frame_skip):
            if self.model.nu >= 2:
                self.data.ctrl[0] = throttle
                self.data.ctrl[1] = throttle
            mujoco.mj_step(self.model, self.data)
            if self.realtime:
                time.sleep(self.dt)

        roll, _ = self._get_rp()
        roll_deg = np.degrees((roll + 2 * np.pi) % (2 * np.pi))
        roll_rate = float(self.data.cvel[self.body_id][3])

        # --- Linear angular velocity reward ---
        scale = 1 - abs(180 - roll_deg) / 180.0
        R_angvel = self.R["angvel"] * abs(roll_rate) * scale

        # --- Linear angle penalty ---
        R_angle = -self.R["angle"] * (roll_deg / 360.0)

        # --- Jerk penalty ---
        jerk = throttle - self.last_throttle
        R_jerk = -self.R["jerk"] * (jerk ** 2)

        # --- Total reward ---
        reward = R_angvel + R_angle + R_jerk - self.R["energy"] * (throttle ** 2) - self.R["time"]

        # --- Success detection ---
        if (roll_deg < 3 or roll_deg > 357) and abs(roll_rate) < 0.05:
            reward += self.R["success"]
            success, done = True, True

        self.last_throttle = throttle
        self.step_count += 1
        if self.step_count >= self.max_steps:
            done = True

        self._render()
        return self._get_obs(), reward, done, {"success": success}

    def close(self):
        if self._viewer_ready:
            glfw.destroy_window(self.window)
            glfw.terminate()
            self._viewer_ready = False


# ===============================================================
# Tile Coder + Agent (Exponential ε-decay)
# ===============================================================
class TileCoder:
    def __init__(self, lows, highs, n_tiles, n_tilings, seed=0):
        self.lows = np.array(lows, dtype=np.float32)
        self.highs = np.array(highs, dtype=np.float32)
        self.n_tiles = np.array(n_tiles, dtype=np.int32)
        self.n_tilings = int(n_tilings)
        self.dim = len(lows)
        rng = np.random.default_rng(seed)
        self.offsets = rng.uniform(0, 1, (self.n_tilings, self.dim)) / self.n_tiles

    def encode(self, s):
        s = np.array(s, dtype=np.float32)
        ratios = (s - self.lows) / (self.highs - self.lows + 1e-8)
        ratios = np.clip(ratios, 0, 0.999999)
        idxs = []
        for t in range(self.n_tilings):
            shifted = (ratios + self.offsets[t]) * self.n_tiles
            tile_coords = np.floor(shifted).astype(int)
            flat = np.ravel_multi_index(tile_coords, self.n_tiles, mode='clip')
            idxs.append(t * int(np.prod(self.n_tiles)) + flat)
        return idxs

    @property
    def total_tiles(self):
        return self.n_tilings * int(np.prod(self.n_tiles))


class TileQAgent:
    def __init__(self, obs_low, obs_high,
                 n_tiles=(18, 12, 6), n_tilings=8,
                 n_actions=9, alpha=0.05, gamma=0.98,
                 eps_start=0.99, eps_end=0.01, total_episodes=5000):
        self.gamma = gamma
        self.alpha = alpha / float(n_tilings)
        self.eps_start, self.eps_end = eps_start, eps_end
        self.total_episodes = total_episodes
        self.n_actions = n_actions
        self.tc = TileCoder(obs_low, obs_high, n_tiles, n_tilings)
        self.n_features = self.tc.total_tiles
        self.w = np.zeros((n_actions, self.n_features), dtype=np.float32)
        self.current_episode = 0
        self.eps = eps_start

    def update_epsilon(self):
        frac = min(1.0, self.current_episode / self.total_episodes)
        decay_rate = 2.0
        self.eps = self.eps_end + (self.eps_start - self.eps_end) * np.exp(-decay_rate * frac)

    def _qvalue(self, obs, a):
        idxs = self.tc.encode(obs)
        return float(np.sum(self.w[a, idxs]))

    def Qs(self, obs):
        idxs = self.tc.encode(obs)
        return np.array([np.sum(self.w[a, idxs]) for a in range(self.n_actions)], dtype=np.float32)

    def act(self, obs):
        if np.random.rand() < self.eps:
            return np.random.randint(self.n_actions)
        return int(np.argmax(self.Qs(obs)))

    def act_greedy(self, obs):
        return int(np.argmax(self.Qs(obs)))

    def learn(self, obs, a, r, next_obs, done):
        q_sa = self._qvalue(obs, a)
        target = r + (0.0 if done else self.gamma * float(np.max(self.Qs(next_obs))))
        delta = target - q_sa
        for idx in self.tc.encode(obs):
            self.w[a, idx] += self.alpha * delta

    def decay_eps(self):
        self.current_episode += 1
        self.update_epsilon()


# ===============================================================
# Evaluation + Training
# ===============================================================
def evaluate_episode(env, agent, max_steps=1500):
    # Force no rendering during evaluation
    env.render_enabled = False
    env.realtime = False

    obs = env.reset()
    total = 0.0
    success_flag = 0
    for _ in range(max_steps):
        a = agent.act_greedy(obs)
        obs, r, done, info = env.step(a)
        total += r
        if done:
            if info.get("success", False):
                success_flag = 1
            break
    return total, success_flag


def train_tileq(episodes=2000, render_every=100, max_steps=1500, eval_every=50):
    env = MonsterTruckFlipEnvYPR(render=False, realtime=False, frame_skip=5, max_steps=max_steps)
    lows, highs = [-math.pi, -20.0, -1.0], [math.pi, 20.0, 1.0]
    agent = TileQAgent(lows, highs, n_tiles=(18, 12, 6), n_tilings=8,
                       n_actions=len(env.actions), total_episodes=episodes)

    rewards_list, success_flags, eval_ep_indices, eval_rewards = [], [], [], []
    successes = 0

    for ep in range(1, episodes + 1):
        # Render only one episode after render_every
        render = (render_every and ep % render_every == 0)
        env.render_enabled = render
        env.realtime = render

        obs = env.reset()
        ep_ret, success_flag = 0.0, 0

        for _ in range(max_steps):
            a = agent.act(obs)
            next_obs, r, done, info = env.step(a)
            agent.learn(obs, a, r, next_obs, done)
            obs = next_obs
            ep_ret += r
            if done:
                if info.get("success", False):
                    success_flag = 1
                    successes += 1
                break

        # Turn rendering off after one episode
        if render:
            env.render_enabled = False
            env.realtime = False

        rewards_list.append(ep_ret)
        success_flags.append(success_flag)
        agent.decay_eps()

        if ep % 10 == 0:
            last10 = rewards_list[-10:]
            print(f"Ep {ep:4d} | eps {agent.eps:5.3f} | successes {successes}")
            print(f"   Last 10 rewards: {[round(r, 2) for r in last10]}  ({sum(success_flags[-10:])}/10 success)")

        if eval_every and (ep % eval_every == 0):
            eval_rs, eval_ss = [], []
            for _ in range(5):
                er, es = evaluate_episode(env, agent, max_steps)
                eval_rs.append(er)
                eval_ss.append(es)
            mean_r, succ_rate = np.mean(eval_rs), np.mean(eval_ss)*100
            eval_ep_indices.append(ep)
            eval_rewards.append(mean_r)
            print(f"   [Eval @ Ep {ep}] avg_reward={mean_r:.1f} | success_rate={succ_rate:.1f}%")

    env.close()

    if len(eval_ep_indices) > 0:
        plt.style.use('seaborn-v0_8-bright')
        window = 5
        smoothed = np.convolve(eval_rewards, np.ones(window)/window, mode='valid')
        plt.figure(figsize=(9,5))
        plt.plot(eval_ep_indices, eval_rewards, 'o--', alpha=0.45, label='Raw eval reward')
        if len(smoothed) > 0:
            plt.plot(eval_ep_indices[window-1:], smoothed, 'r-', lw=2.2, label=f'{window}-pt Moving Avg')
        plt.xlabel("Episode")
        plt.ylabel("Evaluation Reward")
        plt.title("TileQ — Exponential ε-decay")
        plt.legend()
        plt.grid(True, alpha=0.3)
        plt.tight_layout()
        plt.savefig("tileq_eval_rewards_exponential_eps.png", dpi=200)
        print("📈 Saved eval plot to tileq_eval_rewards_exponential_eps.png")

    return rewards_list, (eval_ep_indices, eval_rewards)


if __name__ == "__main__":
    train_tileq(episodes=2000, render_every=500, max_steps=1500, eval_every=50)
