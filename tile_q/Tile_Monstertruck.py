# ===============================================================
# TileQLearning_Monstertruck_YPR_standalone.py
# Instantaneous Angular Velocity Reward (No "Best Angle" Tracking)
# ===============================================================

import os, math, json, time
import numpy as np
import matplotlib.pyplot as plt
import mujoco
from mujoco.glfw import glfw


# ------------------------- small utils -------------------------
def clip(v, lo, hi):
    return lo if v < lo else (hi if v > hi else v)

def quat_to_rp(q):
    """Quaternion -> (roll, pitch)."""
    w, x, y, z = q
    t0 = +2.0 * (w * x + y * z)
    t1 = +1.0 - 2.0 * (x * x + y * y)
    roll = math.atan2(t0, t1)

    t2 = +2.0 * (w * y - z * x)
    t2 = +1.0 if t2 > +1.0 else (-1.0 if t2 < -1.0 else t2)
    pitch = math.asin(t2)
    return roll, pitch


# ===============================================================
# Environment — instantaneous angular velocity shaping
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
        self.hold_counter = 0
        self.hold_needed = 4

        self.body_id = mujoco.mj_name2id(self.model, mujoco.mjtObj.mjOBJ_BODY, "chassis")

        # Reward weights
        self.R = dict(
            angvel=5.0,     # weight for angular velocity shaping
            energy=0.02,    # energy cost
            time=0.05,      # time penalty
            success=800.0,  # success bonus
        )

        self._viewer_ready = False
        if render:
            self._init_viewer()

    # -----------------------------------------------------------
    def _init_viewer(self):
        if self._viewer_ready:
            return
        if not glfw.init():
            raise RuntimeError("GLFW init failed")
        self.window = glfw.create_window(1000, 800, "MonsterTruck YPR Flip", None, None)
        glfw.make_context_current(self.window)
        self.cam = mujoco.MjvCamera()
        self.opt = mujoco.MjvOption()
        self.scene = mujoco.MjvScene(self.model, maxgeom=20000)
        self.context = mujoco.MjrContext(self.model, mujoco.mjtFontScale.mjFONTSCALE_150)
        self.cam.distance, self.cam.azimuth, self.cam.elevation = 3.0, 90, -25
        self._viewer_ready = True

    # -----------------------------------------------------------
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

    # -----------------------------------------------------------
    def reset(self):
        mujoco.mj_resetData(self.model, self.data)
        self.data.qpos[:3] = np.array([0, 0, 0.6]) + 0.02 * np.random.randn(3)
        self.data.qvel[:] = 0.0
        self.data.qpos[3:7] = np.array([0, 1, 0, 0])  # upside-down start
        mujoco.mj_forward(self.model, self.data)

        self.step_count = 0
        self.last_throttle = 0.0
        self.hold_counter = 0
        return self._get_obs()

    # -----------------------------------------------------------
    def _get_rp(self):
        q = np.copy(self.data.qpos[3:7])
        return quat_to_rp(q)

    def _get_obs(self):
        roll, pitch = self._get_rp()
        u = math.cos(roll) * math.cos(pitch)
        return np.array([u, roll, pitch, self.last_throttle], dtype=np.float32)

    # -----------------------------------------------------------
    def step(self, a_idx):
        throttle = float(self.actions[a_idx])
        self.last_throttle = throttle
        done, success = False, False

        for _ in range(self.frame_skip):
            if self.model.nu >= 2:
                self.data.ctrl[0] = throttle
                self.data.ctrl[1] = throttle
            mujoco.mj_step(self.model, self.data)
            if self.realtime:
                time.sleep(self.dt)

        # --- Roll + Angular Velocity ---
        roll, pitch = self._get_rp()
        roll_deg = np.degrees((roll + 2*np.pi) % (2*np.pi))
        roll_rate = float(self.data.xvelr[self.body_id][0])  # roll angular velocity (rad/s)

        # --- Angle-scaled reward (no history / progress) ---
        scale = (180 - roll_deg) / 180.0  # 1 at 180°, 0 at 0°, negative beyond
        scale = clip(scale, -1.0, 1.0)
        R_ang = self.R["angvel"] * (-roll_rate) * scale

        # --- Combine with simple penalties ---
        reward = R_ang - self.R["energy"] * (throttle ** 2) - self.R["time"]

        # --- Success condition (upright within ±10°) ---
        if roll_deg < 10 or roll_deg > 350:
            self.hold_counter += 1
            if self.hold_counter >= self.hold_needed:
                reward += self.R["success"]
                success, done = True, True
        else:
            self.hold_counter = 0

        self.step_count += 1
        if self.step_count >= self.max_steps:
            done = True

        reward = clip(reward, -50, 150)
        self._render()
        return self._get_obs(), reward, done, {"success": success}

    # -----------------------------------------------------------
    def close(self):
        if self._viewer_ready:
            glfw.destroy_window(self.window)
            glfw.terminate()
            self._viewer_ready = False


# ===============================================================
# Tile Coder (CMAC)
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
        indices = []
        for t in range(self.n_tilings):
            shifted = (ratios + self.offsets[t]) * self.n_tiles
            tile_coords = np.floor(shifted).astype(int)
            flat_idx = np.ravel_multi_index(tile_coords, self.n_tiles, mode='clip')
            indices.append(t * int(np.prod(self.n_tiles)) + flat_idx)
        return indices

    @property
    def total_tiles(self):
        return self.n_tilings * int(np.prod(self.n_tiles))


# ===============================================================
# Tile-coded Q-learning Agent
# ===============================================================
class TileQAgent:
    def __init__(self, obs_low, obs_high,
                 n_tiles=(6, 6, 12, 6), n_tilings=8,
                 n_actions=9, alpha=0.05, gamma=0.98,
                 eps_start=0.99, eps_end=0.01, total_episodes=5000):

        self.gamma = gamma
        self.alpha = alpha / float(n_tilings)
        self.eps_start, self.eps_end = eps_start, eps_end
        self.total_episodes = total_episodes
        self.eps = eps_start
        self.eps_decay = (eps_end / eps_start) ** (1.0 / total_episodes)
        self.n_actions = n_actions

        self.tc = TileCoder(obs_low, obs_high, n_tiles, n_tilings)
        self.n_features = self.tc.total_tiles
        self.w = np.zeros((n_actions, self.n_features), dtype=np.float32)

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
        self.eps = max(self.eps_end, self.eps * self.eps_decay)


# ===============================================================
# Evaluation + Training
# ===============================================================
def evaluate_episode(env, agent, max_steps=1500):
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


def train_tileq(episodes=3000, render_every=100, max_steps=1500, eval_every=50):
    env = MonsterTruckFlipEnvYPR(render=False, realtime=False, frame_skip=10, max_steps=max_steps)
    lows  = [-1.0, -math.pi, -math.pi/2, -1.0]
    highs = [ 1.0,  math.pi,  math.pi/2,  1.0]
    agent = TileQAgent(lows, highs, total_episodes=episodes)

    rewards_list, success_flags, eval_ep_indices, eval_rewards = [], [], [], []
    successes = 0

    for ep in range(1, episodes + 1):
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

        rewards_list.append(ep_ret)
        success_flags.append(success_flag)
        agent.decay_eps()

        if ep % 10 == 0:
            last10 = rewards_list[-10:]
            print(f"Ep {ep:4d} | eps {agent.eps:5.3f} | success {successes}")
            print(f"   Last 10 rewards: {[round(r, 2) for r in last10]}  ({sum(success_flags[-10:])}/10 success)")

        if eval_every and (ep % eval_every == 0):
            eval_rs, eval_ss = [], []
            for _ in range(5):
                er, es = evaluate_episode(env, agent, max_steps)
                eval_rs.append(er)
                eval_ss.append(es)
            mean_r, succ_rate = np.mean(eval_rs), np.mean(eval_ss) * 100
            eval_ep_indices.append(ep)
            eval_rewards.append(mean_r)
            print(f"   [Eval @ Ep {ep}] avg_reward={mean_r:.1f} | success_rate={succ_rate:.1f}%")

    env.close()

    if len(eval_ep_indices) > 0:
        plt.style.use('seaborn-v0_8-bright')
        window = 5
        smoothed = np.convolve(eval_rewards, np.ones(window)/window, mode='valid')
        plt.figure(figsize=(9, 5))
        plt.plot(eval_ep_indices, eval_rewards, 'o--', alpha=0.4, label='Raw eval reward')
        plt.plot(eval_ep_indices[window-1:], smoothed, 'r-', linewidth=2.2, label=f'{window}-pt Moving Avg')
        plt.xlabel("Episode"); plt.ylabel("Evaluation Reward")
        plt.title("TileQ — Instantaneous Angular Velocity Shaping")
        plt.legend(); plt.grid(True, alpha=0.3)
        plt.tight_layout(); plt.savefig("tileq_eval_rewards_angvel_instant.png", dpi=200)
        print("📈 Saved eval plot to tileq_eval_rewards_angvel_instant.png")

    return rewards_list, (eval_ep_indices, eval_rewards)


# ===============================================================
# Run training
# ===============================================================
if __name__ == "__main__":
    rewards, (eval_eps, eval_rewards) = train_tileq(
        episodes=2000, render_every=500, max_steps=1500, eval_every=50
    )
