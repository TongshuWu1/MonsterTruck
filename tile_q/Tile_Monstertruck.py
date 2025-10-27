# ===============================================================
# TileQLearning_Monstertruck_YPR_standalone.py (Eval Every 50)
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
# Environment (same reward & success logic as YPR)
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

        self.R = dict(
            upright=20.0,
            spin=1,
            settle=0.5,
            energy=0.01,
            time=0.035,
            success=800.0,
        )

        self.prev_upright = 0.0
        self.prev_roll, self.prev_pitch = 0.0, 0.0

        self._viewer_ready = False
        if render:
            self._init_viewer()

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

    def reset(self):
        mujoco.mj_resetData(self.model, self.data)
        self.data.qpos[:3] = np.array([0, 0, 0.6]) + 0.02 * np.random.randn(3)
        self.data.qvel[:] = 0.0
        self.data.qpos[3:7] = np.array([0, 1, 0, 0])  # upside-down start
        mujoco.mj_forward(self.model, self.data)

        self.step_count = 0
        self.last_throttle = 0.0
        self.hold_counter = 0
        self.prev_upright = -1.0
        self.prev_roll, self.prev_pitch = 0.0, 0.0
        return self._get_obs()

    def _get_rp(self):
        q = np.copy(self.data.qpos[3:7])
        return quat_to_rp(q)

    def _get_obs(self):
        roll, pitch = self._get_rp()
        u = math.cos(roll) * math.cos(pitch)
        return np.array([u, roll, pitch, self.last_throttle], dtype=np.float32)

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

        roll, pitch = self._get_rp()
        u = math.cos(roll) * math.cos(pitch)
        du = u - self.prev_upright
        self.prev_upright = u

        droll = roll - self.prev_roll
        dpitch = pitch - self.prev_pitch
        self.prev_roll, self.prev_pitch = roll, pitch

        upright = (u + 1.0) * 0.5
        ang_motion = abs(droll) + abs(dpitch)
        R = self.R

        upright_reward = R["upright"] * max(du, 0.0)
        spin_reward = R["spin"] * abs(dpitch) * (1.0 - upright) if u > -0.8 else 0.0
        settle_penalty = -R["settle"] * ang_motion if (u > 0.9 and ang_motion > 0.02) else 0.0
        energy_penalty = -R["energy"] * (throttle ** 2)
        time_penalty = -R["time"]

        reward = upright_reward + spin_reward + settle_penalty + energy_penalty + time_penalty

        if u > 0.96 and ang_motion < 0.01:
            self.hold_counter += 1
            if self.hold_counter >= self.hold_needed:
                reward += R["success"]
                success, done = True, True
        else:
            self.hold_counter = 0

        self.step_count += 1
        if self.step_count >= self.max_steps:
            done = True

        reward = clip(reward, -10, 150)
        self._render()
        return self._get_obs(), reward, done, {"success": success}

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


def train_tileq(episodes=3000, render_every=100, max_steps=1500, load_model=False, eval_every=50):
    env = MonsterTruckFlipEnvYPR(render=False, realtime=False, frame_skip=10, max_steps=max_steps)

    lows  = [-1.0, -math.pi, -math.pi/2, -1.0]
    highs = [ 1.0,  math.pi,  math.pi/2,  1.0]

    agent = TileQAgent(lows, highs, total_episodes=episodes)

    rewards_list, success_flags = [], []
    eval_ep_indices, eval_rewards = [], []

    successes = 0

    for ep in range(1, episodes + 1):
        do_render = (render_every and ep % render_every == 0)
        env.render_enabled = do_render
        env.realtime = do_render

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
            recent_rewards = rewards_list[-10:]
            recent_success = sum(success_flags[-10:])
            print(f"Ep {ep:4d} | eps {agent.eps:5.3f} | total success {successes}")
            print(f"   Last 10 rewards: {[round(r, 2) for r in recent_rewards]}")
            print(f"   Success (last 10): {recent_success}/10")

        # === Evaluation every 50 episodes (5 runs each) ===
        if eval_every and (ep % eval_every == 0):
            eval_rs, eval_ss = [], []
            for _ in range(5):
                er, es = evaluate_episode(env, agent, max_steps)
                eval_rs.append(er)
                eval_ss.append(es)
            mean_r = np.mean(eval_rs)
            succ_rate = np.mean(eval_ss) * 100
            eval_ep_indices.append(ep)
            eval_rewards.append(mean_r)
            print(f"   [Eval @ Ep {ep}] avg_reward={mean_r:.1f} | success_rate={succ_rate:.1f}%")

    env.close()

    # === Improved Evaluation Plot ===
    if len(eval_ep_indices) > 0:
        plt.style.use('seaborn-v0_8-bright')
        window = 5
        smoothed = np.convolve(eval_rewards, np.ones(window)/window, mode='valid')

        plt.figure(figsize=(9, 5))
        plt.plot(eval_ep_indices, eval_rewards, 'o--', alpha=0.4, label='Raw eval reward')
        plt.plot(eval_ep_indices[window-1:], smoothed, 'r-', linewidth=2.2, label=f'{window}-pt Moving Avg')
        plt.xlabel("Episode", fontsize=12)
        plt.ylabel("Evaluation Reward", fontsize=12)
        plt.title("TileQ Evaluation — Smoothed Performance Over Training", fontsize=13, pad=10)
        plt.legend()
        plt.grid(True, alpha=0.3)
        plt.tight_layout()
        plt.savefig("tileq_eval_rewards_smooth.png", dpi=200)
        print("📈 Saved smooth evaluation plot to tileq_eval_rewards_smooth.png")

    return rewards_list, (eval_ep_indices, eval_rewards)


# ===============================================================
# Run training
# ===============================================================
if __name__ == "__main__":
    rewards, (eval_eps, eval_rewards) = train_tileq(
        episodes=2000, render_every=500, max_steps=1500,
        load_model=False, eval_every=50
    )
