# ===============================================================
# TileQLearning_Monstertruck_YPR.py
# Pitch-based flipping (roll & pitch uprightness, pitch-only motion)
# MountainCar-style tanh² distance penalty + adaptive velocity shaping
# ===============================================================

import os, math, time, json
import numpy as np
import matplotlib.pyplot as plt
import mujoco
from mujoco.glfw import glfw


# ===============================================================
# Quaternion → roll, pitch
# ===============================================================
def quat_to_rp(q):
    w, x, y, z = q
    t0 = +2.0 * (w * x + y * z)
    t1 = +1.0 - 2.0 * (x * x + y * y)
    roll = math.atan2(t0, t1)
    t2 = +2.0 * (w * y - z * x)
    t2 = max(-1.0, min(+1.0, t2))
    pitch = math.asin(t2)
    return roll, pitch


def clip(v, lo, hi):
    return lo if v < lo else (hi if v > hi else v)


# ===============================================================
# MonsterTruck Flip Environment (MountainCar-style shaping)
# ===============================================================
class MonsterTruckFlipEnvYPR:
    def __init__(self, xml_path="monstertruck.xml",
                 frame_skip=10, max_steps=2000,
                 render=False, realtime=False,
                 num_actions=9):
        if not os.path.exists(xml_path):
            raise FileNotFoundError(f"Cannot find {xml_path}")
        self.model = mujoco.MjModel.from_xml_path(xml_path)
        self.data = mujoco.MjData(self.model)
        self.frame_skip = frame_skip
        self.max_steps = max_steps
        self.render_enabled = render
        self.realtime = realtime
        self.dt = self.model.opt.timestep

        self.actions = np.linspace(-1.0, 1.0, num_actions).astype(np.float32)
        self.last_throttle = 0.0
        self.step_count = 0
        self.hold_counter = 0
        self.hold_needed = 4

        self.body_id = mujoco.mj_name2id(self.model, mujoco.mjtObj.mjOBJ_BODY, "chassis")

        # self.R = dict(
        #     position=1.0,    # distance penalty
        #     velocity=30.0,    # progress reward (du/dt)
        #     stop_boost=0.2,  # damping near upright (−ω²)
        #     energy=0.1,      # control effort penalty
        #     time=0.5,        # constant per-step time cost
        #     success=1000.0    # terminal bonus
        # )

        self.R = dict(
            position=0.8,    # distance penalty
            velocity=20.0,    # progress reward (du/dt)
            stop_boost=0.2,  # damping near upright (−ω²)
            energy=0.1,      # control effort penalty
            time=0.3,        # constant per-step time cost
            success=800.0    # terminal bonus
        )

        self.prev_upright = -1.0
        self.prev_roll, self.prev_pitch = 0.0, 0.0

        self._viewer_ready = False
        if render:
            self._init_viewer()

    # ---------------- Rendering ----------------
    def _init_viewer(self):
        if self._viewer_ready:
            return
        if not glfw.init():
            raise RuntimeError("GLFW init failed")
        self.window = glfw.create_window(1000, 800, "MonsterTruck Pitch Flip", None, None)
        glfw.make_context_current(self.window)
        glfw.swap_interval(1)
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
            self.close()
            self.render_enabled = False
            return
        self.cam.lookat[:] = self.data.xpos[self.body_id]
        mujoco.mjv_updateScene(self.model, self.data, self.opt, None, self.cam,
                               mujoco.mjtCatBit.mjCAT_ALL, self.scene)
        w, h = glfw.get_framebuffer_size(self.window)
        mujoco.mjr_render(mujoco.MjrRect(0, 0, w, h), self.scene, self.context)
        glfw.swap_buffers(self.window)
        glfw.poll_events()

    # ---------------- Core API ----------------
    def reset(self):
        mujoco.mj_resetData(self.model, self.data)
        self.data.qpos[:3] = np.array([0, 0, 0.2])
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

        # Apply control
        for _ in range(self.frame_skip):
            for i in range(min(2, self.model.nu)):
                self.data.ctrl[i] = throttle
            mujoco.mj_step(self.model, self.data)
            if self.realtime:
                time.sleep(self.dt)

        # Orientation and angular motion
        roll, pitch = self._get_rp()
        u = math.cos(roll) * math.cos(pitch)
        du = u - self.prev_upright
        self.prev_upright = u

        droll = roll - self.prev_roll
        dpitch = pitch - self.prev_pitch
        self.prev_roll, self.prev_pitch = roll, pitch

        horizon_dt = max(self.frame_skip * self.dt, 1e-6)
        pitch_rate = dpitch / horizon_dt

        # Normalized upright distance (0 = upright, 1 = upside-down)
        dist_norm = np.clip((1.0 - u) * 0.5, 0.0, 1.0)

        R = self.R

        # --------------------------------------------------------------
        # MountainCar-like shaping
        # --------------------------------------------------------------
        pos_penalty = R["position"] * (np.tanh(3.0 * dist_norm) ** 2)
        vel_progress = R["velocity"] * du / (horizon_dt + 1e-9)
        near_upright = np.exp(-12.0 * dist_norm)
        vel_brake = -R["stop_boost"] * near_upright * (pitch_rate ** 2)
        energy_penalty = R["energy"] * (throttle ** 2)
        time_penalty = R["time"]

        reward = vel_progress - pos_penalty - energy_penalty - time_penalty + vel_brake
        reward = clip(reward, -50.0, 300.0)

        # Success condition
        ang_motion = abs(droll) + abs(dpitch)
        if u > 0.96 and abs(pitch_rate) < 0.1 and ang_motion < 0.01:
            self.hold_counter += 1
            if self.hold_counter >= self.hold_needed:
                reward += R["success"]
                success, done = True, True
        else:
            self.hold_counter = 0

        self.step_count += 1
        if self.step_count >= self.max_steps:
            done = True

        self._render()
        return self._get_obs(), float(reward), done, {"success": success}

    def close(self):
        if self._viewer_ready:
            glfw.destroy_window(self.window)
            glfw.terminate()
            self._viewer_ready = False


# ===============================================================
# Tile Coder + Q-learning Agent
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
                 n_tiles=(18, 12, 6, 6), n_tilings=8,
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
        self._delta_accum = 0.0
        self._delta_count = 0

    def update_epsilon(self):
        frac = min(1.0, self.current_episode / self.total_episodes)
        decay_rate = 2.0
        self.eps = self.eps_end + (self.eps_start - self.eps_end) * np.exp(-decay_rate * frac)

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
        idxs = self.tc.encode(obs)
        q_sa = float(np.sum(self.w[a, idxs]))
        target = r + (0.0 if done else self.gamma * float(np.max(self.Qs(next_obs))))
        delta = target - q_sa
        for idx in idxs:
            self.w[a, idx] += self.alpha * delta
        self._delta_accum += abs(delta)
        self._delta_count += 1

    def avg_update(self):
        if self._delta_count == 0:
            return 0.0
        v = self._delta_accum / self._delta_count
        self._delta_accum = 0.0
        self._delta_count = 0
        return v

    def decay_eps(self):
        self.current_episode += 1
        self.update_epsilon()


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
            success_flag = 1 if info.get("success", False) else 0
            break
    return total, success_flag


def train_tileq(episodes=1000, max_steps=1500, eval_every=50, num_actions=9):
    env = MonsterTruckFlipEnvYPR(render=False, realtime=False,
                                 frame_skip=10, max_steps=max_steps, num_actions=num_actions)

    lows, highs = [-1.0, -math.pi, -math.pi/2, -1.0], [1.0, math.pi, math.pi/2, 1.0]
    agent = TileQAgent(lows, highs, n_tiles=(12, 10, 10, 6), n_tilings=8,
                       n_actions=len(env.actions), total_episodes=episodes)

    rewards_list, success_flags, eval_ep_indices, eval_rewards = [], [], [], []
    successes = 0

    for ep in range(1, episodes + 1):
        obs = env.reset()
        ep_ret, success_flag = 0.0, 0

        for _ in range(env.max_steps):
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
            avgupd = agent.avg_update()
            print(f"Ep {ep:4d} | eps {agent.eps:5.3f} | successes {successes} | ⟨|Δ|⟩ {avgupd:.4f}")
            print(f"   Last 10 rewards: {[round(r, 2) for r in last10]}  "
                  f"({sum(success_flags[-10:])}/10 success)")

        # Evaluation block
        if eval_every and (ep % eval_every == 0):
            eval_rs, eval_ss = [], []
            for i in range(3):
                env.render_enabled = (i == 0)
                env.realtime = (i == 0)
                er, es = evaluate_episode(env, agent, env.max_steps)
                eval_rs.append(er)
                eval_ss.append(es)
            env.render_enabled = False
            env.realtime = False

            mean_r, succ_rate = np.mean(eval_rs), np.mean(eval_ss) * 100
            eval_ep_indices.append(ep)
            eval_rewards.append(mean_r)
            print(f"   [Eval @ Ep {ep}] avg_reward={mean_r:.1f} | success_rate={succ_rate:.1f}%")

    env.close()

    if len(eval_ep_indices) > 0:
        plt.figure(figsize=(9, 5))
        plt.plot(eval_ep_indices, eval_rewards, 'o--', alpha=0.45, label='Raw eval reward')
        window = 5
        if len(eval_rewards) >= window:
            smoothed = np.convolve(eval_rewards, np.ones(window)/window, mode='valid')
            plt.plot(eval_ep_indices[window-1:], smoothed, 'r-', lw=2.2, label=f'{window}-pt Moving Avg')
        plt.xlabel("Episode")
        plt.ylabel("Evaluation Reward")
        plt.title("TileQ — Pitch-based flipping with velocity shaping")
        plt.legend()
        plt.grid(True, alpha=0.3)
        plt.tight_layout()
        plt.savefig("tileq_pitch_eval.png", dpi=200)
        plt.close()
        print("📈 Saved eval plot to tileq_pitch_eval.png")

    return rewards_list, (eval_ep_indices, eval_rewards)


if __name__ == "__main__":
    train_tileq(episodes=1000, max_steps=1500, eval_every=50, num_actions=9)
