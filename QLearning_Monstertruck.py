# QLearning_Monstertruck.py
# Balanced reward shaping for monster-truck flipping
# Adds option to load existing Q-table and continue training
# Requires: pip install mujoco glfw numpy

import os, json, math, time
import numpy as np
import mujoco
from mujoco.glfw import glfw


# ============================================================
# Helper functions
# ============================================================
def clip(v, lo, hi):
    return lo if v < lo else (hi if v > hi else v)


# ============================================================
# Environment
# ============================================================
class MonsterTruckFlipEnv:
    """
    Discrete throttle env for flipping a truck upright.

    Observations:
        up_z         : cosine of uprightness
        ang_speed    : angular speed magnitude
        wheel_speed  : mean of rear wheel speeds
        flip_rate    : roll rate (rad/s)
        throttle     : last applied throttle

    Reward shaping:
        + Continuous upright reward (smooth shaping)
        + Progress reward (encourages improvement)
        + Spin encouragement while upside-down
        + Big terminal bonus when upright and stable
        - Energy and time penalties
    """

    def __init__(self, xml_path="monstertruck.xml",
                 frame_skip=10, max_steps=2000,
                 render=False, realtime=True):

        if not os.path.exists(xml_path):
            raise FileNotFoundError(f"Cannot find {xml_path}")
        self.model = mujoco.MjModel.from_xml_path(xml_path)
        self.data = mujoco.MjData(self.model)

        self.frame_skip = frame_skip
        self.max_steps = max_steps
        self.render_enabled = render
        self.realtime = realtime
        self.dt = self.model.opt.timestep

        # Actions
        self.actions = np.array([-1.0, -0.5, 0.0, 0.5, 1.0], dtype=np.float32)

        # Stability thresholds
        self.upright_threshold = 0.93
        self.stable_rate_tol = 1.5
        self.hold_needed = 4

        # Runtime state
        self.hold_counter = 0
        self.step_count = 0
        self.last_throttle = 0.0
        self.prev_upright = 0.0

        # Mujoco object references
        self.body_id = mujoco.mj_name2id(self.model, mujoco.mjtObj.mjOBJ_BODY, "chassis")
        self.j_rl = mujoco.mj_name2id(self.model, mujoco.mjtObj.mjOBJ_JOINT, "j_rl")
        self.j_rr = mujoco.mj_name2id(self.model, mujoco.mjtObj.mjOBJ_JOINT, "j_rr")
        self.dof_rl = self.model.jnt_dofadr[self.j_rl] if self.j_rl != -1 else None
        self.dof_rr = self.model.jnt_dofadr[self.j_rr] if self.j_rr != -1 else None

        # Rendering
        self.window = None
        if self.render_enabled:
            self._init_viewer()

        # Reward weights (easy tuning)
        self.R = {
            "upright": 10.0,
            "progress": 20.0,
            "spin": 0.3,
            "energy": -0.05,
            "time": -0.001,
            "success": 100.0
        }

    # ---------------------- Rendering ----------------------
    def _init_viewer(self):
        if not glfw.init():
            raise RuntimeError("GLFW init failed")
        self.window = glfw.create_window(1000, 800, "MonsterTruck Q-Learning", None, None)
        if not self.window:
            glfw.terminate()
            raise RuntimeError("GLFW window creation failed")
        glfw.make_context_current(self.window)
        glfw.swap_interval(1)
        self.cam = mujoco.MjvCamera()
        self.opt = mujoco.MjvOption()
        self.scene = mujoco.MjvScene(self.model, maxgeom=20000)
        self.context = mujoco.MjrContext(self.model, mujoco.mjtFontScale.mjFONTSCALE_150)
        self.cam.distance, self.cam.azimuth, self.cam.elevation = 3.0, 90, -25

    def _render(self):
        if not self.render_enabled or self.window is None:
            return
        if glfw.window_should_close(self.window):
            self.render_enabled = False
            glfw.destroy_window(self.window)
            glfw.terminate()
            return
        self.cam.lookat[:] = self.data.xpos[self.body_id]
        mujoco.mjv_updateScene(self.model, self.data, self.opt, None, self.cam,
                               mujoco.mjtCatBit.mjCAT_ALL, self.scene)
        w, h = glfw.get_framebuffer_size(self.window)
        if w > 0 and h > 0:
            mujoco.mjr_render(mujoco.MjrRect(0, 0, w, h), self.scene, self.context)
        glfw.swap_buffers(self.window)
        glfw.poll_events()

    # ---------------------- Core API ----------------------
    def reset(self):
        mujoco.mj_resetData(self.model, self.data)
        self.data.qpos[:3] = np.array([0, 0, 0.6]) + 0.02 * np.random.randn(3)
        self.data.qvel[:] = 0.0
        self.data.qpos[3:7] = np.array([0, 1, 0, 0])  # upside down
        self.hold_counter = 0
        self.step_count = 0
        self.prev_upright = 0.0
        mujoco.mj_forward(self.model, self.data)
        return self._get_obs()

    def _get_obs(self):
        R = self.data.xmat[self.body_id].reshape(3, 3)
        up_z = float(R[2, 2])
        ang_speed = float(np.linalg.norm(self.data.cvel[self.body_id][3:]))
        flip_rate = float((R.T @ self.data.cvel[self.body_id][3:])[0])
        if self.dof_rl is not None and self.dof_rr is not None:
            v_rl = self.data.qvel[self.dof_rl]
            v_rr = self.data.qvel[self.dof_rr]
            wheel_speed = 0.5 * (v_rl + v_rr)
        else:
            wheel_speed = 0.0
        return np.array([up_z, ang_speed, wheel_speed, flip_rate, self.last_throttle], dtype=np.float32)

    def step(self, a_idx: int):
        throttle = float(self.actions[a_idx])
        self.last_throttle = throttle
        done, success = False, False

        for _ in range(self.frame_skip):
            if self.model.nu >= 2:
                self.data.ctrl[0] = throttle
                self.data.ctrl[1] = throttle
            mujoco.mj_step(self.model, self.data)

        R = self.data.xmat[self.body_id].reshape(3, 3)
        up_z = float(R[2, 2])
        ang_vel_world = self.data.cvel[self.body_id][3:]
        ang_vel_body = R.T @ ang_vel_world
        flip_rate = float(ang_vel_body[0])

        # Reward
        upright = (up_z + 1) / 2.0
        delta_upright = upright - self.prev_upright
        self.prev_upright = upright

        upright_reward = self.R["upright"] * (upright ** 2)
        progress_reward = self.R["progress"] * max(delta_upright, 0)
        spin_reward = self.R["spin"] * abs(flip_rate) / 10.0 if up_z < 0 else 0.0
        energy_penalty = self.R["energy"] * abs(throttle)
        time_penalty = self.R["time"]

        success_bonus = 0.0
        if up_z > self.upright_threshold and abs(flip_rate) < self.stable_rate_tol:
            self.hold_counter += 1
            if self.hold_counter >= self.hold_needed:
                success_bonus = self.R["success"]
                success, done = True, True
        else:
            self.hold_counter = 0

        reward = (upright_reward + progress_reward + spin_reward +
                  success_bonus + energy_penalty + time_penalty)
        reward = clip(reward, -10, 200)

        self.step_count += 1
        if self.step_count >= self.max_steps:
            done = True

        self._render()
        return self._get_obs(), reward, done, {"success": success}

    def close(self):
        if self.render_enabled and self.window is not None:
            glfw.destroy_window(self.window)
            glfw.terminate()


# ============================================================
# Q-Learning Agent
# ============================================================
class QAgent:
    def __init__(self,
                 obs_bins=(15, 10, 10, 10, 5),
                 obs_ranges=((-1.0, 1.0), (0, 50), (-120, 120), (-20, 20), (-1, 1)),
                 n_actions=5,
                 alpha=0.3, gamma=0.98,
                 eps_start=0.99, eps_end=0.05, total_episodes=5000):

        self.obs_bins = obs_bins
        self.obs_ranges = obs_ranges
        self.n_actions = n_actions
        self.alpha, self.gamma = alpha, gamma
        self.eps_start, self.eps_end = eps_start, eps_end
        self.total_episodes = total_episodes
        self.eps = eps_start
        self.eps_decay = (eps_end / eps_start) ** (1.0 / total_episodes)

        self.q = np.zeros(obs_bins + (n_actions,), dtype=np.float32)
        self.bin_edges = [np.linspace(lo, hi, nb - 1)
                          for (lo, hi), nb in zip(obs_ranges, obs_bins)]

    def _disc(self, obs):
        idxs = [int(np.digitize(o, edges)) for o, edges in zip(obs, self.bin_edges)]
        return tuple(idxs)

    def act(self, obs):
        s = self._disc(obs)
        if np.random.rand() < self.eps:
            return np.random.randint(self.n_actions)
        return int(np.argmax(self.q[s]))

    def learn(self, obs, action, reward, next_obs, done):
        s, sn = self._disc(obs), self._disc(next_obs)
        best_next = 0.0 if done else np.max(self.q[sn])
        target = reward + self.gamma * best_next
        self.q[s + (action,)] += self.alpha * (target - self.q[s + (action,)])

    def decay_eps(self):
        self.eps = max(self.eps_end, self.eps * self.eps_decay)

    def save(self, path="qtable_monstertruck.npy", meta="qtable_meta.json"):
        np.save(path, self.q)
        with open(meta, "w") as f:
            json.dump({"obs_bins": self.obs_bins, "obs_ranges": self.obs_ranges,
                       "n_actions": self.n_actions}, f, indent=2)
        print(f"💾 Saved Q-table to {path}")

    def load(self, path="qtable_monstertruck.npy", meta="qtable_meta.json"):
        """Loads saved Q-table and metadata."""
        if not (os.path.exists(path) and os.path.exists(meta)):
            print("⚠️ No saved Q-table found, starting fresh.")
            return False
        try:
            self.q = np.load(path)
            with open(meta, "r") as f:
                meta_data = json.load(f)
            self.obs_bins = tuple(meta_data["obs_bins"])
            self.obs_ranges = [tuple(r) for r in meta_data["obs_ranges"]]
            self.n_actions = meta_data["n_actions"]
            print(f"✅ Loaded Q-table from {path}")
            print(f"Shape: {self.q.shape}")
            return True
        except Exception as e:
            print(f"⚠️ Failed to load Q-table: {e}")
            return False


# ============================================================
# Training Loop
# ============================================================
def train(episodes=3000, render_every=200, max_steps=2000, save_every=100, load_model=False):
    agent = QAgent(total_episodes=episodes)

    # 🔹 Load previous model if available and requested
    if load_model:
        loaded = agent.load()
        if loaded:
            print("🔁 Continuing training from saved Q-table...")
        else:
            print("🚀 Starting new training session (no saved table found).")

    env = None
    successes = 0

    for ep in range(1, episodes + 1):
        if env:
            env.close()
        do_render = (render_every and ep % render_every == 0)
        env = MonsterTruckFlipEnv(render=do_render, frame_skip=10, max_steps=max_steps, realtime=True)

        obs = env.reset()
        ep_ret = 0.0

        for _ in range(max_steps):
            a = agent.act(obs)
            next_obs, r, done, info = env.step(a)
            agent.learn(obs, a, r, next_obs, done)
            obs = next_obs
            ep_ret += r
            if done:
                if info.get("success", False):
                    successes += 1
                break

        agent.decay_eps()
        if ep % 10 == 0:
            print(f"Ep {ep:4d} | Ret {ep_ret:7.2f} | eps {agent.eps:5.3f} | success {successes}")

        if save_every and ep % save_every == 0:
            agent.save()

    env.close()
    agent.save()
    print(f"✅ Training complete! Successes: {successes}/{episodes}")


if __name__ == "__main__":
    # 🔹 Set load_model=True to resume training from existing Q-table
    train(episodes=3000, render_every=200, max_steps=2000, save_every=100, load_model=True)
