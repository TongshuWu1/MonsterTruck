# ===============================================================
# TileQLearning_Monstertruck_PitchOnly_FAST.py
# Faster-learning Tile-coded Q for pitch-only truck flip
#  - Obs uses signed pitch [-180,180) to remove 0/360 discontinuity
#  - Pitch-rate encoded in [-300,300] to concentrate tiles
#  - 24x24 tiles, 8 tilings, higher alpha
#  - Momentum-friendly 3-zone shaping with moderate weights
#  - FIXED: α scaling (no double-divide), small per-tiling offsets,
#           clamped indexing (no modulo) for non-periodic dimensions
# ===============================================================

import os, json, math, time
import numpy as np
import matplotlib.pyplot as plt
import mujoco
from mujoco.glfw import glfw


def clip(v, lo, hi):
    return lo if v < lo else (hi if v > hi else v)


# ============================================================
# Environment — pitch-only
# ============================================================
class MonsterTruckFlipEnvPitch:
    def __init__(self, xml_path="monstertruck.xml",
                 frame_skip=5, max_steps=1500,
                 render=False, realtime=False,
                 wy_sign=+1):
        if not os.path.exists(xml_path):
            raise FileNotFoundError(f"Cannot find {xml_path}")
        self.model = mujoco.MjModel.from_xml_path(xml_path)
        self.data = mujoco.MjData(self.model)
        self.frame_skip = frame_skip
        self.max_steps = max_steps
        self.render_enabled = render
        self.realtime = realtime
        self.dt = self.model.opt.timestep
        self.wy_sign = float(wy_sign)

        # 9-level throttle → both rear wheels
        self.actions = np.linspace(-1.0, 1.0, 9)
        self.last_throttle = 0.0
        self.prev_throttle = 0.0
        self.step_count = 0

        self.body_id = mujoco.mj_name2id(self.model, mujoco.mjtObj.mjOBJ_BODY, "chassis")

        # ===== Reward weights (moderate; learn faster) =====
        self.R = dict(
            dist=5,          # bounded potential cost (tanh^2)
            spin_far=300,      # reward |ω| when far to encourage swing
            progress_mid=100, # reward toward motion mid-distance
            brake_near=50,   # penalize |ω| near goal (braking)
            energy=1.0,
            jerk=1.0,
            time=1.0,
            success=8000.0
        )

        # Zone thresholds (deg)
        self.e_far  = 90.0
        self.e_near = 25.0

        # Speed normalizers (deg/s)
        self.omega_scale_far  = 220.0
        self.omega_scale_mid  = 140.0
        self.omega_scale_near = 140.0

        # Potential scale (deg) for tanh(|e|/scale)
        self.potential_scale_deg = 90.0

        # State (deg): unwrapped for reward; wrapped for signed obs
        self.pitch_unwrapped_deg = 180.0
        self.pitch_deg = 180.0
        self.pitch_rate_deg = 0.0

        # Success tolerances & hold
        self.ang_tol = 2.5    # deg
        self.rate_tol = 6.0   # deg/s
        self.hold_steps = 10  # consecutive steps required
        self._hold = 0

        self._viewer_ready = False
        if render:
            self._init_viewer()

    # ---------- Rendering ----------
    def _init_viewer(self):
        if self._viewer_ready:
            return
        if not glfw.init():
            raise RuntimeError("GLFW init failed")
        self.window = glfw.create_window(1000, 800, "MonsterTruck Pitch-Only (TileQ)", None, None)
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

    # ---------- Core API ----------
    def reset(self):
        mujoco.mj_resetData(self.model, self.data)
        self.data.qpos[:3] = np.array([0.0, 0.0, 0.2])
        self.data.qvel[:] = 0.0
        self.data.qpos[3:7] = np.array([0, 1, 0, 0])  # upside-down
        mujoco.mj_forward(self.model, self.data)

        self.step_count = 0
        self.prev_throttle = 0.0
        self.last_throttle = 0.0

        self.pitch_unwrapped_deg = 180.0
        self.pitch_deg = 180.0
        self.pitch_rate_deg = 0.0
        self._hold = 0
        return self._get_obs()

    def _body_local_angvel(self):
        res = np.zeros(6, dtype=np.float64)
        try:
            mujoco.mj_objectVelocity(self.model, self.data,
                                     mujoco.mjtObj.mjOBJ_BODY, self.body_id,
                                     res, 1)
        except TypeError:
            mujoco.mj_objectVelocity(self.model, self.data,
                                     mujoco.mjtObj.mjOBJ_BODY, self.body_id,
                                     1, res)
        return res[3:6]

    def _wrap360(self, a):
        a = a % 360.0
        if a < 0:
            a += 360.0
        return a

    @staticmethod
    def _to_signed_deg(a_wrapped):
        # Map [0,360) → [-180,180)
        return ((a_wrapped + 180.0) % 360.0) - 180.0

    def _get_obs(self):
        # Use signed pitch for learning speed; clamp rate for denser tiles
        signed_pitch = self._to_signed_deg(self.pitch_deg)
        rate = clip(self.pitch_rate_deg, -300.0, 300.0)
        return np.array([signed_pitch, rate], dtype=np.float32)

    def step(self, a_idx):
        prev_throttle = self.last_throttle
        throttle = float(self.actions[a_idx])
        self.last_throttle = throttle
        done, success = False, False

        # Integrate local wy → update UNWRAPPED pitch
        dtheta_deg = 0.0
        wy_last = 0.0
        for _ in range(self.frame_skip):
            if self.model.nu >= 2:
                self.data.ctrl[0] = throttle
                self.data.ctrl[1] = throttle
            mujoco.mj_step(self.model, self.data)
            angvel_local = self._body_local_angvel()
            wy_last = float(angvel_local[1])  # rad/s
            dtheta_deg += (wy_last * 180.0 / math.pi) * self.dt
            if self.realtime:
                time.sleep(self.dt)

        dtheta_deg *= self.wy_sign
        self.pitch_unwrapped_deg += dtheta_deg
        self.pitch_deg = self._wrap360(self.pitch_unwrapped_deg)
        self.pitch_rate_deg = self.wy_sign * (wy_last * 180.0 / math.pi)

        # ---------------- Reward (momentum-friendly) ----------------
        R = self.R
        reward = 0.0

        e  = self.pitch_unwrapped_deg
        ae = abs(e)
        w  = self.pitch_rate_deg
        aw = abs(w)

        # (A) Smooth bounded potential
        phi = math.tanh(ae / self.potential_scale_deg) ** 2
        reward -= R["dist"] * phi

        # (B) Three-zone velocity shaping
        sgn_to_goal = -1.0 if e > 0 else (1.0 if e < 0 else 0.0)
        toward = sgn_to_goal * w

        if ae > self.e_far:
            reward += R["spin_far"] * (aw / self.omega_scale_far) ** 2
        elif ae > self.e_near:
            gate = (ae - self.e_near) / (self.e_far - self.e_near)
            if toward > 0:
                reward += R["progress_mid"] * gate * (toward / self.omega_scale_mid)
        else:
            reward -= R["brake_near"] * (aw / self.omega_scale_near) ** 2

        # (C) Energy, (D) Jerk, (E) Time
        reward -= R["energy"] * (throttle ** 2)
        du = throttle - prev_throttle
        reward -= R["jerk"] * (du ** 2)
        reward -= R["time"]

        # (F) Success with hold
        within = (ae < self.ang_tol) and (abs(w) < self.rate_tol)
        self._hold = self._hold + 1 if within else 0
        if self._hold >= self.hold_steps:
            reward += R["success"]
            success, done = True, True

        self.step_count += 1
        if self.step_count >= self.max_steps:
            done = True

        reward = clip(reward, -400, 400)
        self._render()

        self.prev_throttle = throttle
        return self._get_obs(), reward, done, {"success": success}

    def close(self):
        if self._viewer_ready:
            glfw.destroy_window(self.window)
            glfw.terminate()
            self._viewer_ready = False


# ============================================================
# Tile Coder (2D: signed_pitch [-180,180), pitch_rate [-300,300])
# ============================================================
class TileCoder:
    def __init__(self, obs_ranges, tiles_per_dim=(24, 24), num_tilings=8):
        self.obs_ranges = np.array(obs_ranges, dtype=np.float64)  # shape (2, 2)
        self.tiles_per_dim = np.array(tiles_per_dim, dtype=np.int32)
        self.num_tilings = int(num_tilings)
        assert len(self.obs_ranges) == len(self.tiles_per_dim) == 2

        self.dims = 2
        self.tiles_per_tiling = int(np.prod(self.tiles_per_dim))
        self.n_features = self.num_tilings * self.tiles_per_tiling

        # Spans and normalized tile sizes
        self.spans = self.obs_ranges[:, 1] - self.obs_ranges[:, 0]
        self.spans[self.spans == 0] = 1.0
        tile_size = 1.0 / self.tiles_per_dim.astype(np.float64)

        # Small per-tiling offsets: fractions of one tile per dim
        # Different shift for each dim/tiling; stays within ~1 tile total
        self.offsets = np.zeros((self.num_tilings, self.dims), dtype=np.float64)
        for t in range(self.num_tilings):
            for d in range(self.dims):
                # e.g., for 8 tilings and 24 tiles/dim, offsets are ~ (0.03..0.47) tile widths
                self.offsets[t, d] = ((t + (d + 1) * 0.5) / self.num_tilings) * tile_size[d]

    def encode(self, obs):
        obs = np.asarray(obs, dtype=np.float64)
        # Normalize to [0,1]
        obs_norm = (obs - self.obs_ranges[:, 0]) / self.spans

        active = []
        for t in range(self.num_tilings):
            shifted = obs_norm + self.offsets[t]
            # Convert to tile coordinates; clamp to valid index range
            coords = np.floor(shifted * self.tiles_per_dim).astype(int)
            coords = np.clip(coords, 0, self.tiles_per_dim - 1)  # no modulo (avoids aliasing)
            idx_in_tiling = int(coords[0] * self.tiles_per_dim[1] + coords[1])
            feat_idx = t * self.tiles_per_tiling + idx_in_tiling
            active.append(feat_idx)
        return active


# ============================================================
# Tile-coded Q-learning Agent
# ============================================================
class TileQAgent:
    def __init__(self,
                 obs_ranges=((-180.0, 180.0), (-300.0, 300.0)),  # signed pitch, clipped rate
                 tiles_per_dim=(24, 24), num_tilings=8,
                 n_actions=9,
                 alpha=0.2,   # base step size; scaled by num_tilings
                 gamma=0.995,
                 eps_start=1.0, eps_end=0.05, total_episodes=4000):
        self.tc = TileCoder(obs_ranges, tiles_per_dim, num_tilings)
        self.n_actions = int(n_actions)
        self.gamma = gamma
        # Scale by num tilings once (no per-step division)
        self.alpha = float(alpha) / self.tc.num_tilings
        self.eps_start, self.eps_end = eps_start, eps_end
        self.total_episodes = total_episodes
        self.eps = eps_start
        self.eps_decay = (eps_end / eps_start) ** (1.0 / total_episodes)
        self.w = np.zeros((self.n_actions, self.tc.n_features), dtype=np.float32)

    def q_vals(self, obs):
        feats = self.tc.encode(obs)
        return np.sum(self.w[:, feats], axis=1), feats

    def act(self, obs):
        q, _ = self.q_vals(obs)
        if np.random.rand() < self.eps:
            return np.random.randint(self.n_actions)
        return int(np.argmax(q))

    def act_greedy(self, obs):
        q, _ = self.q_vals(obs)
        return int(np.argmax(q))

    def learn(self, obs, a, r, next_obs, done):
        q_sa, feats = self.q_vals(obs)
        q_s = q_sa[a]
        if done:
            target = r
        else:
            q_next, _ = self.q_vals(next_obs)
            target = r + self.gamma * np.max(q_next)
        delta = target - q_s
        # FIX: do NOT divide by len(feats); alpha is already scaled by num_tilings
        step = self.alpha * delta
        self.w[a, feats] += step

    def decay_eps(self):
        self.eps = max(self.eps_end, self.eps * self.eps_decay)

    def save(self, path="tileq_weights.npy", meta="tileq_meta.json"):
        np.save(path, self.w)
        with open(meta, "w") as f:
            json.dump({
                "obs_ranges": [[-180.0, 180.0], [-300.0, 300.0]],
                "tiles_per_dim": list(self.tc.tiles_per_dim.tolist()),
                "num_tilings": int(self.tc.num_tilings),
                "n_actions": self.n_actions
            }, f, indent=2)
        print(f"💾 Saved TileQ weights to {path}")

    def load(self, path="tileq_weights.npy", meta="tileq_meta.json"):
        if not (os.path.exists(path) and os.path.exists(meta)):
            return False
        try:
            with open(meta, "r") as f:
                m = json.load(f)
            self.tc = TileCoder(tuple(map(tuple, m["obs_ranges"])),
                                tuple(m["tiles_per_dim"]),
                                int(m["num_tilings"]))
            self.n_actions = int(m["n_actions"])
            self.w = np.load(path)
            print("✅ Loaded TileQ weights.")
            return True
        except Exception as e:
            print(f"⚠️ Failed to load TileQ: {e}")
            return False


# ============================================================
# Evaluation helper
# ============================================================
def evaluate_episode(env, agent, max_steps=1200, render=False):
    env.render_enabled = render
    env.realtime = render
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


# ============================================================
# Training
# ============================================================
def train(episodes=4000, render_every=0, max_steps=1500, load_model=False, eval_every=100,
          wy_sign=+1,
          tiles_per_dim=(24, 24), num_tilings=8,
          alpha=0.2, gamma=0.995, eps_start=1.0, eps_end=0.05):

    agent = TileQAgent(
        obs_ranges=((-180.0, 180.0), (-300.0, 300.0)),
        tiles_per_dim=tiles_per_dim,
        num_tilings=num_tilings,
        n_actions=9,
        alpha=alpha,
        gamma=gamma,
        eps_start=eps_start, eps_end=eps_end, total_episodes=episodes,
    )

    if load_model and agent.load():
        print("🔁 Continuing training from saved TileQ weights")

    env = MonsterTruckFlipEnvPitch(render=False, realtime=False,
                                   frame_skip=5, max_steps=max_steps,
                                   wy_sign=wy_sign)

    successes = 0
    rewards_list, success_flags = [], []
    eval_ep_indices, eval_rewards = [], []

    for ep in range(1, episodes + 1):
        do_render = (render_every and ep % render_every == 0)
        env.render_enabled = do_render
        env.realtime = do_render

        obs = env.reset()
        ep_ret = 0.0
        success_flag = 0

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

        if eval_every and (ep % eval_every == 0):
            er, es = evaluate_episode(env, agent, max_steps=max_steps, render=False)
            eval_ep_indices.append(ep)
            eval_rewards.append(er)
            print(f"   [Eval @ Ep {ep}] reward={er:.2f} | success={es}")

    env.close()
    agent.save()
    print(f"✅ Training complete! Successes: {successes}/{episodes}")

    np.save("training_rewards.npy", np.array(rewards_list, dtype=np.float32))
    np.save("eval_rewards.npy",   np.array(eval_rewards, dtype=np.float32))
    np.save("eval_episodes.npy", np.array(eval_ep_indices, dtype=np.int32))

    if len(eval_ep_indices) > 0:
        plt.figure(figsize=(8, 4.8))
        plt.plot(eval_ep_indices, eval_rewards, marker='o')
        plt.xlabel("Episode")
        plt.ylabel("Evaluation Reward")
        plt.title("Evaluation Reward vs Episode (Pitch-Only Tile Q, FAST)")
        plt.grid(True, alpha=0.3)
        plt.tight_layout()
        plt.savefig("eval_rewards.png", dpi=150)
        print("📈 Saved evaluation plot to eval_rewards.png")

    return rewards_list, (eval_ep_indices, eval_rewards)


if __name__ == "__main__":
    assert os.path.exists("monstertruck.xml"), "Put monstertruck.xml next to this script."

    rewards, (eval_eps, eval_rewards) = train(
        episodes=2000,
        render_every=50,
        max_steps=1500,
        load_model=False,
        eval_every=50,
        wy_sign=+1,
        tiles_per_dim=(24, 24),
        num_tilings=8,
        alpha=0.1,
        gamma=0.995,
        eps_start=1.0, eps_end=0.05,
    )
    print("💾 Saved all episode rewards to training_rewards.npy")
