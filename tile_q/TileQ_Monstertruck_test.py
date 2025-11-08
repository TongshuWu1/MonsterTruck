# ===============================================================
# TileQLearning_Monstertruck_Pitch_SIGNED_FORWARD_stable_min.py
# - Keeps YOUR reward design unchanged
# - Adds stability fixes to prevent "forgetting":
#     * Per-dim random offsets for each tiling (deterministic by seed)
#     * Double Q-learning with per-feature decayed step sizes
#     * Best-snapshot restore (early stop)
# - Signed flip angle φ in [-180°, +180°]
# ===============================================================

import os, math, time, json
import numpy as np
import matplotlib.pyplot as plt
import mujoco
from mujoco.glfw import glfw


def clip(v, lo, hi):
    return lo if v < lo else (hi if v > hi else v)


# ===============================================================
# Environment (Signed φ + YOUR reward design intact)
# ===============================================================
class MonsterTruckFlipEnvPitchSigned:
    def __init__(self, xml_path="monstertruck.xml",
                 frame_skip=10, max_steps=2000,
                 render=False, realtime=False,
                 num_actions=9, seed: int = 0):
        if not os.path.exists(xml_path):
            raise FileNotFoundError(f"Cannot find {xml_path}")
        self.model = mujoco.MjModel.from_xml_path(xml_path)
        self.data = mujoco.MjData(self.model)
        self.frame_skip = frame_skip
        self.max_steps = max_steps
        self.render_enabled = render
        self.realtime = realtime
        self.dt = self.model.opt.timestep

        # reproducible RNG for env-only randomness
        self.rng = np.random.default_rng(seed)

        self.actions = np.linspace(-1.0, 1.0, num_actions).astype(np.float32)
        self.last_throttle = 0.0
        self.step_count = 0
        self.hold_counter = 0
        self.hold_needed = 4

        self.body_id = mujoco.mj_name2id(self.model, mujoco.mjtObj.mjOBJ_BODY, "chassis")

        # === YOUR reward weights (unchanged) ===
        self.R = dict(
            position=3.00,     # directional distance penalty via d_fwd
            momentum=2.0,      # MountainCar-style momentum reward (|ω| when far)
            stop_boost=0.0,    # near-upright brake on ω^2
            energy=0.5,        # control effort penalty
            time=1.0,          # per-step time cost
            jerk=0.5,          # |Δu|
            success=3000.0     # terminal bonus
        )

        # State memory
        self.prev_phi_deg = 0.0
        self.last_rate_deg = 0.0
        self._flip_sign = 0.0  # +1 forward, -1 backward; resolved after slight motion

        # Viewer
        self._viewer_ready = False
        if render:
            self._init_viewer()

    # ---------------- Rendering ----------------
    def _init_viewer(self):
        if self._viewer_ready:
            return
        if not glfw.init():
            raise RuntimeError("GLFW init failed")
        self.window = glfw.create_window(1000, 800, "MonsterTruck Flip (Signed φ, Forward Goal)", None, None)
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

    # ---------------- Signed flip angle φ ----------------
    def _get_signed_flip_deg(self):
        """
        Signed angle φ between world up z_w and roof normal r_w = -Z_body,
        measured about the body Y axis (y_w = R[:,1]):

            φ = sign( (z_w × r_w) · y_w ) * arccos(z_w · r_w),  φ ∈ [-180, +180]

        Convention:
          φ = +180° : forward upright
          φ = -180° : backward upright
          φ =   0°  : upside-down
        """
        R = self.data.xmat[self.body_id].reshape(3, 3)  # body->world
        z_w = np.array([0.0, 0.0, 1.0], dtype=float)
        r_w = -R[:, 2]   # roof normal in world
        y_w =  R[:, 1]   # body Y in world (pitch axis)

        cosang = clip(float(np.dot(z_w, r_w)), -1.0, 1.0)
        theta = math.degrees(math.acos(cosang))   # unsigned [0, 180]
        tri = float(np.dot(np.cross(z_w, r_w), y_w))  # signed area about y_w

        # Resolve flip sign once (avoid ambiguity at φ≈0)
        if theta > 0.5 and abs(tri) > 1e-9:
            self._flip_sign = 1.0 if tri > 0.0 else -1.0
        s = self._flip_sign if self._flip_sign != 0.0 else 1.0
        return s * theta

    # ---------------- Core API ----------------
    def reset(self):
        mujoco.mj_resetData(self.model, self.data)
        # Upside-down spawn
        self.data.qpos[:3] = np.array([0, 0, 0.3]) + 0.01 * self.rng.normal(size=3)
        self.data.qvel[:] = 0.0
        self.data.qpos[3:7] = np.array([0, 1, 0, 0])  # 180° about X → upside-down
        mujoco.mj_forward(self.model, self.data)

        self.step_count = 0
        self.last_throttle = 0.0
        self.hold_counter = 0
        self._flip_sign = 0.0

        phi = self._get_signed_flip_deg()   # ~0° at start
        self.prev_phi_deg = phi
        self.last_rate_deg = 0.0
        return np.array([phi, 0.0], dtype=np.float32)

    def step(self, a_idx):
        throttle = float(self.actions[a_idx])
        prev_throttle = self.last_throttle
        self.last_throttle = throttle
        done, success = False, False

        # Apply control for frame_skip steps
        for _ in range(self.frame_skip):
            for i in range(min(2, self.model.nu)):
                self.data.ctrl[i] = throttle
            mujoco.mj_step(self.model, self.data)
            if self.realtime:
                time.sleep(self.dt)

        # Signed angle and rate (DEGREES)
        phi_deg = self._get_signed_flip_deg()                  # φ ∈ [-180, +180]
        horizon_dt = max(self.frame_skip * self.dt, 1e-6)
        dphi_deg = (phi_deg - self.prev_phi_deg)
        phi_rate_deg = dphi_deg / horizon_dt                   # deg/s
        phi_rate_rad = math.radians(phi_rate_deg)

        if self.render_enabled:
            print(
                f"[Render] step={self.step_count:4d} | "
                f"phi={phi_deg:+7.2f}° | dφ/dt={phi_rate_deg:+8.2f}°/s"
            )

        # --------- YOUR FORWARD-GOAL DESIGN (unchanged) ----------
        d_fwd = (180.0 - phi_deg) / 180.0           # in [0,2]; >1 when φ < 0 (backward side)
        d_fwd_clip2 = np.clip(d_fwd, 0.0, 2.0)      # for penalty shaping
        d_fwd_clip1 = np.clip(d_fwd, 0.0, 1.0)      # for near/far gates

        # Position penalty (unchanged)
        pos_penalty = self.R["position"] * (np.tanh(2.2 * d_fwd_clip2) ** 2)

        # Gates (unchanged)
        near_gate = np.exp(-12.0 * d_fwd_clip1)
        far_gate  = 1.0 - np.exp(-8.0  * d_fwd_clip1)

        # Momentum reward (unchanged)
        mom_reward = self.R["momentum"] * far_gate * abs(phi_rate_rad)

        # Near-upright braking (unchanged)
        vel_brake = -self.R["stop_boost"] * near_gate * (phi_rate_rad ** 2)

        energy_pen = self.R["energy"] * (throttle ** 2)
        time_pen   = self.R["time"]
        jerk_pen   = self.R["jerk"] * abs(throttle - prev_throttle)

        # === YOUR reward equation (unchanged) ===
        reward = -pos_penalty + mom_reward + vel_brake - energy_pen - time_pen - jerk_pen
        reward = clip(reward, -10.0, 10.0)

        # Success condition (UNCHANGED from your code)
        if (abs(phi_deg) > 178.0):
            self.hold_counter += 1
            if self.hold_counter >= self.hold_needed:
                reward += self.R["success"]
                success, done = True, True
        else:
            self.hold_counter = 0

        self.prev_phi_deg = phi_deg
        self.last_rate_deg = phi_rate_deg
        self.step_count += 1
        if self.step_count >= self.max_steps:
            done = True

        self._render()

        next_obs = np.array([phi_deg, phi_rate_deg], dtype=np.float32)
        info = {
            "success": success,
            "phi_deg": phi_deg,
            "phi_rate_deg": phi_rate_deg
        }
        return next_obs, float(reward), done, info

    def close(self):
        if self._viewer_ready:
            glfw.destroy_window(self.window)
            glfw.terminate()
            self._viewer_ready = False


# ===============================================================
# Tile Coder with per-dimension random offsets (deterministic via seed)
# ===============================================================
class TileCoder:
    def __init__(self, lows, highs, n_tiles, n_tilings, seed: int = 0, random_offsets: bool = True):
        self.lows = np.array(lows, dtype=np.float32)
        self.highs = np.array(highs, dtype=np.float32)
        self.n_tiles = np.array(n_tiles, dtype=np.int32)
        self.n_tilings = int(n_tilings)
        self.dim = len(lows)

        if random_offsets:
            rng = np.random.default_rng(seed)
            # offsets in [0,1) scaled by tile counts per dim
            self.offsets = rng.random((self.n_tilings, self.dim)) / self.n_tiles
        else:
            til_axis = (np.arange(self.n_tilings) * 2 + 1) / (2.0 * self.n_tilings)
            self.offsets = np.tile(til_axis[:, None], (1, self.dim)) / self.n_tiles

    def encode(self, s):
        s = np.array(s, dtype=np.float32)
        ratios = (s - self.lows) / (self.highs - self.lows + 1e-8)  # normalize to ~[0,1]
        ratios = np.clip(ratios, 0.0, 0.999999)
        idxs = []
        base = int(np.prod(self.n_tiles))
        for t in range(self.n_tilings):
            shifted = (ratios + self.offsets[t]) * self.n_tiles
            tile_coords = np.floor(shifted).astype(int)
            flat = np.ravel_multi_index(tile_coords, self.n_tiles, mode='clip')
            idxs.append(t * base + flat)
        return idxs

    @property
    def total_tiles(self):
        return self.n_tilings * int(np.prod(self.n_tiles))


# ===============================================================
# Tile Q Agent: Double Q + per-feature decayed step sizes
# ===============================================================
class TileQAgent:
    def __init__(self, obs_low, obs_high,
                 n_tiles=(48, 48), n_tilings=8,
                 n_actions=9, alpha=0.02, gamma=0.98,
                 eps_start=0.99, eps_end=0.02, total_episodes=5000,
                 seed: int = 0, use_double_q: bool = True):

        self.gamma = gamma
        self.n_actions = n_actions
        self.use_double_q = use_double_q

        # base stepsize per tiling; we will decay per-feature
        self.alpha0 = alpha / float(n_tilings)

        # exploration schedule (exponential to floor)
        self.eps_start, self.eps_end = eps_start, eps_end
        self.total_episodes = total_episodes
        self.current_episode = 0
        self.eps = eps_start

        self.rng = np.random.default_rng(seed)

        self.tc = TileCoder(obs_low, obs_high, n_tiles, n_tilings, seed=seed, random_offsets=True)
        self.n_features = self.tc.total_tiles

        if self.use_double_q:
            self.wA = np.zeros((n_actions, self.n_features), dtype=np.float32)
            self.wB = np.zeros((n_actions, self.n_features), dtype=np.float32)
        else:
            self.w = np.zeros((n_actions, self.n_features), dtype=np.float32)

        # per-(action,feature) visit counts for decayed step sizes
        self.counts = np.zeros((n_actions, self.n_features), dtype=np.int32)

        # TD tracking
        self._delta_accum = 0.0
        self._delta_count = 0

    # ---------- schedules ----------
    def update_epsilon(self):
        # exponential decay to floor
        self.eps = max(self.eps_end, self.eps_start * (0.995 ** self.current_episode))

    def decay_eps(self):
        self.current_episode += 1
        self.update_epsilon()

    # ---------- Q helpers ----------
    def _Qs_from(self, W, obs):
        idxs = self.tc.encode(obs)
        return np.array([np.sum(W[a, idxs]) for a in range(self.n_actions)], dtype=np.float32)

    def Qs(self, obs):
        if self.use_double_q:
            return 0.5 * (self._Qs_from(self.wA, obs) + self._Qs_from(self.wB, obs))
        else:
            idxs = self.tc.encode(obs)
            return np.array([np.sum(self.w[a, idxs]) for a in range(self.n_actions)], dtype=np.float32)

    # ---------- act ----------
    def act(self, obs):
        if self.rng.random() < self.eps:
            return int(self.rng.integers(self.n_actions))
        return int(np.argmax(self.Qs(obs)))

    def act_greedy(self, obs):
        return int(np.argmax(self.Qs(obs)))

    # ---------- learn ----------
    def learn(self, obs, a, r, next_obs, done):
        idxs = self.tc.encode(obs)
        next_idxs = self.tc.encode(next_obs)

        if self.use_double_q:
            # randomly pick which head to update (Double Q)
            if self.rng.random() < 0.5:
                Qa_next = np.array([np.sum(self.wA[aa, next_idxs]) for aa in range(self.n_actions)], dtype=np.float32)
                a_star = int(np.argmax(Qa_next))
                q_sa   = float(np.sum(self.wA[a, idxs]))
                target = r if done else r + self.gamma * float(np.sum(self.wB[a_star, next_idxs]))
                delta  = target - q_sa
                for idx in idxs:
                    self.counts[a, idx] += 1
                    lr = self.alpha0 / math.sqrt(1 + self.counts[a, idx])
                    self.wA[a, idx] += lr * delta
            else:
                Qb_next = np.array([np.sum(self.wB[aa, next_idxs]) for aa in range(self.n_actions)], dtype=np.float32)
                a_star = int(np.argmax(Qb_next))
                q_sa   = float(np.sum(self.wB[a, idxs]))
                target = r if done else r + self.gamma * float(np.sum(self.wA[a_star, next_idxs]))
                delta  = target - q_sa
                for idx in idxs:
                    self.counts[a, idx] += 1
                    lr = self.alpha0 / math.sqrt(1 + self.counts[a, idx])
                    self.wB[a, idx] += lr * delta
        else:
            q_sa = float(np.sum(self.w[a, idxs]))
            target = r if done else r + self.gamma * float(np.max(self.Qs(next_obs)))
            delta = target - q_sa
            for idx in idxs:
                self.counts[a, idx] += 1
                lr = self.alpha0 / math.sqrt(1 + self.counts[a, idx])
                self.w[a, idx] += lr * delta

        self._delta_accum += abs(delta)
        self._delta_count += 1

    def avg_update(self):
        if self._delta_count == 0:
            return 0.0
        v = self._delta_accum / self._delta_count
        self._delta_accum = 0.0
        self._delta_count = 0
        return v

    # ---------- snapshot ----------
    def snapshot(self):
        if self.use_double_q:
            return (self.wA.copy(), self.wB.copy(), self.counts.copy(), self.eps)
        else:
            return (self.w.copy(), self.counts.copy(), self.eps)

    def load_snapshot(self, snap):
        if self.use_double_q:
            self.wA[:], self.wB[:], self.counts[:], self.eps = snap
        else:
            self.w[:], self.counts[:], self.eps = snap


# ===============================================================
# Eval helpers (unchanged)
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


def run_greedy_trace(env, agent, max_steps=1500):
    obs = env.reset()

    t_steps, phi_deg_hist, phi_rate_hist, throttle_hist, rewards = [], [], [], [], []

    step = 0
    while step < max_steps:
        p  = float(obs[0])   # φ (deg, signed)
        pr = float(obs[1])   # φ rate (deg/s)
        a = agent.act_greedy(obs)

        t_steps.append(step)
        phi_deg_hist.append(p)
        phi_rate_hist.append(pr)
        throttle_hist.append(float(env.actions[a]))

        obs, r, done, info = env.step(a)
        rewards.append(float(r))

        step += 1
        if done:
            break

    traces = dict(
        t=np.array(t_steps, dtype=int),
        phi_deg=np.array(phi_deg_hist, dtype=float),
        phi_rate_deg=np.array(phi_rate_hist, dtype=float),
        throttle=np.array(throttle_hist, dtype=float),
        reward=np.array(rewards, dtype=float),
        cum_reward=np.cumsum(np.array(rewards, dtype=float)),
    )
    return traces


def save_episode_traces_figure(traces, outfile="tileq_episode_summary.png"):
    def ema(x, alpha=0.25):
        if len(x) == 0: return x
        y = np.empty_like(x, dtype=float)
        y[0] = x[0]
        for i in range(1, len(x)):
            y[i] = alpha * x[i] + (1 - alpha) * y[i - 1]
        return y

    t   = traces["t"]
    p   = traces["phi_deg"]
    pr  = traces["phi_rate_deg"]
    thr = traces["throttle"]
    cre = traces["cum_reward"]
    thr_s = ema(thr, alpha=0.3)

    fig, axs = plt.subplots(2, 2, figsize=(12, 9))
    (ax1, ax2), (ax3, ax4) = axs

    ax1.plot(t, p, lw=2)
    ax1.set_title("Signed Flip Angle φ vs Timestep (0°=upside-down, +180°=goal)")
    ax1.set_xlabel("Timestep")
    ax1.set_ylabel("φ (deg)")
    ax1.grid(True, alpha=0.3)

    ax2.plot(p, pr, lw=1.8)
    ax2.set_title("dφ/dt vs φ")
    ax2.set_xlabel("φ (deg)")
    ax2.set_ylabel("dφ/dt (deg/s)")
    ax2.grid(True, alpha=0.3)

    ax3.plot(t, thr, lw=1, alpha=0.25)
    ax3.plot(t, thr_s, lw=2)
    ax3.set_title("Throttle vs Timestep (smoothed)")
    ax3.set_xlabel("Timestep")
    ax3.set_ylabel("Throttle")
    ax3.grid(True, alpha=0.3)

    ax4.plot(t[:len(cre)], cre, lw=2)
    ax4.set_title("Cumulative Reward vs Timestep")
    ax4.set_xlabel("Timestep")
    ax4.set_ylabel("Cumulative Reward")
    ax4.grid(True, alpha=0.3)

    fig.suptitle("MonsterTruck Flip — Greedy Episode Summary (Signed φ + Forward Distance)", fontsize=14)
    fig.tight_layout(rect=[0, 0.03, 1, 0.96])
    fig.savefig(outfile, dpi=220)
    plt.close(fig)
    print(f"📊 Saved episode traces figure to {outfile}")


# ===============================================================
# Training (keeps your obs range; adds best-snapshot early stop)
# ===============================================================
def train_tileq(episodes=1000, max_steps=1500, eval_every=50, num_actions=9, seed: int = 0):
    np.random.seed(seed)

    env = MonsterTruckFlipEnvPitchSigned(render=False, realtime=False,
                                         frame_skip=10, max_steps=max_steps,
                                         num_actions=num_actions, seed=seed)

    # OBS: [φ (signed), φ_rate]  — kept as in your code
    lows  = [-180.0, -720.0]
    highs = [ 180.0,  720.0]

    agent = TileQAgent(lows, highs,
                       n_tiles=(48, 48), n_tilings=8,
                       n_actions=len(env.actions),
                       total_episodes=episodes,
                       eps_start=0.99, eps_end=0.02,
                       alpha=0.02, gamma=0.98,
                       seed=seed, use_double_q=True)

    rewards_list, success_flags, eval_ep_indices, eval_rewards = [], [], [], []
    successes = 0

    # Best-snapshot early stop (to avoid "learned then lost")
    best_eval = -float("inf")
    best_snap = None
    no_improve = 0
    patience = 6  # eval rounds

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

        # Evaluation (render only the first of the 3)
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

            # snapshot logic (no reward-design changes)
            if mean_r > best_eval:
                best_eval = mean_r
                best_snap = agent.snapshot()
                no_improve = 0
            else:
                no_improve += 1
                # tighten exploration floor if drifting
                if no_improve == 3:
                    agent.eps_end = max(0.005, agent.eps_end * 0.5)
                if no_improve >= patience:
                    if best_snap is not None:
                        agent.load_snapshot(best_snap)
                    print(f"   Early stop: restoring best snapshot (avg_reward={best_eval:.1f})")
                    break

    env.render_enabled = False
    env.realtime = False

    # Save eval curve
    if len(eval_ep_indices) > 0:
        plt.figure(figsize=(9, 5))
        plt.plot(eval_ep_indices, eval_rewards, 'o--', alpha=0.45, label='Eval reward')
        window = 5
        if len(eval_rewards) >= window:
            smoothed = np.convolve(eval_rewards, np.ones(window)/window, mode='valid')
            plt.plot(eval_ep_indices[window-1:], smoothed, '-', lw=2.2, label=f'{window}-pt Moving Avg')
        plt.xlabel("Episode")
        plt.ylabel("Evaluation Reward")
        plt.title("TileQ — Signed φ with forward-goal distance")
        plt.legend()
        plt.grid(True, alpha=0.3)
        plt.tight_layout()
        plt.savefig("tileq_pitch_eval.png", dpi=200)
        plt.close()
        print("📈 Saved eval plot to tileq_pitch_eval.png")

    traces = run_greedy_trace(env, agent, env.max_steps)
    save_episode_traces_figure(traces, outfile="tileq_episode_summary.png")

    env.close()
    return rewards_list, (eval_ep_indices, eval_rewards)


if __name__ == "__main__":
    # set your favorite seed here
    train_tileq(episodes=600, max_steps=1500, eval_every=50, num_actions=9, seed=42)
