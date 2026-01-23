# ===============================================================
# QLearning_Monstertruck_PitchSigned_LinMom_BATCH10_TARGET90.py
# Target: |phi| = 90 degrees (instead of 180 upright)
# ===============================================================

import os, math, time, random, json, csv
import numpy as np
import matplotlib.pyplot as plt
import mujoco
from mujoco.glfw import glfw


# ---------------- utils ----------------
def clip(v, lo, hi):
    return lo if v < lo else (hi if v > hi else v)

def angdiff_deg(curr, prev):
    """Minimal signed angle difference in degrees, result in [-180, 180)."""
    return ((curr - prev + 180.0) % 360.0) - 180.0


# ===============================================================
# Environment: Signed φ + target |φ|=90°
# - Linear momentum term pushes TOWARD target (not always toward 180°)
# - Success: hysteresis + velocity gate around target
# ===============================================================
class MonsterTruckFlipEnvPitchSigned:
    def __init__(self, xml_path="monstertruck.xml",
                 frame_skip=10, max_steps=2000,
                 render=False, realtime=False, seed: int = 0):
        if not os.path.exists(xml_path):
            raise FileNotFoundError(f"Cannot find {xml_path}")
        self.model = mujoco.MjModel.from_xml_path(xml_path)
        self.data = mujoco.MjData(self.model)
        self.frame_skip = frame_skip
        self.max_steps = max_steps
        self.render_enabled = render
        self.realtime = realtime
        self.dt = self.model.opt.timestep
        self.rng = np.random.default_rng(seed)

        self.last_throttle = 0.0
        self.step_count = 0
        self.hold_counter = 0
        self.hold_needed = 2

        self.body_id = mujoco.mj_name2id(self.model, mujoco.mjtObj.mjOBJ_BODY, "chassis")

        # ---------------- Target angle ----------------
        # φ definition:
        #   φ=0   -> upside-down
        #   |φ|=180 -> upright
        # NEW GOAL: |φ| = 90°
        self.target_deg = 90.0

        # Reward weights
        self.R = dict(
            position=2.0,
            momentum=4.0,   # scales signed momentum (toward target)
            stop_boost=0.00,
            energy=0.0,
            time=1.0,
            jerk=0.0,
            success=800.0
        )

        # Success hysteresis around target (in degrees ERROR)
        self.success_enter_err_deg   = 2.0   # start counting hold when | |phi|-target | <= 2°
        self.success_release_err_deg = 3.5   # reset hold only if error exceeds 3.5°
        self.max_upright_rate_deg = 300.0    # must be relatively still to count hold

        self.prev_phi_deg = 0.0
        self.last_rate_deg = 0.0
        self._flip_sign = 0.0  # +1 forward, -1 backward

        # Drive all four wheel actuators (fallback: drive all actuators)
        self.throttle_ids = []
        for name in ["front_left_motor", "front_right_motor", "rear_left_motor", "rear_right_motor"]:
            aid = mujoco.mj_name2id(self.model, mujoco.mjtObj.mjOBJ_ACTUATOR, name)
            if aid != -1:
                self.throttle_ids.append(aid)
        if not self.throttle_ids:
            self.throttle_ids = list(range(self.model.nu))

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
        self.window = glfw.create_window(1000, 800, "MonsterTruck Flip (Signed φ, Target=90°)", None, None)
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
        measured about the body Y axis (pitch axis):
            φ = sign( (z_w × r_w) · y_w ) * arccos(z_w · r_w) ∈ [-180,+180]
        """
        R = self.data.xmat[self.body_id].reshape(3, 3)  # body->world
        z_w = np.array([0.0, 0.0, 1.0], dtype=float)
        r_w = -R[:, 2]   # roof normal in world
        y_w =  R[:, 1]   # body Y in world (pitch axis)

        cosang = clip(float(np.dot(z_w, r_w)), -1.0, 1.0)
        theta = math.degrees(math.acos(cosang))   # unsigned [0,180]
        tri = float(np.dot(np.cross(z_w, r_w), y_w))  # signed area about y_w

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

        phi = self._get_signed_flip_deg()
        self.prev_phi_deg = phi
        self.last_rate_deg = 0.0
        return np.array([phi, 0.0], dtype=np.float32)

    def step(self, throttle):
        """Apply throttle ∈ [-1,1] to ALL wheel actuators for frame_skip steps."""
        throttle = float(np.clip(throttle, -1.0, 1.0))
        prev_throttle = self.last_throttle
        self.last_throttle = throttle
        done, success = False, False

        for _ in range(self.frame_skip):
            if self.model.nu > 0:
                self.data.ctrl[:self.model.nu] = 0.0
                for aid in self.throttle_ids:
                    self.data.ctrl[aid] = throttle
            mujoco.mj_step(self.model, self.data)
            if self.realtime:
                time.sleep(self.dt)

        # Signed angle and wrapped rate (DEGREES)
        phi_deg = self._get_signed_flip_deg()
        horizon_dt = max(self.frame_skip * self.dt, 1e-6)
        dphi_deg = angdiff_deg(phi_deg, self.prev_phi_deg)
        phi_rate_deg = dphi_deg / horizon_dt
        phi_rate_rad = math.radians(phi_rate_deg)

        # ---------------------------------------------------------
        # Target distance: goal is |phi| = target_deg (90°)
        # ---------------------------------------------------------
        phi_abs = abs(phi_deg)
        err_deg = abs(phi_abs - self.target_deg)          # 0 when at target
        err = err_deg / max(self.target_deg, 1e-6)        # normalized error

        err_clip2 = np.clip(err, 0.0, 2.0)
        err_clip1 = np.clip(err, 0.0, 1.0)

        # Position penalty: minimize error to target
        pos_penalty = self.R["position"] * (np.tanh(2.2 * err_clip2) ** 2)

        # Near-target gate (1 near target, 0 far)
        near_gate = np.exp(-12.0 * err_clip1)

        # ---------------------------------------------------------
        # Momentum reward: encourage moving TOWARD target
        # ---------------------------------------------------------
        # approx: d|phi|/dt ≈ sign(phi) * phi_rate
        sign_phi = 1.0 if phi_deg >= 0.0 else -1.0
        abs_rate_rad = sign_phi * phi_rate_rad    # >0 means |phi| increasing

        # If below target -> want |phi| to increase
        # If above target -> want |phi| to decrease
        dir_to_target = 1.0 if phi_abs < self.target_deg else -1.0

        # Stronger far away from target, weaker near target
        w_lin = np.clip(err_clip1, 0.0, 1.0)

        mom_reward = self.R["momentum"] * w_lin * (dir_to_target * abs_rate_rad)

        # Optional braking near target (still off by default unless stop_boost > 0)
        vel_brake  = -self.R["stop_boost"] * near_gate * (phi_rate_rad ** 2)

        energy_pen = self.R["energy"] * (throttle ** 2)
        time_pen   = self.R["time"]
        jerk_pen   = self.R["jerk"] * abs(throttle - prev_throttle)

        reward = -pos_penalty + mom_reward + vel_brake - energy_pen - time_pen - jerk_pen
        reward = clip(reward, -10.0, 10.0)

        # ---------------------------------------------------------
        # Success: hysteresis on error-to-target + velocity gate
        # ---------------------------------------------------------
        if (err_deg <= self.success_enter_err_deg) and (abs(phi_rate_deg) <= self.max_upright_rate_deg):
            self.hold_counter += 1
        elif err_deg > self.success_release_err_deg:
            self.hold_counter = 0

        if self.hold_counter >= self.hold_needed:
            reward += self.R["success"]
            success, done = True, True

        # TileQ-style render print (only when rendering is enabled)
        if self.render_enabled:
            ctrl_vals = np.round(self.data.ctrl[:self.model.nu], 3)
            print(
                f"[Render] step={self.step_count:4d} | "
                f"phi={phi_deg:+7.2f}° | |phi|={phi_abs:7.2f}° | err={err_deg:6.2f}° | "
                f"dφ/dt={phi_rate_deg:+8.2f}°/s | "
                f"u={throttle:+.3f} | ctrl={ctrl_vals} | hold={self.hold_counter}"
            )

        # Bookkeeping
        self.prev_phi_deg = phi_deg
        self.last_rate_deg = phi_rate_deg
        self.step_count += 1
        if self.step_count >= self.max_steps:
            done = True

        self._render()
        next_obs = np.array([phi_deg, phi_rate_deg], dtype=np.float32)
        return next_obs, float(reward), done, {"success": success}

    def close(self):
        if hasattr(self, "_viewer_ready") and self._viewer_ready:
            glfw.destroy_window(self.window)
            glfw.terminate()
            self._viewer_ready = False


# ===============================================================
# Minimal Tabular Q-learning Agent (discrete actions = 9)
# ===============================================================
class QAgent:
    def __init__(self,
                 bins_phi=121, bins_rate=97,
                 phi_range=(-180.0, 180.0), rate_range=(-720.0, 720.0),
                 n_actions=9,
                 alpha=0.05, gamma=0.98,
                 eps_start=0.99, eps_end=0.01, total_episodes=3000,
                 seed: int = 0):
        self.rng = np.random.default_rng(seed)

        self.bins_phi  = int(bins_phi)
        self.bins_rate = int(bins_rate)
        self.phi_range  = tuple(phi_range)
        self.rate_range = tuple(rate_range)

        self.n_actions = int(n_actions)
        self.actions = np.linspace(-1.0, 1.0, self.n_actions, dtype=np.float32)

        self.alpha = float(alpha)
        self.gamma = float(gamma)
        self.eps_start, self.eps_end = float(eps_start), float(eps_end)
        self.total_episodes = int(total_episodes)
        self.eps = self.eps_start
        self._eps_step = (self.eps_end - self.eps_start) / max(1, self.total_episodes)

        # Q-table
        self.q = np.zeros((self.bins_phi, self.bins_rate, self.n_actions), dtype=np.float32)

        # bin edges
        self.edges_phi  = np.linspace(self.phi_range[0],  self.phi_range[1],  self.bins_phi - 1)
        self.edges_rate = np.linspace(self.rate_range[0], self.rate_range[1], self.bins_rate - 1)

    def _disc(self, obs):
        phi, rate = float(obs[0]), float(obs[1])
        i_phi  = int(np.digitize(phi,  self.edges_phi))
        i_rate = int(np.digitize(rate, self.edges_rate))
        i_phi  = 0 if i_phi < 0 else (self.bins_phi  - 1 if i_phi  >= self.bins_phi  else i_phi)
        i_rate = 0 if i_rate < 0 else (self.bins_rate - 1 if i_rate >= self.bins_rate else i_rate)
        return i_phi, i_rate

    def act(self, obs):
        if self.rng.random() < self.eps:
            return int(self.rng.integers(self.n_actions))
        i_phi, i_rate = self._disc(obs)
        return int(np.argmax(self.q[i_phi, i_rate, :]))

    def act_greedy(self, obs):
        i_phi, i_rate = self._disc(obs)
        return int(np.argmax(self.q[i_phi, i_rate, :]))

    def learn(self, obs, a_idx, r, next_obs, done):
        i_phi, i_rate = self._disc(obs)
        i_phi_n, i_rate_n = self._disc(next_obs)
        q_sa = self.q[i_phi, i_rate, a_idx]
        target = r if done else (r + self.gamma * np.max(self.q[i_phi_n, i_rate_n, :]))
        self.q[i_phi, i_rate, a_idx] = q_sa + self.alpha * (target - q_sa)

    def decay_eps(self):
        self.eps = max(self.eps_end, self.eps + self._eps_step)


# ===============================================================
# Eval & plotting helpers
# ===============================================================
def evaluate_episode(env, agent, max_steps=1500, render=False):
    obs = env.reset()
    total = 0.0
    success_flag = 0
    env.render_enabled = render
    env.realtime = render
    for _ in range(max_steps):
        a_idx = agent.act_greedy(obs)
        throttle = float(agent.actions[a_idx])
        obs, r, done, info = env.step(throttle)
        total += r
        if done:
            success_flag = 1 if info.get("success", False) else 0
            break
    env.render_enabled = False
    env.realtime = False
    return total, success_flag

def run_greedy_trace(env, agent, max_steps=1500):
    obs = env.reset()
    t_steps, phi_deg_hist, phi_rate_hist, throttle_hist, rewards = [], [], [], [], []
    step = 0
    while step < max_steps:
        p  = float(obs[0])   # φ (deg, signed)
        pr = float(obs[1])   # φ rate (deg/s)
        a_idx = agent.act_greedy(obs)
        throttle = float(agent.actions[a_idx])

        t_steps.append(step)
        phi_deg_hist.append(p)
        phi_rate_hist.append(pr)
        throttle_hist.append(throttle)

        obs, r, done, info = env.step(throttle)
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

def save_episode_traces_figure(traces, outfile="qlearn_episode_summary.png"):
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
    ax1.set_title("Signed Flip Angle φ vs Timestep (0°=upside-down, ±90°=goal)")
    ax1.set_xlabel("Timestep"); ax1.set_ylabel("φ (deg)"); ax1.grid(True, alpha=0.3)

    ax2.plot(p, pr, lw=1.8)
    ax2.set_title("dφ/dt vs φ"); ax2.set_xlabel("φ (deg)"); ax2.set_ylabel("dφ/dt (deg/s)")
    ax2.grid(True, alpha=0.3)

    ax3.plot(t, thr, lw=1, alpha=0.25)
    ax3.plot(t, thr_s, lw=2)
    ax3.set_title("Throttle vs Timestep (smoothed)")
    ax3.set_xlabel("Timestep"); ax3.set_ylabel("Throttle"); ax3.grid(True, alpha=0.3)

    ax4.plot(t[:len(cre)], cre, lw=2)
    ax4.set_title("Cumulative Reward vs Timestep")
    ax4.set_xlabel("Timestep"); ax4.set_ylabel("Cumulative Reward"); ax4.grid(True, alpha=0.3)

    fig.suptitle("MonsterTruck Flip — Greedy Episode Summary (Tabular Q, Target=90°)", fontsize=14)
    fig.tight_layout(rect=[0, 0.03, 1, 0.96])
    fig.savefig(outfile, dpi=220)
    plt.close(fig)
    print(f"📊 Saved episode traces figure to {outfile}")

def save_eval_plots_and_csv(eval_ep_indices, eval_rewards):
    """Save (1) simple eps vs eval reward plot, (2) moving-avg plot, (3) CSV."""
    if len(eval_ep_indices) == 0:
        return
    # 1) Explicit eps vs evaluation reward
    plt.figure(figsize=(9, 5))
    plt.plot(eval_ep_indices, eval_rewards, marker='o', lw=2)
    plt.xlabel("Episode")
    plt.ylabel("Evaluation Reward")
    plt.title("Episodes vs Evaluation Reward")
    plt.grid(True, alpha=0.3)
    plt.tight_layout()
    plt.savefig("eval_eps_vs_reward.png", dpi=200)
    plt.close()
    print("📈 Saved eps-vs-eval-reward plot to eval_eps_vs_reward.png")

    # 2) Moving-avg variant
    plt.figure(figsize=(9, 5))
    plt.plot(eval_ep_indices, eval_rewards, 'o--', alpha=0.55, label='Eval reward')
    window = 5
    if len(eval_rewards) >= window:
        smoothed = np.convolve(eval_rewards, np.ones(window)/window, mode='valid')
        plt.plot(eval_ep_indices[window-1:], smoothed, '-', lw=2.2, label=f'{window}-pt Moving Avg')
    plt.xlabel("Episode"); plt.ylabel("Evaluation Reward")
    plt.title("Q-Learning — Signed φ with momentum toward target=90°")
    plt.legend(); plt.grid(True, alpha=0.3); plt.tight_layout()
    plt.savefig("qlearn_pitch_eval.png", dpi=200); plt.close()
    print("📈 Saved moving-average eval plot to qlearn_pitch_eval.png")

    # 3) CSV dump
    with open("eval_curve.csv", "w", newline="") as f:
        w = csv.writer(f)
        w.writerow(["episode", "eval_reward"])
        for e, r in zip(eval_ep_indices, eval_rewards):
            w.writerow([int(e), float(r)])
    print("💾 Saved evaluation curve to eval_curve.csv")


# ===============================================================
# Training (batch print every 10 episodes; render first eval rollout)
# ===============================================================
def train(episodes=3000, max_steps=1000, eval_every=100, seed: int = 42,
          bins_phi=121, bins_rate=97, alpha=0.05, gamma=0.98):

    np.random.seed(seed); random.seed(seed)

    env = MonsterTruckFlipEnvPitchSigned(render=False, realtime=False,
                                         frame_skip=10, max_steps=max_steps, seed=seed)
    print(env.dt)

    agent = QAgent(bins_phi=bins_phi, bins_rate=bins_rate,
                   alpha=alpha, gamma=gamma,
                   total_episodes=episodes, seed=seed)

    rewards_list, success_flags = [], []
    eval_ep_indices, eval_rewards = [], []
    successes = 0

    for ep in range(1, episodes + 1):
        obs = env.reset()
        ep_ret = 0.0
        success_flag = 0

        for _ in range(max_steps):
            a_idx = agent.act(obs)
            throttle = float(agent.actions[a_idx])
            next_obs, r, done, info = env.step(throttle)
            agent.learn(obs, a_idx, r, next_obs, done)
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

        # ---- Batch print: last-10 rewards at each multiple of 10 ----
        if ep % 10 == 0:
            last10 = rewards_list[-10:]
            succ10 = sum(success_flags[-10:])
            print(f"Ep {ep:4d} | eps {agent.eps:5.3f} | successes {successes}")
            print(f"   Last 10 rewards: {[round(r, 2) for r in last10]}  ({succ10}/10 success)")

        # ---- Evaluation block (render only the FIRST rollout) ----
        if eval_every and (ep % eval_every == 0):
            eval_rs, eval_ss = [], []
            for i in range(3):
                er, es = evaluate_episode(env, agent, max_steps=max_steps, render=(i == 0))
                eval_rs.append(er); eval_ss.append(es)
            mean_r, succ_rate = np.mean(eval_rs), np.mean(eval_ss) * 100
            eval_ep_indices.append(ep); eval_rewards.append(mean_r)
            print(f"[Eval @ Ep {ep}] avg_reward={mean_r:+9.3f} | success_rate={succ_rate:5.1f}% (rendered first)")

    # Save eval curve plots + CSV
    save_eval_plots_and_csv(eval_ep_indices, eval_rewards)

    # Greedy trace summary figure
    traces = run_greedy_trace(env, agent, max_steps)
    save_episode_traces_figure(traces, outfile="qlearn_episode_summary.png")

    env.close()
    return rewards_list, (eval_ep_indices, eval_rewards)


if __name__ == "__main__":
    rewards, eval_curve = train(
        episodes=1000, max_steps=1000, eval_every=40, seed=42,
        bins_phi=121, bins_rate=97, alpha=0.05, gamma=0.98
    )
    print("✅ Done. Saved eval plots (eps vs reward + moving average), CSV, and greedy episode summary.")
