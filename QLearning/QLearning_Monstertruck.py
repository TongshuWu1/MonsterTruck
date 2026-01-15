# ===============================================================
# QLearning_CarFlip_IMU.py
#
# Tabular Q-learning on state [phi_deg, phi_rate_deg] with 9 actions.
# - State from free-joint quaternion + IMU gyro (pitch axis)
# - Reward: signed-φ with linear, signed momentum term (MonsterTruck style)
# - Success: upright hysteresis + velocity gate
# - Eval: 3 eval rollouts every N episodes, all headless (no render)
# - Outputs: evaluation curves (plots + CSV) and a greedy episode summary figure
# ===============================================================

import os, math, time, random, csv
import numpy as np
import matplotlib.pyplot as plt
import mujoco
from mujoco.glfw import glfw

# All images / CSVs will go here
OUTPUT_DIR = "carflip_qlearn_output"


# ---------------- utils ----------------
def clip(v, lo, hi):
    return lo if v < lo else (hi if v > hi else v)


def angdiff_deg(curr, prev):
    """Minimal signed angle difference in degrees, result in [-180, 180)."""
    return ((curr - prev + 180.0) % 360.0) - 180.0


# ---------- quaternion & unwrap helpers ----------
def quat_to_R_and_pitch(qw, qx, qy, qz):
    """
    Convert unit quaternion (w, x, y, z) to rotation matrix R (body->world)
    and a standard Euler pitch (for debugging).
    """
    R00 = 1 - 2 * (qy * qy + qz * qz)
    R01 = 2 * (qx * qy - qw * qz)
    R02 = 2 * (qx * qz + qw * qy)

    R10 = 2 * (qx * qy + qw * qz)
    R11 = 1 - 2 * (qx * qx + qz * qz)
    R12 = 2 * (qy * qz - qw * qx)

    R20 = 2 * (qx * qz - qw * qy)
    R21 = 2 * (qy * qz + qw * qx)
    R22 = 1 - 2 * (qx * qx + qy * qy)

    pitch = -math.asin(max(-1.0, min(1.0, R20)))  # pitch in rad

    R = np.array(
        [[R00, R01, R02],
         [R10, R11, R12],
         [R20, R21, R22]],
        dtype=float,
    )
    return R, pitch


def unwrap_angle(prev_angle, prev_unwrapped, angle):
    """
    Incremental unwrap of an angle in [-pi, pi] so it becomes continuous.
    """
    if prev_angle is None:
        return angle, angle
    d = angle - prev_angle
    if d > math.pi:
        angle_unwrapped = prev_unwrapped + (d - 2 * math.pi)
    elif d < -math.pi:
        angle_unwrapped = prev_unwrapped + (d + 2 * math.pi)
    else:
        angle_unwrapped = prev_unwrapped + d
    return angle, angle_unwrapped


# ===============================================================
# Environment: IMU-based car flip with MonsterTruck-style reward
# ===============================================================
class CarFlipIMUEnv:
    def __init__(
        self,
        xml_path="monstertruck.xml",
        frame_skip=10,
        max_steps=2000,
        render=False,
        realtime=False,
        seed: int = 0,
    ):
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

        # Body
        self.body_id = mujoco.mj_name2id(
            self.model, mujoco.mjtObj.mjOBJ_BODY, "chassis"
        )

        # Free joint quaternion indices (chassis freejoint)
        free_j = next(
            j
            for j in range(self.model.njnt)
            if self.model.jnt_type[j] == mujoco.mjtJoint.mjJNT_FREE
        )
        # qpos layout for mjJNT_FREE: [x, y, z, qw, qx, qy, qz]
        self.qadr = self.model.jnt_qposadr[free_j] + 3

        # IMU sensors
        gyro_id = mujoco.mj_name2id(
            self.model, mujoco.mjtObj.mjOBJ_SENSOR, "imu_gyro"
        )
        if gyro_id < 0:
            raise RuntimeError("imu_gyro sensor not found in model XML")
        self.gyro_adr = self.model.sensor_adr[gyro_id]

        acc_id = mujoco.mj_name2id(
            self.model, mujoco.mjtObj.mjOBJ_SENSOR, "imu_acc"
        )
        if acc_id < 0:
            raise RuntimeError("imu_acc sensor not found in model XML")
        self.acc_adr = self.model.sensor_adr[acc_id]

        # Angle-unwrapping state
        self._prev_theta = None
        self._theta_unwrapped = 0.0
        self._theta0 = None

        self.prev_phi_deg = 0.0
        self.last_rate_deg = 0.0

        # Reward weights (MonsterTruck-style)
        self.R = dict(
            position=3.0,
            momentum=3.0,   # scales linear signed momentum
            stop_boost=0.00,
            energy=0.0,
            time=1.0,
            jerk=0.0,
            success=800.0,
        )

        # Success hysteresis + velocity gate
        # Relaxed a bit vs original to make success less brittle under noise
        self.success_enter_deg = 175.0    # start counting hold when |phi| >= 175°
        self.success_release_deg = 170.0  # reset if |phi| < 170°
        self.max_upright_rate_deg = 400.0

        # Wheel actuators (fallback: all actuators)
        self.throttle_ids = []
        for name in [
            "front_left_motor",
            "front_right_motor",
            "rear_left_motor",
            "rear_right_motor",
        ]:
            aid = mujoco.mj_name2id(self.model, mujoco.mjtObj.mjOBJ_ACTUATOR, name)
            if aid != -1:
                self.throttle_ids.append(aid)
        if not self.throttle_ids:
            self.throttle_ids = list(range(self.model.nu))

        # Viewer (available but disabled by default)
        self._viewer_ready = False
        if render:
            self._init_viewer()

    # ---------------- Rendering ----------------
    def _init_viewer(self):
        if self._viewer_ready:
            return
        if not glfw.init():
            raise RuntimeError("GLFW init failed")
        self.window = glfw.create_window(
            1000,
            800,
            "Car Flip (IMU, MonsterTruck-style Reward)",
            None,
            None,
        )
        glfw.make_context_current(self.window)
        glfw.swap_interval(1)
        self.cam = mujoco.MjvCamera()
        self.opt = mujoco.MjvOption()
        self.scene = mujoco.MjvScene(self.model, maxgeom=20000)
        self.context = mujoco.MjrContext(
            self.model, mujoco.mjtFontScale.mjFONTSCALE_150
        )
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
        mujoco.mjv_updateScene(
            self.model,
            self.data,
            self.opt,
            None,
            self.cam,
            mujoco.mjtCatBit.mjCAT_ALL,
            self.scene,
        )
        w, h = glfw.get_framebuffer_size(self.window)
        mujoco.mjr_render(mujoco.MjrRect(0, 0, w, h), self.scene, self.context)
        glfw.swap_buffers(self.window)
        glfw.poll_events()

    # ---------------- φ and φ̇ from quat + IMU gyro ----------------
    def _get_flip_state_from_imu(self):
        """
        Compute:
          - phi_deg: unwrapped flip angle in degrees
                     0°   ≈ initial upside-down pose
                     ±180° ≈ upright goal
          - phi_rate_deg: pitch rate from IMU (deg/s), around body Y
        """
        # Quaternion from free joint
        qw, qx, qy, qz = self.data.qpos[self.qadr: self.qadr + 4]
        R, _ = quat_to_R_and_pitch(qw, qx, qy, qz)

        # Up vector in world = body Z axis
        up_x, up_y, up_z = R[0, 2], R[1, 2], R[2, 2]

        # Angle of up vector in (z,x) plane: atan2(v_x, v_z) ∈ [-pi, pi]
        theta = math.atan2(up_x, up_z)

        # Unwrap over time
        self._prev_theta, self._theta_unwrapped = unwrap_angle(
            self._prev_theta, self._theta_unwrapped, theta
        )

        # Reference so we start near 0
        if self._theta0 is None:
            self._theta0 = self._theta_unwrapped

        flip_rel = self._theta_unwrapped - self._theta0
        phi_deg = math.degrees(flip_rel)

        # IMU gyro for pitch rate: gyro[1] is rotation around body Y (rad/s)
        gyro = self.data.sensordata[self.gyro_adr: self.gyro_adr + 3]
        pitch_rate_rad = float(gyro[1])
        phi_rate_deg = math.degrees(pitch_rate_rad)

        return phi_deg, phi_rate_deg

    # ---------------- Core API ----------------
    def reset(self):
        mujoco.mj_resetData(self.model, self.data)

        # Upside-down spawn
        self.data.qpos[:3] = np.array([0, 0, 0.3]) + 0.01 * self.rng.normal(size=3)
        self.data.qvel[:] = 0.0
        # 180° about X → upside-down (matches quat="0 1 0 0")
        self.data.qpos[3:7] = np.array([0, 1, 0, 0])
        mujoco.mj_forward(self.model, self.data)

        self.step_count = 0
        self.last_throttle = 0.0
        self.hold_counter = 0

        # Reset angle-unwrapping state
        self._prev_theta = None
        self._theta_unwrapped = 0.0
        self._theta0 = None

        phi_deg, phi_rate_deg = self._get_flip_state_from_imu()
        self.prev_phi_deg = phi_deg
        self.last_rate_deg = phi_rate_deg

        return np.array([phi_deg, phi_rate_deg], dtype=np.float32)

    def step(self, throttle):
        """Apply throttle ∈ [-1,1] to all wheel actuators for frame_skip steps."""
        throttle = float(np.clip(throttle, -1.0, 1.0))
        prev_throttle = self.last_throttle
        self.last_throttle = throttle
        done, success = False, False

        # Integrate dynamics and check success *inside* the frame-skip loop
        steps_taken = 0
        for _ in range(self.frame_skip):
            if self.model.nu > 0:
                self.data.ctrl[:self.model.nu] = 0.0
                for aid in self.throttle_ids:
                    self.data.ctrl[aid] = throttle
            mujoco.mj_step(self.model, self.data)
            steps_taken += 1

            # Check success based on instantaneous IMU φ and φ̇
            phi_deg_inst, phi_rate_deg_inst = self._get_flip_state_from_imu()
            phi_abs_inst = abs(phi_deg_inst)

            if (phi_abs_inst >= self.success_enter_deg) and (
                abs(phi_rate_deg_inst) <= self.max_upright_rate_deg
            ):
                self.hold_counter += 1
            elif phi_abs_inst < self.success_release_deg:
                self.hold_counter = 0

            if self.hold_counter >= self.hold_needed:
                success, done = True, True
                break

            if self.realtime:
                time.sleep(self.dt)

        # Signed flip angle and rate (degrees) from IMU / quaternion
        # Use horizon_dt based on actual steps_taken
        phi_deg_raw, _ = self._get_flip_state_from_imu()
        horizon_dt = max(steps_taken * self.dt, 1e-6)
        dphi_deg = angdiff_deg(phi_deg_raw, self.prev_phi_deg)
        phi_rate_deg = dphi_deg / horizon_dt
        phi_rate_rad = math.radians(phi_rate_deg)

        # Mild clamping for stability
        phi_deg = float(np.clip(phi_deg_raw, -270.0, 270.0))
        phi_rate_deg = float(np.clip(phi_rate_deg, -1000.0, 1000.0))

        # --------- MonsterTruck-style reward terms ----------

        # Forward-goal distance:
        #   φ = +180° → d_fwd = 0 (goal)
        #   φ = 0°    → d_fwd = 1 (upside-down)
        #   φ = -180° → d_fwd = 2 (backward upright, treated as far)
        d_fwd = (180.0 - phi_deg) / 180.0
        d_fwd_clip2 = np.clip(d_fwd, 0.0, 2.0)
        d_fwd_clip1 = np.clip(d_fwd, 0.0, 1.0)

        # 1) Position penalty (distance to forward upright)
        pos_penalty = self.R["position"] * (np.tanh(2.2 * d_fwd_clip2) ** 2)

        # 2) Linear, signed momentum (commit to whichever side you’re on)
        # weight: 1 at upside-down (|φ|=0), 0 at upright (|φ|=180)
        w_lin = np.clip((180.0 - abs(phi_deg)) / 180.0, 0.0, 1.0)
        # sign: >0 when motion increases |φ|, <0 when motion reduces |φ|
        sign_phi = 1.0 if phi_deg >= 0.0 else -1.0
        mom_reward = self.R["momentum"] * w_lin * (sign_phi * phi_rate_rad)

        # 3) Optional braking near upright (off by default since stop_boost=0)
        near_gate = np.exp(-12.0 * d_fwd_clip1)
        vel_brake = -self.R["stop_boost"] * near_gate * (phi_rate_rad ** 2)

        # 4) Energy / time / jerk penalties
        energy_pen = self.R["energy"] * (throttle ** 2)
        time_pen = self.R["time"]
        jerk_pen = self.R["jerk"] * abs(throttle - prev_throttle)

        reward = -pos_penalty + mom_reward + vel_brake - energy_pen - time_pen - jerk_pen
        reward = clip(reward, -10.0, 10.0)

        # Add success bonus after clipping, if we hit success in this step
        if success:
            reward += self.R["success"]

        # Timeout: end episode but no extra penalty
        self.step_count += 1
        if self.step_count >= self.max_steps:
            done = True

        # Optional debug print when rendering
        if self.render_enabled:
            ctrl_vals = np.round(self.data.ctrl[: self.model.nu], 3)
            print(
                f"[Render] step={self.step_count:4d} | "
                f"phi={phi_deg:+7.2f}° | dφ/dt={phi_rate_deg:+8.2f}°/s | "
                f"u={throttle:+.3f} | ctrl={ctrl_vals} | hold={self.hold_counter}"
            )

        self.prev_phi_deg = phi_deg
        self.last_rate_deg = phi_rate_deg

        self._render()
        next_obs = np.array([phi_deg, phi_rate_deg], dtype=np.float32)
        return next_obs, float(reward), done, {"success": success}

    def close(self):
        if hasattr(self, "_viewer_ready") and self._viewer_ready:
            glfw.destroy_window(self.window)
            glfw.terminate()
            self._viewer_ready = False


# ===============================================================
# Tabular Q-learning Agent (discrete actions)
# ===============================================================
class QAgent:
    def __init__(
        self,
        bins_phi=121,
        bins_rate=97,
        phi_range=(-180.0, 180.0),
        rate_range=(-720.0, 720.0),
        n_actions=9,
        alpha=0.05,
        gamma=0.98,
        eps_start=0.99,
        eps_end=0.05,
        total_episodes=3000,
        seed: int = 0,
    ):
        self.rng = np.random.default_rng(seed)

        self.bins_phi = int(bins_phi)
        self.bins_rate = int(bins_rate)
        self.phi_range = tuple(phi_range)
        self.rate_range = tuple(rate_range)

        self.n_actions = int(n_actions)
        self.actions = np.linspace(-1.0, 1.0, self.n_actions, dtype=np.float32)

        # RL hyperparameters
        self.base_alpha = float(alpha)  # base step size; will decay per-visit
        self.gamma = float(gamma)

        self.eps_start, self.eps_end = float(eps_start), float(eps_end)
        self.total_episodes = int(total_episodes)
        self.eps = self.eps_start
        # Exponential epsilon decay over episodes
        self.eps_decay = (self.eps_end / self.eps_start) ** (
            1.0 / max(1, self.total_episodes)
        )

        # Q-table
        self.q = np.zeros(
            (self.bins_phi, self.bins_rate, self.n_actions), dtype=np.float32
        )
        # Per-(s,a) visit counts for decaying learning rate
        self.visits = np.zeros_like(self.q, dtype=np.int32)

        # Bin edges
        self.edges_phi = np.linspace(
            self.phi_range[0], self.phi_range[1], self.bins_phi - 1
        )
        self.edges_rate = np.linspace(
            self.rate_range[0], self.rate_range[1], self.bins_rate - 1
        )

    def _disc(self, obs):
        phi, rate = float(obs[0]), float(obs[1])
        i_phi = int(np.digitize(phi, self.edges_phi))
        i_rate = int(np.digitize(rate, self.edges_rate))
        i_phi = 0 if i_phi < 0 else (self.bins_phi - 1 if i_phi >= self.bins_phi else i_phi)
        i_rate = (
            0
            if i_rate < 0
            else (self.bins_rate - 1 if i_rate >= self.bins_rate else i_rate)
        )
        return i_phi, i_rate

    def act(self, obs):
        # Near upright, turn off exploration so success gate isn't destroyed by noise
        phi = float(obs[0])
        if abs(phi) >= 170.0:
            i_phi, i_rate = self._disc(obs)
            return int(np.argmax(self.q[i_phi, i_rate, :]))

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

        # Update visit count for (s,a)
        self.visits[i_phi, i_rate, a_idx] += 1
        n_sa = self.visits[i_phi, i_rate, a_idx]
        # Decaying step size: α / sqrt(n)
        alpha_sa = self.base_alpha / math.sqrt(max(1, n_sa))

        q_sa = self.q[i_phi, i_rate, a_idx]
        target = r if done else (r + self.gamma * np.max(self.q[i_phi_n, i_rate_n, :]))
        self.q[i_phi, i_rate, a_idx] = q_sa + alpha_sa * (target - q_sa)

    def decay_eps(self):
        self.eps = max(self.eps_end, self.eps * self.eps_decay)


# ===============================================================
# Eval & plotting helpers
# ===============================================================
def evaluate_episode(env, agent, max_steps=1500, render=False):
    """
    Evaluate one greedy episode (always headless here).
    """
    obs = env.reset()
    total = 0.0
    success_flag = 0

    # Stay headless; ignore 'render' arg for training-time eval
    for _ in range(max_steps):
        a_idx = agent.act_greedy(obs)
        throttle = float(agent.actions[a_idx])
        obs, r, done, info = env.step(throttle)
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
        p = float(obs[0])
        pr = float(obs[1])
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
    os.makedirs(OUTPUT_DIR, exist_ok=True)
    outfile = os.path.join(OUTPUT_DIR, outfile)

    def ema(x, alpha=0.25):
        if len(x) == 0:
            return x
        y = np.empty_like(x, dtype=float)
        y[0] = x[0]
        for i in range(1, len(x)):
            y[i] = alpha * x[i] + (1 - alpha) * y[i - 1]
        return y

    t = traces["t"]
    p = traces["phi_deg"]
    pr = traces["phi_rate_deg"]
    thr = traces["throttle"]
    cre = traces["cum_reward"]
    thr_s = ema(thr, alpha=0.3)

    fig, axs = plt.subplots(2, 2, figsize=(12, 9))
    (ax1, ax2), (ax3, ax4) = axs

    ax1.plot(t, p, lw=2)
    ax1.set_title("Signed Flip Angle φ vs Timestep (0°=upside-down, ±180°=goal)")
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

    ax4.plot(t[: len(cre)], cre, lw=2)
    ax4.set_title("Cumulative Reward vs Timestep")
    ax4.set_xlabel("Timestep")
    ax4.set_ylabel("Cumulative Reward")
    ax4.grid(True, alpha=0.3)

    fig.suptitle(
        "Car Flip — Greedy Episode Summary (Tabular Q, IMU, Lin-Mom Reward)",
        fontsize=14,
    )
    fig.tight_layout(rect=[0, 0.03, 1, 0.96])
    fig.savefig(outfile, dpi=220)
    plt.close(fig)
    print(f"Saved episode traces figure to {outfile}")


def save_eval_plots_and_csv(eval_ep_indices, eval_rewards):
    """Save (1) eps vs eval reward, (2) moving-avg plot, (3) CSV, all in OUTPUT_DIR."""
    if len(eval_ep_indices) == 0:
        return

    os.makedirs(OUTPUT_DIR, exist_ok=True)

    # 1) Explicit eps vs evaluation reward
    path_eps_vs_reward = os.path.join(OUTPUT_DIR, "eval_eps_vs_reward.png")
    plt.figure(figsize=(9, 5))
    plt.plot(eval_ep_indices, eval_rewards, marker="o", lw=2)
    plt.xlabel("Episode")
    plt.ylabel("Evaluation Reward")
    plt.title("Episodes vs Evaluation Reward")
    plt.grid(True, alpha=0.3)
    plt.tight_layout()
    plt.savefig(path_eps_vs_reward, dpi=200)
    plt.close()
    print(f"Saved eps-vs-eval-reward plot to {path_eps_vs_reward}")

    # 2) Moving-avg variant
    path_moving_avg = os.path.join(OUTPUT_DIR, "qlearn_pitch_eval.png")
    plt.figure(figsize=(9, 5))
    plt.plot(eval_ep_indices, eval_rewards, "o--", alpha=0.55, label="Eval reward")
    window = 5
    if len(eval_rewards) >= window:
        smoothed = np.convolve(eval_rewards, np.ones(window) / window, mode="valid")
        plt.plot(
            eval_ep_indices[window - 1:],
            smoothed,
            "-",
            lw=2.2,
            label=f"{window}-pt Moving Avg",
        )
    plt.xlabel("Episode")
    plt.ylabel("Evaluation Reward")
    plt.title("Q-Learning — IMU Signed φ with Linear Momentum Reward")
    plt.legend()
    plt.grid(True, alpha=0.3)
    plt.tight_layout()
    plt.savefig(path_moving_avg, dpi=200)
    plt.close()
    print(f"Saved moving-average eval plot to {path_moving_avg}")

    # 3) CSV dump
    csv_path = os.path.join(OUTPUT_DIR, "eval_curve.csv")
    with open(csv_path, "w", newline="") as f:
        w = csv.writer(f)
        w.writerow(["episode", "eval_reward"])
        for e, r in zip(eval_ep_indices, eval_rewards):
            w.writerow([int(e), float(r)])
    print(f"Saved evaluation curve to {csv_path}")


# ===============================================================
# Training
# ===============================================================
def train(
    episodes=3000,
    max_steps=1500,
    eval_every=10,      # 3 eval rollouts every 10 episodes
    seed: int = 42,
    bins_phi=121,
    bins_rate=97,
    alpha=0.05,
    gamma=0.98,
):
    np.random.seed(seed)
    random.seed(seed)

    env = CarFlipIMUEnv(
        render=False,
        realtime=False,
        frame_skip=10,
        max_steps=max_steps,
        seed=seed,
    )

    agent = QAgent(
        bins_phi=bins_phi,
        bins_rate=bins_rate,
        alpha=alpha,
        gamma=gamma,
        total_episodes=episodes,
        seed=seed,
    )

    rewards_list, success_flags = [], []
    eval_ep_indices, eval_rewards = [], []
    eps_history = []
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
        eps_history.append(agent.eps)

        # Batch print: last-10 rewards at each multiple of 10
        if ep % 10 == 0:
            last10 = rewards_list[-10:]
            succ10 = sum(success_flags[-10:])
            print(f"Ep {ep:4d} | eps {agent.eps:5.3f} | successes {successes}")
            print(
                f"   Last 10 rewards: {[round(r, 2) for r in last10]}  "
                f"({succ10}/10 success)"
            )

        # Evaluation block: 3 eval rollouts (headless)
        if eval_every and (ep % eval_every == 0):
            eval_rs, eval_ss = [], []
            for _ in range(3):
                er, es = evaluate_episode(env, agent, max_steps=max_steps, render=False)
                eval_rs.append(er)
                eval_ss.append(es)
            mean_r = np.mean(eval_rs)
            succ_rate = np.mean(eval_ss) * 100
            eval_ep_indices.append(ep)
            eval_rewards.append(mean_r)
            print(
                f"[Eval @ Ep {ep}] avg_reward={mean_r:+9.3f} | "
                f"success_rate={succ_rate:5.1f}% (3 headless rollouts)"
            )

    # Save training curve CSV + plot
    os.makedirs(OUTPUT_DIR, exist_ok=True)
    train_csv = os.path.join(OUTPUT_DIR, "train_curve.csv")
    with open(train_csv, "w", newline="") as f:
        w = csv.writer(f)
        w.writerow(["episode", "train_return", "success", "epsilon"])
        for ep_idx, (ret, succ, eps) in enumerate(
            zip(rewards_list, success_flags, eps_history), start=1
        ):
            w.writerow([ep_idx, float(ret), int(succ), float(eps)])
    print(f"Saved training curve to {train_csv}")

    # Training return plot
    train_png = os.path.join(OUTPUT_DIR, "train_return.png")
    plt.figure(figsize=(9, 5))
    plt.plot(range(1, len(rewards_list) + 1), rewards_list, alpha=0.4, label="Episode return")
    window = 20
    if len(rewards_list) >= window:
        ma = np.convolve(rewards_list, np.ones(window) / window, mode="valid")
        plt.plot(
            range(window, len(rewards_list) + 1),
            ma,
            lw=2,
            label=f"{window}-ep moving avg",
        )
    plt.xlabel("Episode")
    plt.ylabel("Training return")
    plt.title("Training Return vs Episode (CarFlip IMU, Lin-Mom Reward)")
    plt.grid(True, alpha=0.3)
    plt.legend()
    plt.tight_layout()
    plt.savefig(train_png, dpi=200)
    plt.close()
    print(f"Saved training return plot to {train_png}")

    # Save eval curve plots + CSV
    save_eval_plots_and_csv(eval_ep_indices, eval_rewards)

    # Greedy trace summary figure
    traces = run_greedy_trace(env, agent, max_steps)
    save_episode_traces_figure(traces, outfile="qlearn_episode_summary.png")

    env.close()
    return rewards_list, (eval_ep_indices, eval_rewards)


if __name__ == "__main__":
    rewards, eval_curve = train(
        episodes=1000,
        max_steps=1000,
        eval_every=20,   # 3 evals every 20 episodes
        seed=42,
        bins_phi=121,
        bins_rate=97,
        alpha=0.05,
        gamma=0.98,
    )
    print(
        "Done. Saved eval plots, training plots/CSVs, and greedy episode summary in folder:",
        OUTPUT_DIR,
    )
