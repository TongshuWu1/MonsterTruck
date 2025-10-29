# ===============================================================
# TileQLearning_Monstertruck.py
# Single-stage: axis-correct roll (0..360°) + local ωx
# Directional progress shaping toward 0°, no payoff for pure spinning
# ===============================================================

import os, math, time
import numpy as np
import matplotlib.pyplot as plt
import mujoco
from mujoco.glfw import glfw


# ---------------- Axis helpers (BODY-FRAME CONSISTENT) ----------------
def roll_rad_body_x(model, data, body_id):
    """
    Roll (about BODY X) in [-pi, pi], independent of yaw/pitch.
    Method: express world-up in BODY frame, then roll = atan2(-u_y, u_z).
    0 => upright, ±pi => upside-down.
    """
    R = data.xmat[body_id].reshape(3, 3)   # body->world
    up_world = np.array([0.0, 0.0, 1.0])
    up_body = R.T @ up_world               # world->body
    up_body /= (np.linalg.norm(up_body) + 1e-12)
    return math.atan2(-up_body[1], up_body[2])  # [-pi, pi]


def roll_deg_0_360(model, data, body_id):
    """
    Map roll (about BODY X) to [0, 360) with:
      0°   = upright (goal)
      180° = upside-down
      360° = upright (wrong-way)
    """
    r = roll_rad_body_x(model, data, body_id)          # [-pi, pi]
    r = (r + 2.0 * math.pi) % (2.0 * math.pi)          # [0, 2pi)
    return math.degrees(r)


def roll_rate_local_x(model, data, body_id):
    """
    True roll rate about local X using MuJoCo's velocity API.
    Returns wx (rad/s) in the BODY frame.
    """
    try:
        res = np.zeros(6, dtype=float)
        mujoco.mj_objectVelocity(model, data, mujoco.mjtObj.mjOBJ_BODY, body_id, res, 1)
        return float(res[3])  # BODY-frame wx (rad/s)
    except Exception:
        # Fallback: project WORLD angular velocity onto BODY x-axis in world coords
        ang_world = np.array(data.cvel[body_id, :3], dtype=float)  # ω_world
        R = data.xmat[body_id].reshape(3, 3)                       # body->world
        x_world = R[:, 0]
        return float(np.dot(ang_world, x_world))


# ===============================================================
# MonsterTruck Flip Environment
# ===============================================================
class MonsterTruckFlipEnvYPR:
    def __init__(self, xml_path="monstertruck.xml",
                 frame_skip=6, max_steps=1500,
                 render=False, realtime=False,
                 num_actions=5):
        if not os.path.exists(xml_path):
            raise FileNotFoundError(f"Cannot find {xml_path}")
        self.model = mujoco.MjModel.from_xml_path(xml_path)
        self.data = mujoco.MjData(self.model)
        self.frame_skip = frame_skip
        self.max_steps = max_steps
        self.render_enabled = render
        self.realtime = realtime
        self.dt = self.model.opt.timestep

        # Fixed action set (commands in [-1, 1])
        self.actions = np.linspace(-1.0, 1.0, int(num_actions)).astype(np.float32)
        self.last_cmd = 0.0
        self.step_count = 0
        self.body_id = mujoco.mj_name2id(self.model, mujoco.mjtObj.mjOBJ_BODY, "chassis")

        # Reward weights (single env.R only)
        self.R = dict(
            vel_progress=20.0,     # directional velocity progress toward 0°
            angle_quad=0.5,       # quadratic distance penalty to 0° (wrap-aware)
            energy=0.01,          # (throttle)^2
            jerk=0.02,            # (Δthrottle)^2
            time=0.03,            # per-step
            success=900.0,        # terminal bonus near 0°
            wrongway=1200.0       # terminal penalty near 360° (if settled)
        )

        # Success tolerances (slightly looser to allow settling)
        self.success_angle_tol = 5.0     # deg around 0°
        self.success_rate_tol  = 0.2     # rad/s (local ωx)  <-- relaxed from 0.05

        # Wrong-way penalty near 360°
        self.enable_wrongway_penalty = True

        # Rear motor indices
        self._rear_idx = self._find_rear_actuators()

        # Track best-so-far normalized distance to 0° (for optional progress bonus if desired)
        self._best_goal_dist = 1.0

        # Velocity clip for reward (deg/s) to avoid runaway spinning bonuses
        self._vel_deg_clip = 720.0

        self._viewer_ready = False
        if render:
            self._init_viewer()

    def _find_rear_actuators(self):
        """Find indices of rear motors by name; fallback to first two actuators."""
        names = ["rear_left_motor", "rear_right_motor"]
        idxs = []
        for nm in names:
            try:
                a_id = mujoco.mj_name2id(self.model, mujoco.mjtObj.mjOBJ_ACTUATOR, nm)
                idxs.append(a_id)
            except Exception:
                pass
        if len(idxs) < 2:
            idxs = [i for i in range(min(2, self.model.nu))]
        return idxs

    # ---------------------- Rendering ----------------------
    def _init_viewer(self):
        if self._viewer_ready:
            return
        if not glfw.init():
            raise RuntimeError("GLFW init failed")
        self.window = glfw.create_window(1000, 800, "MonsterTruck Flip", None, None)
        glfw.make_context_current(self.window)
        glfw.swap_interval(1)  # vsync

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

    # ---------------------- Core API ----------------------
    def reset(self):
        mujoco.mj_resetData(self.model, self.data)
        # Start pose: exactly upside-down, no noise
        self.data.qpos[:3] = np.array([0.0, 0.0, 0.2])            # height ~0.2
        self.data.qvel[:] = 0.0
        self.data.qpos[3:7] = np.array([0.0, 1.0, 0.0, 0.0])      # 180° about x
        mujoco.mj_forward(self.model, self.data)

        self.step_count = 0
        self.last_cmd = 0.0

        roll_deg = self._get_roll_deg_360()
        self._best_goal_dist = self._norm_dist_to_goal(roll_deg)

        return self._get_obs()

    def _get_roll_rad(self):
        return roll_rad_body_x(self.model, self.data, self.body_id)

    def _get_roll_deg_360(self):
        return roll_deg_0_360(self.model, self.data, self.body_id)

    def _get_roll_rate_local_x(self):
        return roll_rate_local_x(self.model, self.data, self.body_id)

    def _get_obs(self):
        roll = self._get_roll_rad()
        roll_rate = self._get_roll_rate_local_x()
        return np.array([roll, roll_rate, self.last_cmd], dtype=np.float32)

    @staticmethod
    def _norm_dist_to_goal(roll_deg):
        """
        Wrap-aware normalized distance to 0°:
          0 when exactly at 0°, 1 at 180°, linear in between.
        """
        return min(roll_deg, 360.0 - roll_deg) / 180.0  # ∈ [0,1]

    @staticmethod
    def _signed_deg_error_to_zero(roll_deg):
        """
        Signed shortest error (deg) to 0°, in (-180, 180].
        Positive means angle > 0° in the short path sense; negative if "past" 0°.
        """
        e = (roll_deg + 180.0) % 360.0 - 180.0
        if e == -180.0:
            e = 180.0
        return e

    def step(self, a_idx):
        if not (0 <= a_idx < len(self.actions)):
            raise ValueError(f"a_idx {a_idx} outside [0, {len(self.actions)-1}]")

        cmd = float(self.actions[a_idx])   # ∈ [-1, 1]
        throttle = cmd
        done, success = False, False

        # Drive the two rear motors (direct command)
        for _ in range(self.frame_skip):
            for i in self._rear_idx:
                self.data.ctrl[i] = throttle
            mujoco.mj_step(self.model, self.data)
            if self.realtime:
                time.sleep(self.dt)

        # -------- Angle & rate --------
        roll_deg = self._get_roll_deg_360()
        roll_rate_rad = self._get_roll_rate_local_x()
        roll_rate_deg = np.degrees(roll_rate_rad)

        # -------- Reward terms --------
        # Angle scale: 1 at 180°, 0 at 0°/360° (fades velocity payoff near upright)
        angle_scale = 1.0 - abs(180.0 - roll_deg) / 180.0
        angle_scale = float(np.clip(angle_scale, 0.0, 1.0))

        # Signed shortest error to 0° (deg)
        e_deg = self._signed_deg_error_to_zero(roll_deg)

        # Directional velocity progress: positive iff |e| is shrinking
        # d|e|/dt ≈ sign(e) * de/dt, with de/dt = roll_rate_deg
        # So - sign(e) * roll_rate_deg is positive when error magnitude decreases.
        # Fade out near upright via angle_scale to discourage chatter.
        roll_rate_deg_clipped = float(np.clip(roll_rate_deg, -self._vel_deg_clip, self._vel_deg_clip))
        dir_progress = -np.sign(e_deg) * roll_rate_deg_clipped
        R_velprog = self.R["vel_progress"] * angle_scale * dir_progress / self._vel_deg_clip  # normalize

        # Quadratic distance penalty (wrap-aware), stronger than your old linear
        # Normalize by 180 so it's O(1)
        R_angle = -self.R["angle_quad"] * (e_deg / 180.0) ** 2

        # Jerk / energy / time
        prev_throttle = self.last_cmd
        jerk = throttle - prev_throttle
        R_jerk = -self.R["jerk"] * (jerk ** 2)
        R_energy = -self.R["energy"] * (throttle ** 2)
        R_time = -self.R["time"]

        reward = R_velprog + R_angle + R_jerk + R_energy + R_time

        # -------- Terminal conditions --------
        near0    = (roll_deg < self.success_angle_tol)
        near360  = abs(roll_deg - 360.0) < self.success_angle_tol

        # SUCCESS: strictly near 0° (not 360°)
        if near0 and (abs(roll_rate_rad) < self.success_rate_tol):
            reward += self.R["success"]
            success, done = True, True
        # WRONG-WAY: strictly near 360°
        elif self.enable_wrongway_penalty and near360 and (abs(roll_rate_rad) < self.success_rate_tol):
            reward -= self.R["wrongway"]
            done = True

        self.last_cmd = cmd
        self.step_count += 1
        if self.step_count >= self.max_steps:
            done = True

        self._render()
        return self._get_obs(), reward, done, {"success": success}

    def close(self):
        if self._viewer_ready:
            try:
                mujoco.mjr_freeContext(self.context)
            except Exception:
                pass
            try:
                mujoco.mjv_freeScene(self.scene)
            except Exception:
                pass
            try:
                glfw.destroy_window(self.window)
                glfw.terminate()
            except Exception:
                pass
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
        # diagnostics
        self._delta_accum = 0.0
        self._delta_count = 0

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
        idxs = self.tc.encode(obs)
        q_sa = float(np.sum(self.w[a, idxs]))
        target = r + (0.0 if done else self.gamma * float(np.max(self.Qs(next_obs))))
        delta = target - q_sa
        for idx in idxs:
            self.w[a, idx] += self.alpha * delta
        # diagnostics
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
            success_flag = 1 if info.get("success", False) else 0
            break
    return total, success_flag


def train_tileq(episodes=2000, render_every=100, max_steps=1500, eval_every=50,
                num_actions=9):
    env = MonsterTruckFlipEnvYPR(render=False, realtime=False, frame_skip=6,
                                 max_steps=max_steps, num_actions=num_actions)

    # Observations: roll in [-pi, pi], roll_rate in rad/s, last_cmd in [-1,1]
    lows, highs = [-math.pi, -20.0, -1.0], [math.pi, 20.0, 1.0]
    agent = TileQAgent(lows, highs, n_tiles=(18, 12, 6), n_tilings=8,
                       n_actions=len(env.actions), total_episodes=episodes)

    rewards_list, success_flags, eval_ep_indices, eval_rewards = [], [], [], []
    successes = 0

    for ep in range(1, episodes + 1):
        render = (render_every and ep % render_every == 0)
        env.render_enabled = render
        env.realtime = render

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

        if render:
            env.render_enabled = False
            env.realtime = False

        rewards_list.append(ep_ret)
        success_flags.append(success_flag)
        agent.decay_eps()

        if ep % 10 == 0:
            last10 = rewards_list[-10:]
            avgupd = agent.avg_update()
            print(f"Ep {ep:4d} | eps {agent.eps:5.3f} | successes {successes} | ⟨|Δ|⟩ {avgupd:.4f}")
            print(f"   Last 10 rewards: {[round(r, 2) for r in last10]}  "
                  f"({sum(success_flags[-10:])}/10 success)")

        if eval_every and (ep % eval_every == 0):
            eval_rs, eval_ss = [], []
            for _ in range(5):
                er, es = evaluate_episode(env, agent, env.max_steps)
                eval_rs.append(er)
                eval_ss.append(es)
            mean_r, succ_rate = np.mean(eval_rs), np.mean(eval_ss) * 100
            eval_ep_indices.append(ep)
            eval_rewards.append(mean_r)
            print(f"   [Eval @ Ep {ep}] avg_reward={mean_r:.1f} | success_rate={succ_rate:.1f}%")

    env.close()

    if len(eval_ep_indices) > 0:
        try:
            plt.style.use('seaborn-v0_8-bright')
        except Exception:
            pass
        window = 5
        smoothed = np.convolve(eval_rewards, np.ones(window)/window, mode='valid') if len(eval_rewards) >= window else []
        plt.figure(figsize=(9, 5))
        plt.plot(eval_ep_indices, eval_rewards, 'o--', alpha=0.45, label='Raw eval reward')
        if len(smoothed) > 0:
            plt.plot(eval_ep_indices[window-1:], smoothed, 'r-', lw=2.2, label=f'{window}-pt Moving Avg')
        plt.xlabel("Episode")
        plt.ylabel("Evaluation Reward")
        plt.title("TileQ — Directional progress to 0° (axis-correct roll & local ωx)")
        plt.legend()
        plt.grid(True, alpha=0.3)
        plt.tight_layout()
        plt.savefig("tileq_eval_rewards_single_stage.png", dpi=200)
        plt.close()
        print("📈 Saved eval plot to tileq_eval_rewards_single_stage.png")

    return rewards_list, (eval_ep_indices, eval_rewards)


if __name__ == "__main__":
    # For stronger impulses early on, try num_actions=3 ([-1,0,1]).
    train_tileq(episodes=4000, render_every=4000, max_steps=1500, eval_every=50,
                num_actions=9)


