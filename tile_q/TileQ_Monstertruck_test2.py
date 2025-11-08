# ===============================================================
# MonsterTruck TileQ (Continuous) with Rare-Contact Exploration
# - State:   s = [φ (deg, signed), φ_rate (deg/s)]
# - Action:  a = throttle ∈ [-1, 1]
# - Tiles:   3D over [φ, φ_rate, a]  (NO previous-throttle in tiles)
# - Reward:  EXACTLY your original design (no changes)
# - Extras:  UCB optimism, Archive reset curriculum, Pulse actions
#            Optional eval-guided mentor replay (Dyna-style)
#            Policy-side "commit-to-flip" prior (action inertia + directional sign)
# ===============================================================

import os, math, time, random
import numpy as np
import matplotlib.pyplot as plt
import mujoco
from mujoco.glfw import glfw


def clip(v, lo, hi):
    return lo if v < lo else (hi if v > hi else v)

def angdiff_deg(curr, prev):
    """Minimal signed angle difference in degrees, result in [-180, 180)."""
    return ((curr - prev + 180.0) % 360.0) - 180.0


# ===============================================================
# Environment (Signed φ + forward-goal distance) — reward unchanged
# ===============================================================
class MonsterTruckFlipEnvPitchSigned:
    def __init__(self, xml_path="monstertruck.xml",
                 frame_skip=10, max_steps=2000,
                 render=False, realtime=False,
                 seed: int = 0):
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

        self.last_throttle = 0.0
        self.step_count = 0
        self.hold_counter = 0
        self.hold_needed = 4

        self.body_id = mujoco.mj_name2id(self.model, mujoco.mjtObj.mjOBJ_BODY, "chassis")

        # Reward weights (UNCHANGED)
        self.R = dict(
            position=3.50,     # directional distance penalty via d_fwd
            momentum=2.0,      # MountainCar-style momentum reward (|ω| when far)
            stop_boost=0.0,    # near-upright brake on ω^2
            energy=0.15,        # control effort penalty
            time=1.0,          # per-step time cost
            jerk=0.3,          # |Δu|
            success=2000.0     # terminal bonus
        )

        # State memory
        self.prev_phi_deg = 0.0
        self.last_rate_deg = 0.0
        self._flip_sign = 0.0  # +1 forward, -1 backward; resolved after slight motion

        # -------- Rare-contact curriculum knobs --------
        self.archive = []              # list of (qpos, qvel) near-contact snapshots
        self.archive_max = 64
        self.near_contact_eps = 0.04   # meters above ground counts as "near contact"
        self.reset_mix_prob = 0.25     # chance to start from archive snapshot
        self._wheel_gids = None

        # -------- Optional potential shaping (OFF by default) ----
        self._use_potential = False
        self.gamma_train = 0.98
        self._prev_min_height = 0.0  # used only when potential shaping is on

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

    # ---------------- Wheel helpers (for curriculum / optional shaping) -----
    def _find_wheel_geom_ids(self):
        if self._wheel_gids is not None:
            return self._wheel_gids
        gids = []
        for gid in range(self.model.ngeom):
            name = mujoco.mj_id2name(self.model, mujoco.mjtObj.mjOBJ_GEOM, gid) or ""
            if any(k in name.lower() for k in ["wheel", "tire", "tyre"]):
                gids.append(gid)
        self._wheel_gids = gids
        return gids

    def _min_wheel_height(self):
        gids = self._find_wheel_geom_ids()
        if not gids:
            # fallback proxy: chassis height
            return float(self.data.xpos[self.body_id, 2])
        zvals = [self.data.geom_xpos[g, 2] for g in gids]
        return float(np.min(zvals))

    def _snapshot(self):
        return (self.data.qpos.copy(), self.data.qvel.copy())

    def _restore(self, snap):
        qpos, qvel = snap
        self.data.qpos[:] = qpos
        self.data.qvel[:] = qvel
        mujoco.mj_forward(self.model, self.data)

    # ---------------- Core API ----------------
    def reset(self):
        mujoco.mj_resetData(self.model, self.data)

        # Archive-mixed resets: sometimes start near-contact to actually SEE success
        if self.archive and (self.rng.random() < self.reset_mix_prob):
            # FIX: choose by index (archive holds tuples with arrays; ragged for np.choice)
            i = int(self.rng.integers(len(self.archive)))
            snap = self.archive[i]
            self._restore(snap)
        else:
            # Upside-down spawn
            self.data.qpos[:3] = np.array([0, 0, 0.3]) + 0.01 * self.rng.normal(size=3)
            self.data.qvel[:] = 0.0
            self.data.qpos[3:7] = np.array([0, 1, 0, 0])  # 180° about X → upside-down
            mujoco.mj_forward(self.model, self.data)

        self.step_count = 0
        self.last_throttle = 0.0
        self.hold_counter = 0
        self._flip_sign = 0.0

        phi = self._get_signed_flip_deg()   # ~0° at start (upside-down)
        self.prev_phi_deg = phi
        self.last_rate_deg = 0.0

        # init potential (training-only feature)
        self._prev_min_height = self._min_wheel_height()

        # obs = [phi_deg, phi_rate_deg]
        return np.array([phi, 0.0], dtype=np.float32)

    def step(self, throttle):
        """Continuous throttle in [-1, 1]."""
        throttle = float(np.clip(throttle, -1.0, 1.0))
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
        dphi_deg = angdiff_deg(phi_deg, self.prev_phi_deg)     # wrap-safe
        phi_rate_deg = dphi_deg / horizon_dt
        phi_rate_rad = math.radians(phi_rate_deg)

        if self.render_enabled:
            print(
                f"[Render] step={self.step_count:4d} | "
                f"phi={phi_deg:+7.2f}° | dφ/dt={phi_rate_deg:+8.2f}°/s | u={throttle:+.3f}"
            )

        # --------- FORWARD-GOAL DISTANCE (directional) ----------
        d_fwd = (180.0 - phi_deg) / 180.0           # in [0,2]; >1 when φ < 0 (backward side)
        d_fwd_clip2 = np.clip(d_fwd, 0.0, 2.0)      # for penalty shaping
        d_fwd_clip1 = np.clip(d_fwd, 0.0, 1.0)      # for near/far gates

        # Position penalty (smooth, saturating; larger if φ goes backward)
        pos_penalty = self.R["position"] * (np.tanh(2.2 * d_fwd_clip2) ** 2)

        # Gates by forward distance (not symmetric)
        near_gate = np.exp(-12.0 * d_fwd_clip1)        # ~1 near goal (+180), ~0 far
        far_gate  = 1.0 - np.exp(-8.0  * d_fwd_clip1)  # ~1 far from goal

        # MountainCar-style momentum: reward speed in EITHER direction when far
        mom_reward = self.R["momentum"] * far_gate * abs(phi_rate_rad)

        # Near-upright braking to prevent overshoot (kept; weight may be 0)
        vel_brake = -self.R["stop_boost"] * near_gate * (phi_rate_rad ** 2)

        energy_pen = self.R["energy"] * (throttle ** 2)
        time_pen   = self.R["time"]
        jerk_pen   = self.R["jerk"] * abs(throttle - prev_throttle)

        reward_base = -pos_penalty + mom_reward + vel_brake - energy_pen - time_pen - jerk_pen
        reward_base = clip(reward_base, -10.0, 10.0)

        # Success: (kept exactly as before)
        if (abs(phi_deg) > 178.0):
            self.hold_counter += 1
            if self.hold_counter >= self.hold_needed:
                reward_base += self.R["success"]
                success, done = True, True
        else:
            self.hold_counter = 0

        # Archive promising "near-contact" states (to make rare contact discoverable)
        if not success:
            if self._min_wheel_height() < self.near_contact_eps:
                snap = self._snapshot()
                if len(self.archive) < self.archive_max:
                    self.archive.append(snap)
                else:
                    i = int(self.rng.integers(self.archive_max))
                    self.archive[i] = snap

        # Optional potential shaping (OFF by default) — preserves optimality
        reward = reward_base
        if self._use_potential:
            h_now = self._min_wheel_height()
            phi_prev = -self._prev_min_height
            phi_now  = -h_now
            reward += self.gamma_train * phi_now - phi_prev
            self._prev_min_height = h_now

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
            "phi_rate_deg": phi_rate_deg,
            "reward_base": reward_base
        }
        return next_obs, float(reward), done, info

    def close(self):
        if self._viewer_ready:
            glfw.destroy_window(self.window)
            glfw.terminate()
            self._viewer_ready = False


# ===============================================================
# Tile coder shell over 3D [φ, φ_rate, a] (offsets & shapes)
# ===============================================================
class TileCoder:
    def __init__(self, lows, highs, n_tiles, n_tilings, seed: int = 0):
        self.lows = np.array(lows, dtype=np.float32)
        self.highs = np.array(highs, dtype=np.float32)
        self.n_tiles = np.array(n_tiles, dtype=np.int32)   # e.g., (48, 48, 21)
        self.n_tilings = int(n_tilings)
        self.dim = len(lows)
        assert self.dim == 3, "Expecting 3D tiles over [phi, phi_rate, action]"

        # Deterministic, centered offsets per tiling (same across dims, scaled by 1/n_tiles)
        til_axis = (np.arange(self.n_tilings) * 2 + 1) / (2.0 * self.n_tilings)  # (T,)
        self.offsets = np.tile(til_axis[:, None], (1, self.dim)) / self.n_tiles  # (T, dim)

    @property
    def total_tiles(self):
        return self.n_tilings * int(np.prod(self.n_tiles))


# ===============================================================
# Continuous-action Tile Q-Learning + UCB optimism + policy prior
# Q(s,a) = sum(w[idxs(s,a)]); two-stage argmax; feature-visit UCB
# ===============================================================
class TileQContAgent:
    def __init__(self,
                 obs_low, obs_high,          # [phi_low, rate_low]
                 act_low=-1.0, act_high=1.0,
                 n_tiles=(48, 48, 21), n_tilings=8,
                 alpha=0.02, gamma=0.98,
                 eps_start=0.99, eps_end=0.01, total_episodes=5000,
                 n_coarse=17, n_refine=13,
                 ucb_beta=1.2,
                 seed: int = 0):
        self.gamma = gamma
        self.alpha = alpha / float(n_tilings)
        self.eps_start, self.eps_end = eps_start, eps_end
        self.total_episodes = total_episodes

        self.rng = np.random.default_rng(seed)

        lows  = np.array([obs_low[0],  obs_low[1],  act_low],  dtype=np.float32)
        highs = np.array([obs_high[0], obs_high[1], act_high], dtype=np.float32)
        self.tc = TileCoder(lows, highs, n_tiles, n_tilings, seed=seed)

        self.n_features = self.tc.total_tiles
        self.w = np.zeros(self.n_features, dtype=np.float32)

        # feature visit counts for UCB optimism
        self.visit = np.zeros_like(self.w, dtype=np.int32)
        self.ucb_beta = float(ucb_beta)

        # cache shapes/strides
        self.T = self.tc.n_tilings
        self.n_phi, self.n_rate, self.n_act = map(int, self.tc.n_tiles)
        self.base = int(np.prod(self.tc.n_tiles))                # tiles per tiling
        self.t_offsets = (np.arange(self.T, dtype=np.int64) * self.base)  # (T,)

        self.obs_low  = np.array(obs_low[:2], dtype=np.float32)
        self.obs_high = np.array(obs_high[:2], dtype=np.float32)
        self.act_low, self.act_high = float(act_low), float(act_high)

        # candidate grids
        self.n_coarse = int(n_coarse)
        self.n_refine = int(n_refine)

        self.current_episode = 0
        self.eps = eps_start
        self._delta_accum = 0.0
        self._delta_count = 0

        # -------- Policy-side action prior (no reward change) --------
        self.last_action = 0.0   # inertia anchor
        self.gain_sign = 0       # +1: +u tends to increase φ; -1: +u tends to decrease φ; 0: unknown
        self.prior_scale = 0.30  # strength for sign alignment (try 0.15–0.5)
        self.inertia_lambda = 0.05  # discourage abrupt sign flip in selection (try 0.02–0.08)

    # ---------- helpers (vectorized) ----------
    def _state_bases(self, s):
        """
        For state s=[phi, rate], compute for each tiling t:
            base_term_t = n_act * (i_rate_t + n_rate * i_phi_t)
        so flat = base_term_t + i_act_t
        Returns int64 array (T,)
        """
        s = np.asarray(s, dtype=np.float32)
        ratios = (s - self.obs_low) / (self.obs_high - self.obs_low + 1e-8)  # (2,)
        ratios = np.clip(ratios, 0.0, 0.999999)

        offs = self.tc.offsets[:, :2]                                   # (T,2)
        shifted = (ratios[None, :] + offs) * self.tc.n_tiles[:2]        # (T,2)
        coords = np.floor(shifted).astype(np.int64)
        coords[:, 0] = np.clip(coords[:, 0], 0, self.n_phi - 1)         # i_phi
        coords[:, 1] = np.clip(coords[:, 1], 0, self.n_rate - 1)        # i_rate

        i_phi = coords[:, 0]
        i_rate = coords[:, 1]
        base_term = self.n_act * (i_rate + self.n_rate * i_phi)         # (T,)
        return base_term

    def _act_bins_multi(self, a_vec):
        """
        Given K actions, return i_act per tiling: int64 array (T,K)
        """
        a_vec = np.asarray(a_vec, dtype=np.float32)
        r = (a_vec - self.act_low) / (self.act_high - self.act_low + 1e-8)  # (K,)
        r = np.clip(r, 0.0, 0.999999)
        shift = (r[None, :] + self.tc.offsets[:, 2:3]) * self.n_act  # (T,K)
        i_act = np.floor(shift).astype(np.int64)
        np.clip(i_act, 0, self.n_act - 1, out=i_act)
        return i_act

    def _indices_for_sa(self, s, a):
        base_term = self._state_bases(s)                         # (T,)
        i_act = self._act_bins_multi(np.array([a], dtype=np.float32))[:, 0]  # (T,)
        flat = base_term + i_act                                 # (T,)
        idxs = self.t_offsets + flat                              # (T,)
        return idxs

    def _ucb_bonus_for_cols(self, idx_mat_TK):
        # idx_mat_TK: (T,K) active indices for each candidate
        # bonus per candidate: β / sqrt(1 + mean_t visits)
        mean_visits = np.mean(self.visit[idx_mat_TK], axis=0)
        return self.ucb_beta / np.sqrt(1.0 + mean_visits + 1e-9)

    def Qs_vectorized(self, s, a_vec, add_ucb=False):
        base_term = self._state_bases(s).astype(np.int64)        # (T,)
        i_act = self._act_bins_multi(a_vec).astype(np.int64)     # (T,K)
        idxs = (self.t_offsets[:, None] + base_term[:, None] + i_act)  # (T,K)
        q = np.sum(self.w[idxs], axis=0)                         # (K,)
        if add_ucb:
            q = q + self._ucb_bonus_for_cols(idxs)
        return q, idxs

    # -------- Policy-side prior (no reward change) --------
    def observe(self, s, a, s_next):
        """Update empirical torque→φ̇ sign and remember last action (metadata only)."""
        # Learn whether positive action tends to increase φ (through φ̇)
        if abs(a) > 0.4 and abs(s_next[1]) > 10.0:  # need some signal
            corr = np.sign(a) * np.sign(s_next[1])  # +1: +u → +φ̇ ; -1: +u → -φ̇
            self.gain_sign = 1 if corr >= 0 else -1
        self.last_action = float(a)

    def _prior_bonus(self, s, a_vec):
        """Small selection bias: (i) inertia; (ii) directional commit when far from goal."""
        a_vec = np.asarray(a_vec, dtype=np.float32)
        phi = float(s[0])
        d_fwd = (180.0 - phi) / 180.0
        d_fwd = float(np.clip(d_fwd, 0.0, 1.0))
        far_gate = 1.0 - math.exp(-8.0 * d_fwd)

        sign_bonus = 0.0
        if self.gain_sign != 0:
            desired_sign = +1 if self.gain_sign > 0 else -1
            sign_bonus = self.prior_scale * far_gate * (np.sign(a_vec) == desired_sign).astype(np.float32)

        inertia = -self.inertia_lambda * np.abs(a_vec - self.last_action)
        return sign_bonus + inertia

    # ---------- action selection (two-stage, fast) ----------
    def _coarse(self):
        return np.linspace(self.act_low, self.act_high, self.n_coarse, dtype=np.float32)

    def _refine_around(self, a_star, width=0.15):
        lo = max(self.act_low,  a_star - width)
        hi = min(self.act_high, a_star + width)
        return np.linspace(lo, hi, self.n_refine, dtype=np.float32)

    def _argmax_two_stage(self, s, with_ucb):
        # stage 1: coarse grid
        A1 = self._coarse()
        Q1, _ = self.Qs_vectorized(s, A1, add_ucb=with_ucb)
        Q1 = Q1 + self._prior_bonus(s, A1)       # policy-side prior
        i1 = int(np.argmax(Q1))
        a1 = float(A1[i1])

        # stage 2: refine locally
        A2 = self._refine_around(a1, width=max(0.05, (self.act_high - self.act_low) / self.n_coarse))
        Q2, _ = self.Qs_vectorized(s, A2, add_ucb=with_ucb)
        Q2 = Q2 + self._prior_bonus(s, A2)       # policy-side prior
        i2 = int(np.argmax(Q2))
        return float(A2[i2]), float(Q2[i2])

    def update_epsilon(self):
        frac = min(1.0, self.current_episode / max(1, self.total_episodes))
        self.eps = self.eps_start + (self.eps_end - self.eps_start) * frac

    def act(self, obs):
        # ε-greedy between uniform random and optimistic argmax
        if self.rng.random() < self.eps:
            return float(self.rng.uniform(self.act_low, self.act_high))
        a, _ = self._argmax_two_stage(obs, with_ucb=True)
        return float(a)

    def act_greedy(self, obs):
        a, _ = self._argmax_two_stage(obs, with_ucb=False)
        return float(a)

    def learn(self, obs, a, r, next_obs, done):
        # current Q
        idxs_sa = self._indices_for_sa(obs, a)     # (T,)
        q_sa = float(np.sum(self.w[idxs_sa]))

        # target
        if done:
            target = r
        else:
            # greedy (no optimism) for target
            _, qmax_next = self._argmax_two_stage(next_obs, with_ucb=False)
            target = r + self.gamma * qmax_next

        delta = target - q_sa
        self.w[idxs_sa] += self.alpha * delta      # semi-gradient update
        self.visit[idxs_sa] += 1                   # count updates for UCB

        self._delta_accum += abs(delta)
        self._delta_count += 1

    def avg_update(self):
        if self._delta_count == 0: return 0.0
        v = self._delta_accum / self._delta_count
        self._delta_accum = 0.0
        self._delta_count = 0
        return v

    def decay_eps(self):
        self.current_episode += 1
        self.update_epsilon()


# ===============================================================
# Eval & plotting helpers
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

def collect_mentor_demo(env, agent, store_to, max_steps=1500):
    """Run one greedy episode and push (s,a,r,s',d) into store_to list."""
    obs = env.reset()
    t = 0
    while t < max_steps:
        a = agent.act_greedy(obs)
        next_obs, r, done, info = env.step(a)
        # meta update for policy prior
        agent.observe(obs, a, next_obs)
        store_to.append((obs.copy(), float(a), float(r), next_obs.copy(), bool(done)))
        obs = next_obs
        t += 1
        if done:
            break

def run_greedy_trace(env, agent, max_steps=1500):
    obs = env.reset()

    t_steps, phi_deg_hist, phi_rate_hist, throttle_hist, rewards = [], [], [], [], []

    step = 0
    while step < max_steps:
        p  = float(obs[0])   # φ (deg, signed)
        pr = float(obs[1])   # φ rate (deg/s)
        a  = agent.act_greedy(obs)

        t_steps.append(step)
        phi_deg_hist.append(p)
        phi_rate_hist.append(pr)
        throttle_hist.append(a)

        obs, r, done, info = env.step(a)
        # keep policy prior metadata consistent during trace
        agent.observe(np.array([p, pr], dtype=np.float32), a, obs)
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
# Training (Pulse actions + curriculum + optional mentor replay)
# ===============================================================
def train_tileq_cont(episodes=600, max_steps=1500, eval_every=50, seed: int = 42,
                     # Pulse macro-actions (hold same throttle for PULSE_LEN env steps)
                     PULSE_LEN=4,
                     # Archive-curriculum knobs
                     reset_mix_prob=0.25, near_contact_eps=0.04, archive_max=64,
                     # Optional potential shaping (OFF by default)
                     use_potential=False,
                     # Optional eval-guided mentor replay
                     mentor_mix_prob=0.0, mentor_updates_per_step=1):
    np.random.seed(seed)
    random.seed(seed)

    env = MonsterTruckFlipEnvPitchSigned(render=False, realtime=False,
                                         frame_skip=10, max_steps=max_steps,
                                         seed=seed)
    # wire curriculum knobs
    env.reset_mix_prob = float(reset_mix_prob)
    env.near_contact_eps = float(near_contact_eps)
    env.archive_max = int(archive_max)
    # optional shaping
    env._use_potential = bool(use_potential)

    # OBS bounds: [φ (signed), φ_rate]
    obs_lows  = [-180.0, -720.0]
    obs_highs = [ 180.0,  720.0]

    agent = TileQContAgent(
        obs_low=obs_lows, obs_high=obs_highs,
        act_low=-1.0, act_high=1.0,
        n_tiles=(48, 48, 21), n_tilings=8,
        alpha=0.02, gamma=0.98,
        eps_start=0.99, eps_end=0.01, total_episodes=episodes,
        n_coarse=17, n_refine=13,
        ucb_beta=1.2,
        seed=seed
    )
    # keep env's shaping gamma in sync if you turn shaping ON
    env.gamma_train = agent.gamma

    # mentor buffer (optional)
    mentor = []

    rewards_list, success_flags, eval_ep_indices, eval_rewards = [], [], [], []
    successes = 0

    for ep in range(1, episodes + 1):
        obs = env.reset()
        ep_ret, success_flag = 0.0, 0
        t = 0
        agent.last_action = 0.0  # reset inertia anchor each episode

        while t < env.max_steps:
            a = agent.act(obs)  # continuous throttle

            # hold action for a short pulse to build rocking momentum
            chunk = 0
            done = False
            while chunk < PULSE_LEN and not done and t < env.max_steps:
                next_obs, r, done, info = env.step(a)

                # NEW: update policy prior metadata
                agent.observe(obs, a, next_obs)

                # usual TD update
                agent.learn(obs, a, r, next_obs, done)

                # optional mentor replay update(s)
                if mentor and mentor_mix_prob > 0.0 and (np.random.random() < mentor_mix_prob):
                    for _ in range(max(1, int(mentor_updates_per_step))):
                        ss, aa, rr, sn, dd = random.choice(mentor)
                        agent.learn(ss, aa, rr, sn, dd)

                obs = next_obs
                ep_ret += r
                chunk += 1
                t += 1

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
                  f"({sum(success_flags[-10:])}/10 success)  | "
                  f"archive={len(env.archive)} | mentor_mix={mentor_mix_prob:.3f}")

        # Evaluation (render only the first of the 3)
        if eval_every and (ep % eval_every == 0):
            eval_rs, eval_ss = [], []
            any_success = False
            for i in range(3):
                env.render_enabled = (i == 0)
                env.realtime = (i == 0)
                # turn off potential shaping during eval if you enabled it for training
                old_shape = env._use_potential
                env._use_potential = False
                er, es = evaluate_episode(env, agent, env.max_steps)
                env._use_potential = old_shape
                eval_rs.append(er)
                eval_ss.append(es)
                any_success = any_success or bool(es)

            env.render_enabled = False
            env.realtime = False

            mean_r, succ_rate = np.mean(eval_rs), np.mean(eval_ss) * 100
            eval_ep_indices.append(ep)
            eval_rewards.append(mean_r)
            print(f"   [Eval @ Ep {ep}] avg_reward={mean_r:.1f} | success_rate={succ_rate:.1f}%")

            # If eval found a success, record a greedy demo (fills mentor and also
            # populates the archive with near-contact snapshots during the rollout).
            if any_success:
                print("      ✓ Eval success captured. Guiding training (demo + archive)…")
                old_shape = env._use_potential
                env._use_potential = False
                collect_mentor_demo(env, agent, mentor, max_steps=env.max_steps)
                env._use_potential = old_shape

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
        plt.title("TileQ-Cont — Signed φ with forward-goal distance")
        plt.legend()
        plt.grid(True, alpha=0.3)
        plt.tight_layout()
        plt.savefig("tileq_pitch_eval.png", dpi=200)
        plt.close()
        print("📈 Saved eval plot to tileq_pitch_eval.png")

    # Greedy trace (no shaping, deterministic argmax)
    old_shape = env._use_potential
    env._use_potential = False
    traces = run_greedy_trace(env, agent, env.max_steps)
    env._use_potential = old_shape
    save_episode_traces_figure(traces, outfile="tileq_episode_summary.png")

    env.close()
    return rewards_list, (eval_ep_indices, eval_rewards)


if __name__ == "__main__":
    # You can tweak these few knobs quickly:
    rewards, eval_curve = train_tileq_cont(
        episodes=300, max_steps=2000, eval_every=50, seed=42,
        PULSE_LEN=6,                  # 3–6 is good for rocking
        reset_mix_prob=0.25,          # 0.15–0.35
        near_contact_eps=0.04,        # 0.02–0.05 m depending on wheel radius
        archive_max=64,
        use_potential=False,          # keep False to preserve pure reward; True is OK (optimality preserved)
        mentor_mix_prob=0.05,         # 0.02–0.10 lets eval demos guide learning
        mentor_updates_per_step=1
    )