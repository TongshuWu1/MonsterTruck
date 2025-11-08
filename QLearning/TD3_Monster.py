# TD3Monstertruck_angle_symmetric.py
# Continuous-control TD3 for Monster Truck flip (MuJoCo) with direction-agnostic rewards
# Requires: pip install mujoco glfw numpy torch
# For GPU: install a CUDA-enabled PyTorch; AMP uses torch.amp.* (new API)

import os, json, math, random, time
from collections import deque, namedtuple

import numpy as np
import torch
import torch.nn as nn
import torch.optim as optim
import mujoco
from mujoco.glfw import glfw


# =======================
# Environment (continuous)
# =======================
class MonsterTruckFlipEnv:
    """
    Observation: [up_z, ang_speed, wheel_speed]
      - up_z        : cosine of body Z vs world Z (1 upright, -1 upside down)
      - ang_speed   : ||angular velocity|| (world)
      - wheel_speed : mean rear wheel joint speeds
    Action: throttle in [-1, 1], applied to both rear motors.
    """

    def __init__(self, xml_path="monstertruck.model",
                 frame_skip=40,           # large action repeat for speed
                 max_steps=600,           # shorter episodes
                 render=False,
                 realtime=False):
        if not os.path.exists(xml_path):
            raise FileNotFoundError(f"Cannot find {xml_path}")
        self.model = mujoco.MjModel.from_xml_path(xml_path)
        self.data = mujoco.MjData(self.model)

        self.frame_skip = frame_skip
        self.max_steps = max_steps
        self.render_enabled = render
        self.realtime = realtime
        self.dt = self.model.opt.timestep

        # IDs / cached addresses
        self.body_id = mujoco.mj_name2id(self.model, mujoco.mjtObj.mjOBJ_BODY, "chassis")
        self.j_rl    = mujoco.mj_name2id(self.model, mujoco.mjtObj.mjOBJ_JOINT, "j_rl")
        self.j_rr    = mujoco.mj_name2id(self.model, mujoco.mjtObj.mjOBJ_JOINT, "j_rr")
        self.dof_rl  = self.model.jnt_dofadr[self.j_rl] if self.j_rl != -1 else None
        self.dof_rr  = self.model.jnt_dofadr[self.j_rr] if self.j_rr != -1 else None

        # Success / stability — all direction-agnostic (angle-based)
        self.near_deg             = 12.0                   # near-upright threshold (degrees)
        self.far_deg              = 15.0                   # far-from-upright threshold (for momentum)
        self.upright_threshold    = math.radians(self.near_deg)  # angle_err < this = near upright
        self.stable_rate_tol      = 1.5                    # |flip_rate_x| <= tol for stability
        self.hold_needed          = 8                      # consecutive agent steps near&stable

        # Episode state
        self.hold_counter     = 0
        self.step_count       = 0
        self.best_angle_err   = math.pi
        self._no_improve_steps = 0

        # Early termination (speed)
        self.max_radius       = 3.0     # terminate if drift too far from origin
        self.max_no_improve   = 200     # terminate if no progress N steps
        self.min_improve_deg  = 0.5     # count progress only if better than this (deg)

        # Viewer / pacing
        self.window = None
        self._wall_start = None
        if self.render_enabled:
            self._init_viewer()

    # ---------- Viewer ----------
    def _init_viewer(self):
        if not glfw.init():
            raise RuntimeError("GLFW init failed")
        self.window = glfw.create_window(1000, 800, "MonsterTruck TD3 (eval)", None, None)
        if not self.window:
            glfw.terminate()
            raise RuntimeError("GLFW window creation failed")
        glfw.make_context_current(self.window)
        glfw.swap_interval(1)  # ask for VSync; driver may override
        self.cam = mujoco.MjvCamera()
        self.opt = mujoco.MjvOption()
        self.scene = mujoco.MjvScene(self.model, maxgeom=4000)  # cheaper than 20000
        self.context = mujoco.MjrContext(self.model, mujoco.mjtFontScale.mjFONTSCALE_150)
        self.cam.distance, self.cam.azimuth, self.cam.elevation = 3.0, 90, -25

    def _destroy_viewer(self):
        if self.window is not None:
            glfw.destroy_window(self.window)
            self.window = None
        glfw.terminate()

    def _render(self):
        if not self.render_enabled or self.window is None:
            return
        if glfw.window_should_close(self.window):
            self.render_enabled = False
            self._destroy_viewer()
            return
        self.cam.lookat[:] = self.data.xpos[self.body_id]
        mujoco.mjv_updateScene(self.model, self.data, self.opt, None, self.cam,
                               mujoco.mjtCatBit.mjCAT_ALL, self.scene)
        w, h = glfw.get_framebuffer_size(self.window)
        if w > 0 and h > 0:
            mujoco.mjr_render(mujoco.MjrRect(0, 0, w, h), self.scene, self.context)
        glfw.swap_buffers(self.window)
        glfw.poll_events()

    # ---------- Env API ----------
    def reset(self):
        mujoco.mj_resetData(self.model, self.data)
        self.data.qpos[:3]  = np.array([0.0, 0.0, 0.6]) + 0.02 * np.random.randn(3)
        self.data.qvel[:]   = 0.0
        self.data.qpos[3:7] = np.array([0.0, 1.0, 0.0, 0.0])  # upside down
        self.hold_counter   = 0
        self.step_count     = 0
        self.best_angle_err = math.pi
        self._no_improve_steps = 0
        mujoco.mj_forward(self.model, self.data)
        self._wall_start = time.perf_counter() if (self.render_enabled and self.realtime) else None
        return self._get_obs()

    def _get_obs(self):
        R = self.data.xmat[self.body_id].reshape(3, 3)
        up_z = float(R[2, 2])
        ang_speed = float(np.linalg.norm(self.data.cvel[self.body_id][3:]))
        if self.dof_rl is not None and self.dof_rr is not None:
            v_rl = self.data.qvel[self.dof_rl]
            v_rr = self.data.qvel[self.dof_rr]
            wheel_speed = 0.5 * (v_rl + v_rr)
        else:
            wheel_speed = 0.0
        return np.array([up_z, ang_speed, wheel_speed], dtype=np.float32)

    def step(self, throttle: float):
        throttle = float(np.clip(throttle, -1.0, 1.0))
        done = False
        success = False

        # Physics (action repeat)
        for _ in range(self.frame_skip):
            if self.model.nu >= 2:
                self.data.ctrl[0] = throttle
                self.data.ctrl[1] = throttle
            mujoco.mj_step(self.model, self.data)

        # No real-time pacing unless rendering
        if self.render_enabled and self.realtime and self._wall_start is not None:
            target_wall = self._wall_start + self.data.time
            now = time.perf_counter()
            if now < target_wall:
                time.sleep(target_wall - now)

        self.step_count += 1

        # Kinematics
        R = self.data.xmat[self.body_id].reshape(3, 3)
        up_z = float(R[2, 2])
        ang_vel_world = self.data.cvel[self.body_id][3:]
        ang_vel_body  = R.T @ ang_vel_world
        flip_rate     = float(ang_vel_body[0])  # rotation about x in body frame
        pos_xy        = float(np.linalg.norm(self.data.xpos[self.body_id][:2]))

        # Direction-agnostic angle error to upright
        body_up   = R[:, 2]
        cos_theta = float(np.clip(np.dot(body_up, np.array([0, 0, 1])), -1.0, 1.0))
        angle_err = math.acos(cos_theta)  # [0, pi]; 0 = upright

        # ---------- Direction-agnostic reward shaping ----------
        # 1) Alignment shaping: smooth, symmetric (0..1)
        r_align = 1.0 - (angle_err / math.pi) ** 2

        # 2) Progress bonus: reward reductions in angle_err (direction-independent)
        r_progress = 0.0
        improve_rad = math.radians(self.min_improve_deg)
        if angle_err < self.best_angle_err - improve_rad:
            r_progress = 8.0 * (self.best_angle_err - angle_err)
            self.best_angle_err = angle_err
            self._no_improve_steps = 0
        else:
            self._no_improve_steps += 1

        # 3) Flipping momentum: encourage rotation only when far from upright
        far_from_upright = angle_err > math.radians(self.far_deg)
        r_flip = 0.10 * abs(flip_rate) * float(far_from_upright)

        # 4) Over-flip penalty: near upright, discourage rotation
        near_upright = angle_err < self.upright_threshold
        r_overflip = -0.08 * abs(flip_rate) * float(near_upright)

        # 5) Drift penalty: keep near origin
        r_drift = -0.02 * (pos_xy ** 2)

        # 6) Small time penalty
        r_time = -0.001

        reward = r_align + r_progress + r_flip + r_overflip + r_drift + r_time

        # ---------- Success (direction-agnostic) ----------
        if near_upright and (abs(flip_rate) <= self.stable_rate_tol):
            self.hold_counter += 1
        else:
            self.hold_counter = 0

        if self.hold_counter >= self.hold_needed:
            reward += 25.0
            done, success = True, True

        # Early termination for speed
        if pos_xy > self.max_radius:
            reward -= 2.0
            done = True
        if self._no_improve_steps >= self.max_no_improve:
            reward -= 1.0
            done = True

        if self.step_count >= self.max_steps:
            done = True

        obs = self._get_obs()
        if not np.isfinite(obs).all():
            reward -= 5.0
            done = True

        self._render()
        return obs, reward, done, {"success": success, "angle_err": angle_err, "pos_xy": pos_xy}

    def close(self):
        if self.render_enabled and self.window is not None:
            self._destroy_viewer()


# ======================
# TD3 (GPU AMP, no compile)
# ======================
Transition = namedtuple("Transition", ["s","a","r","sn","done"])

class Replay:
    def __init__(self, cap=100_000):  # smaller = faster sampling
        self.buf = deque(maxlen=cap)
    def push(self, *args): self.buf.append(Transition(*args))
    def __len__(self): return len(self.buf)
    def sample(self, n, pin=False):
        idx = np.random.choice(len(self.buf), size=n, replace=False)
        b = [self.buf[i] for i in idx]
        s  = np.stack([x.s  for x in b]).astype(np.float32)
        a  = np.array([x.a for x in b], dtype=np.float32).reshape(-1,1)
        r  = np.array([x.r for x in b], dtype=np.float32).reshape(-1,1)
        sn = np.stack([x.sn for x in b]).astype(np.float32)
        d  = np.array([x.done for x in b], dtype=np.float32).reshape(-1,1)
        to = (lambda arr: torch.from_numpy(arr).pin_memory()) if pin else torch.from_numpy
        return to(s), to(a), to(r), to(sn), to(d)

def mlp(in_dim, out_dim, hidden=128):
    return nn.Sequential(
        nn.Linear(in_dim, hidden), nn.ReLU(),
        nn.Linear(hidden, hidden), nn.ReLU(),
        nn.Linear(hidden, out_dim)
    )

class Actor(nn.Module):
    def __init__(self, obs_dim, act_limit=1.0):
        super().__init__()
        self.net = mlp(obs_dim, 1, 128)
        self.act_limit = act_limit
    def forward(self, s):
        return torch.tanh(self.net(s)) * self.act_limit

class Critic(nn.Module):
    def __init__(self, obs_dim):
        super().__init__()
        self.q1 = mlp(obs_dim + 1, 1, 128)
        self.q2 = mlp(obs_dim + 1, 1, 128)
    def forward(self, s, a):
        x = torch.cat([s, a], dim=-1)
        return self.q1(x), self.q2(x)

class TD3:
    def __init__(self, obs_dim=3, act_limit=1.0, device="cuda"):
        self.device = torch.device(device if device else ('cuda' if torch.cuda.is_available() else 'cpu'))
        self.actor      = Actor(obs_dim, act_limit).to(self.device)
        self.actor_tgt  = Actor(obs_dim, act_limit).to(self.device)
        self.critic     = Critic(obs_dim).to(self.device)
        self.critic_tgt = Critic(obs_dim).to(self.device)
        self.actor_tgt.load_state_dict(self.actor.state_dict())
        self.critic_tgt.load_state_dict(self.critic.state_dict())

        self.pi_opt = optim.Adam(self.actor.parameters(),  lr=1e-3)
        self.q_opt  = optim.Adam(self.critic.parameters(), lr=1e-3)

        # AMP scaler (new API)
        self.scaler = torch.amp.GradScaler(enabled=(self.device.type == "cuda"))

        # TD3 hyperparams
        self.gamma = 0.98
        self.tau = 0.01              # slightly faster target tracking
        self.policy_noise = 0.2
        self.noise_clip = 0.5
        self.policy_delay = 2
        self.total_it = 0
        self.act_limit = act_limit

    @torch.no_grad()
    def act(self, obs, noise_std=0.1):
        s = torch.as_tensor(obs, dtype=torch.float32, device=self.device).unsqueeze(0)
        a = self.actor(s).squeeze(0)
        if noise_std > 0:
            a = a + noise_std * torch.randn_like(a)
        return float(torch.clamp(a, -self.act_limit, self.act_limit).item())

    def train_step(self, replay: Replay, batch_size=128):
        self.total_it += 1
        pin = (self.device.type == "cuda")
        s, a, r, sn, d = replay.sample(batch_size, pin=pin)
        s  = s.to(self.device, non_blocking=pin)
        a  = a.to(self.device, non_blocking=pin)
        r  = r.to(self.device, non_blocking=pin)
        sn = sn.to(self.device, non_blocking=pin)
        d  = d.to(self.device, non_blocking=pin)

        # Critic update
        with torch.amp.autocast(device_type="cuda", enabled=(self.device.type=="cuda")):
            with torch.no_grad():
                an = self.actor_tgt(sn)
                noise = (torch.randn_like(an) * self.policy_noise).clamp(-self.noise_clip, self.noise_clip)
                an = (an + noise).clamp(-self.act_limit, self.act_limit)
                q1n, q2n = self.critic_tgt(sn, an)
                qn = torch.min(q1n, q2n)
                y = r + (1.0 - d) * self.gamma * qn

            q1, q2 = self.critic(s, a)
            q_loss = nn.functional.mse_loss(q1, y) + nn.functional.mse_loss(q2, y)

        self.q_opt.zero_grad(set_to_none=True)
        self.scaler.scale(q_loss).backward()
        nn.utils.clip_grad_norm_(self.critic.parameters(), 10.0)
        self.scaler.step(self.q_opt)
        self.scaler.update()

        # Delayed actor update
        if self.total_it % self.policy_delay == 0:
            with torch.amp.autocast(device_type="cuda", enabled=(self.device.type=="cuda")):
                a_pi = self.actor(s)
                q1_pi, _ = self.critic(s, a_pi)
                pi_loss = -q1_pi.mean()

            self.pi_opt.zero_grad(set_to_none=True)
            self.scaler.scale(pi_loss).backward()
            nn.utils.clip_grad_norm_(self.actor.parameters(), 10.0)
            self.scaler.step(self.pi_opt)
            self.scaler.update()

            # Soft target updates
            with torch.no_grad():
                for p, pt in zip(self.critic.parameters(), self.critic_tgt.parameters()):
                    pt.data.mul_(1 - self.tau).add_(self.tau * p.data)
                for p, pt in zip(self.actor.parameters(), self.actor_tgt.parameters()):
                    pt.data.mul_(1 - self.tau).add_(self.tau * p.data)


# ===============
# Training (FAST)
# ===============
def train_td3(
    episodes=500,
    max_steps=600,
    start_random_steps=2000,
    batch_size=128,
    updates_per_step=1,
    save_every=100,
    seed=0,
    device="cuda",
    xml_path="monstertruck.model",
    do_eval_render=False
):
    random.seed(seed); np.random.seed(seed); torch.manual_seed(seed)
    agent = TD3(obs_dim=3, act_limit=1.0, device=device)
    replay = Replay(cap=100_000)

    # One persistent headless env for all training episodes
    env = MonsterTruckFlipEnv(xml_path=xml_path, render=False, realtime=False,
                              frame_skip=40, max_steps=max_steps)

    successes = 0
    global_step = 0

    for ep in range(1, episodes + 1):
        obs = env.reset()
        ep_ret = 0.0
        best_angle = math.pi

        for t in range(max_steps):
            if global_step < start_random_steps:
                a = np.random.uniform(-1.0, 1.0)
            else:
                noise_std = 0.1 if ep < int(episodes * 0.7) else 0.05
                a = agent.act(obs, noise_std=noise_std)

            next_obs, r, done, info = env.step(a)
            replay.push(obs, a, r, next_obs, float(done))

            obs = next_obs
            ep_ret += r
            best_angle = min(best_angle, info.get("angle_err", math.pi))
            global_step += 1

            if len(replay) >= start_random_steps:
                for _ in range(updates_per_step):
                    agent.train_step(replay, batch_size=batch_size)

            if done:
                if info.get("success", False):
                    successes += 1
                break

        if ep % 10 == 0:
            print(f"Ep {ep:4d} | Ret {ep_ret:7.2f} | best_angle {best_angle*180/math.pi:6.1f}° "
                  f"| buf {len(replay):6d} | success {successes}")

        if save_every and ep % save_every == 0:
            torch.save({
                "actor": agent.actor.state_dict(),
                "critic": agent.critic.state_dict(),
                "episodes": ep,
                "global_step": global_step
            }, f"td3_monstertruck_ep{ep}.pt")
            with open("td3_meta.json", "w") as f:
                json.dump({"obs_dim": 3, "act_limit": 1.0}, f, indent=2)

        # Optional: slow evaluation with rendering
        if do_eval_render and ep % 100 == 0:
            eval_env = MonsterTruckFlipEnv(xml_path=xml_path, render=True, realtime=True,
                                           frame_skip=10, max_steps=800)
            o = eval_env.reset()
            done = False
            while not done:
                a = agent.act(o, noise_std=0.0)
                o, _, done, _ = eval_env.step(a)
            eval_env.close()

    env.close()
    torch.save({"actor": agent.actor.state_dict(),
                "critic": agent.critic.state_dict()}, "td3_monstertruck_final.pt")
    print(f"Done! Successes: {successes}/{episodes}")


if __name__ == "__main__":
    # For extra sim speed (optional), in monstertruck.model use:
    # <option integrator="Euler" timestep="0.002"/>
    train_td3(
        episodes=500,
        max_steps=600,
        start_random_steps=2000,
        batch_size=128,
        updates_per_step=1,
        save_every=100,
        seed=0,
        device="cuda",          # set "cpu" to force CPU
        xml_path="../tile_q/monstertruck.xml",
        do_eval_render=False
    )
