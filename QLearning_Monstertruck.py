# QLearningMonstertruck.py
# Monster-truck flip learning with direction-independent upright reward
# Requires: pip install mujoco glfw numpy

import os, json, math, random
import numpy as np
import mujoco
from mujoco.glfw import glfw


# ============================================================
# Small helpers
# ============================================================
def clip(v, lo, hi):
    return lo if v < lo else (hi if v > hi else v)

def digitize(x, bins):
    return int(np.digitize([x], bins)[0])


# ============================================================
# Environment
# ============================================================
class MonsterTruckFlipEnv:
    """
    Discrete throttle env for flipping a truck upright (both forward and backward).

    Observations:
        up_z        : cosine of uprightness (1 = upright, -1 = upside down)
        ang_speed   : total angular speed
        wheel_speed : mean of rear wheel joint speeds

    Reward:
        + smoother upright shaping
        + big progress reward when angle error to upright decreases
        + flipping momentum reward around x-axis (while not upright)
        - over-flip penalty if still rotating after upright
        - drift penalty for leaving origin
        - small time penalty
        + terminal bonus for stable upright
    """

    def __init__(self, xml_path="monstertruck.xml",
                 frame_skip=10, max_steps=1200, render=False):
        if not os.path.exists(xml_path):
            raise FileNotFoundError(f"Cannot find {xml_path}")
        self.model = mujoco.MjModel.from_xml_path(xml_path)
        self.data = mujoco.MjData(self.model)

        self.frame_skip = frame_skip
        self.max_steps = max_steps
        self.render_enabled = render
        self.dt = self.model.opt.timestep

        # ids / action map
        self.body_id = mujoco.mj_name2id(self.model, mujoco.mjtObj.mjOBJ_BODY, "chassis")
        self.actions = np.array([-1.0, -0.5, 0.0, 0.5, 1.0], dtype=np.float32)

        # success criteria and over-flip control
        self.upright_threshold = 0.92
        self.stop_flip_threshold = 0.92
        self.stable_rate_tol = 1.5
        self.hold_needed = 4

        # episode state
        self.hold_counter = 0
        self.step_count = 0
        self.best_angle_err = math.pi  # track smallest angle to upright
        self.prev_best_angle_err = math.pi

        # viewer
        self.window = None
        if self.render_enabled:
            self._init_viewer()

    # ---------------------- Rendering ----------------------
    def _init_viewer(self):
        if not glfw.init():
            raise RuntimeError("GLFW init failed")
        self.window = glfw.create_window(1000, 800, "MonsterTruck Q-Learning", None, None)
        glfw.make_context_current(self.window)
        glfw.swap_interval(1)
        self.cam = mujoco.MjvCamera()
        self.opt = mujoco.MjvOption()
        self.scene = mujoco.MjvScene(self.model, maxgeom=20000)
        self.context = mujoco.MjrContext(self.model, mujoco.mjtFontScale.mjFONTSCALE_150)
        self.cam.distance, self.cam.azimuth, self.cam.elevation = 3.0, 90, -25

    def _render(self):
        if not self.render_enabled:
            return
        self.cam.lookat[:] = self.data.xpos[self.body_id]
        mujoco.mjv_updateScene(self.model, self.data, self.opt, None, self.cam,
                               mujoco.mjtCatBit.mjCAT_ALL, self.scene)
        w, h = glfw.get_framebuffer_size(self.window)
        mujoco.mjr_render(mujoco.MjrRect(0, 0, w, h), self.scene, self.context)
        glfw.swap_buffers(self.window)
        glfw.poll_events()

    # ---------------------- Env API ----------------------
    def reset(self):
        mujoco.mj_resetData(self.model, self.data)
        self.data.qpos[:3] = np.array([0.0, 0.0, 0.6]) + 0.02 * np.random.randn(3)
        self.data.qvel[:] = 0.0
        self.data.qpos[3:7] = np.array([0.0, 1.0, 0.0, 0.0])  # upside down
        self.hold_counter = 0
        self.step_count = 0
        self.best_angle_err = math.pi
        self.prev_best_angle_err = math.pi
        mujoco.mj_forward(self.model, self.data)
        return self._get_obs()

    def _get_obs(self):
        R = self.data.xmat[self.body_id].reshape(3, 3)
        up_z = float(R[2, 2])
        ang_speed = float(np.linalg.norm(self.data.cvel[self.body_id][3:]))
        j_rl = mujoco.mj_name2id(self.model, mujoco.mjtObj.mjOBJ_JOINT, "j_rl")
        j_rr = mujoco.mj_name2id(self.model, mujoco.mjtObj.mjOBJ_JOINT, "j_rr")
        v_rl = self.data.qvel[self.model.jnt_dofadr[j_rl]]
        v_rr = self.data.qvel[self.model.jnt_dofadr[j_rr]]
        wheel_speed = 0.5 * (v_rl + v_rr)
        return np.array([up_z, ang_speed, wheel_speed], dtype=np.float32)

    def step(self, a_idx: int):
        throttle = float(self.actions[a_idx])
        done, success = False, False

        # step physics
        for _ in range(self.frame_skip):
            if self.model.nu >= 2:
                self.data.ctrl[0] = throttle
                self.data.ctrl[1] = throttle
            mujoco.mj_step(self.model, self.data)

        self.step_count += 1
        R = self.data.xmat[self.body_id].reshape(3, 3)
        up_z = float(R[2, 2])
        ang_vel_world = self.data.cvel[self.body_id][3:]
        ang_vel_body = R.T @ ang_vel_world
        flip_rate = float(ang_vel_body[0])
        ang_speed = float(np.linalg.norm(ang_vel_world))
        pos_xy = float(np.linalg.norm(self.data.xpos[self.body_id][:2]))

        # compute uprightness angle error
        body_up = R[:, 2]
        world_up = np.array([0, 0, 1])
        cos_theta = np.dot(body_up, world_up)
        cos_theta = np.clip(cos_theta, -1.0, 1.0)
        angle_err = math.acos(cos_theta)  # radians: 0 upright, pi upside down

        # ---------------- Reward terms ----------------
        r_upright = ((up_z + 1.0) / 2.0) ** 3

        # direction-independent progress reward
        r_progress = 0.0
        if angle_err < self.best_angle_err:
            r_progress = 8.0 * (self.best_angle_err - angle_err)
            self.best_angle_err = angle_err
        self.prev_best_angle_err = self.best_angle_err

        # flipping momentum (encourage strong rotation while not upright)
        r_flip = 0.10 * abs(flip_rate) * (1.0 - max(up_z, 0.0))

        # over-flip penalty: when upright, stop rotation
        r_overflip = 0.0
        if up_z >= self.stop_flip_threshold:
            r_overflip = -0.08 * abs(flip_rate)

        # drift penalty
        r_drift = -0.05 * (pos_xy ** 2)

        # time penalty
        r_time = -0.001

        reward = r_upright + r_progress + r_flip + r_overflip + r_drift + r_time

        # success: upright and stable
        if (up_z >= self.upright_threshold) and (abs(flip_rate) <= self.stable_rate_tol):
            self.hold_counter += 1
        else:
            self.hold_counter = 0

        if self.hold_counter >= self.hold_needed:
            reward += 25.0
            done, success = True, True

        # timeout
        if self.step_count >= self.max_steps:
            done = True

        obs = self._get_obs()
        if not np.isfinite(obs).all():
            reward -= 5.0
            done = True

        self._render()
        return obs, reward, done, {"success": success, "angle_err": angle_err}

    def close(self):
        if self.render_enabled and self.window is not None:
            glfw.terminate()


# ============================================================
# Tabular Q-Learning Agent
# ============================================================
class QAgent:
    def __init__(self,
                 obs_bins=(17, 12, 12),
                 obs_ranges=((-1.0, 1.0), (0.0, 50.0), (-120.0, 120.0)),
                 n_actions=5,
                 alpha=0.3, gamma=0.98,
                 eps_start=1.0, eps_end=0.05, eps_decay=0.9995):
        self.obs_bins = obs_bins
        self.obs_ranges = obs_ranges
        self.n_actions = n_actions
        self.alpha, self.gamma = alpha, gamma
        self.eps, self.eps_end, self.eps_decay = eps_start, eps_end, eps_decay

        self.q = np.zeros(obs_bins + (n_actions,), dtype=np.float32)
        self.bin_edges = [np.linspace(lo, hi, nb - 1)
                          for (lo, hi), nb in zip(obs_ranges, obs_bins)]

    def _disc(self, obs):
        idxs = []
        for o, (lo, hi), edges in zip(obs, self.obs_ranges, self.bin_edges):
            idxs.append(digitize(clip(float(o), lo, hi), edges))
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
            json.dump({
                "obs_bins": self.obs_bins,
                "obs_ranges": self.obs_ranges,
                "n_actions": self.n_actions
            }, f, indent=2)


# ============================================================
# Training
# ============================================================
def train(episodes=1000, render_every=50, max_steps=1500, save_every=100):
    agent = QAgent()
    env = None
    successes = 0

    for ep in range(1, episodes + 1):
        if env:
            env.close()
        do_render = (render_every is not None) and (ep % render_every == 0)
        env = MonsterTruckFlipEnv(render=do_render, frame_skip=10, max_steps=max_steps)

        obs = env.reset()
        ep_ret = 0.0
        best_angle = math.pi

        for _ in range(max_steps):
            a = agent.act(obs)
            next_obs, r, done, info = env.step(a)
            agent.learn(obs, a, r, next_obs, done)
            obs = next_obs
            ep_ret += r
            best_angle = min(best_angle, info.get("angle_err", math.pi))
            if done:
                if info.get("success", False):
                    successes += 1
                break

        agent.decay_eps()

        if ep % 10 == 0:
            print(f"Ep {ep:4d} | Ret {ep_ret:7.2f} | best_angle {best_angle*180/math.pi:6.1f}° | eps {agent.eps:5.3f} | success {successes}")

        if save_every and ep % save_every == 0:
            agent.save()

    env.close()
    agent.save()
    print(f"Done! Successes: {successes}/{episodes}")


if __name__ == "__main__":
    train(episodes=5000, render_every=200, max_steps=2000, save_every=100)
