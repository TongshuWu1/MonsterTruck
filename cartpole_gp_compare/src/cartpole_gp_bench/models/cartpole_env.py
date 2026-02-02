# cartpole_env.py

from __future__ import annotations

import math
from typing import Any, Dict, Optional, Tuple

import numpy as np
import gymnasium as gym
from gymnasium import spaces
from gymnasium.wrappers import TimeLimit
from gymnasium.utils import seeding

if not hasattr(np, "bool8"):
    np.bool8 = np.bool_

U_MIN, U_MAX = -1.0, 1.0
MAX_EPISODE_STEPS = 600


def wrap_pi(x: float) -> float:
    return (x + np.pi) % (2.0 * np.pi) - np.pi


def obs_to_state(obs: np.ndarray) -> Tuple[float, float, float, float]:
    x, xdot, th, thdot = float(obs[0]), float(obs[1]), float(obs[2]), float(obs[3])
    return x, xdot, wrap_pi(th), float(thdot)


def state_to_features(
    x: float,
    xdot: float,
    theta: float,
    thetadot: float,
    u: float,
    x_scale: float = 2.4,
    v_scale: float = 3.0,
    w_scale: float = 8.0,
) -> np.ndarray:
    return np.array(
        [
            np.tanh(x / x_scale),
            np.tanh(xdot / v_scale),
            np.sin(theta),
            np.cos(theta),
            np.tanh(thetadot / w_scale),
            float(u),
        ],
        dtype=np.float64,
    )


class ContinuousCartPoleSwingUpEnv(gym.Env):
    metadata = {"render_modes": ["rgb_array", "human"], "render_fps": 50}

    def __init__(self, render_mode: Optional[str] = None, start_down: bool = True):
        super().__init__()

        self.gravity = 9.8
        self.masscart = 1.0
        self.masspole = 0.1
        self.total_mass = self.masspole + self.masscart
        self.length = 0.5
        self.polemass_length = self.masspole * self.length

        self.force_mag = 30.0
        self.tau = 0.02
        self.min_action = -1.0
        self.max_action = 1.0

        self.x_threshold = 2.4
        self.start_down = bool(start_down)

        self.render_mode = render_mode
        self.state: Optional[Tuple[float, float, float, float]] = None
        self.np_random = None
        self.seed()

        high = np.array(
            [self.x_threshold * 2.0, np.finfo(np.float32).max, np.finfo(np.float32).max, np.finfo(np.float32).max],
            dtype=np.float32,
        )
        self.action_space = spaces.Box(
            low=np.array([self.min_action], dtype=np.float32),
            high=np.array([self.max_action], dtype=np.float32),
            shape=(1,),
            dtype=np.float32,
        )
        self.observation_space = spaces.Box(-high, high, dtype=np.float32)

        self._pygame_inited = False
        self._screen = None
        self._clock = None
        self._surf = None

    def seed(self, seed: Optional[int] = None):
        self.np_random, seed = seeding.np_random(seed)
        return [seed]

    def stepPhysics(self, force: float):
        x, x_dot, theta, theta_dot = self.state
        costheta = math.cos(theta)
        sintheta = math.sin(theta)

        temp = (force + self.polemass_length * theta_dot * theta_dot * sintheta) / self.total_mass
        thetaacc = (self.gravity * sintheta - costheta * temp) / (
            self.length * (4.0 / 3.0 - self.masspole * costheta * costheta / self.total_mass)
        )
        xacc = temp - self.polemass_length * thetaacc * costheta / self.total_mass

        x = x + self.tau * x_dot
        x_dot = x_dot + self.tau * xacc
        theta = theta + self.tau * theta_dot
        theta_dot = theta_dot + self.tau * thetaacc
        return (x, x_dot, theta, theta_dot)

    def step(self, action):
        action = np.asarray(action, dtype=np.float32).reshape(1,)
        if not self.action_space.contains(action):
            raise ValueError(f"{action} invalid")

        u = float(action[0])
        force = self.force_mag * u

        self.state = self.stepPhysics(force)
        x, x_dot, theta, theta_dot = self.state

        terminated = bool(x < -self.x_threshold or x > self.x_threshold)
        truncated = False

        reward = (
            +1.0 * math.cos(theta)
            - 0.01 * (x * x)
            - 0.001 * (x_dot * x_dot)
            - 0.001 * (theta_dot * theta_dot)
            - 0.001 * (u * u)
        )

        obs = np.array([x, x_dot, theta, theta_dot], dtype=np.float32)
        info: Dict[str, Any] = dict(x=x, x_dot=x_dot, theta=theta, theta_dot=theta_dot, u=u)
        return obs, float(reward), terminated, truncated, info

    def reset(self, *, seed: Optional[int] = None, options: Optional[dict] = None):
        super().reset(seed=seed)
        if seed is not None:
            self.seed(seed)

        x = float(self.np_random.uniform(low=-0.05, high=0.05))
        x_dot = float(self.np_random.uniform(low=-0.05, high=0.05))
        theta_dot = float(self.np_random.uniform(low=-0.05, high=0.05))

        if self.start_down:
            theta = float(math.pi + self.np_random.uniform(low=-0.10, high=0.10))
        else:
            theta = float(self.np_random.uniform(low=-0.10, high=0.10))

        self.state = (x, x_dot, theta, theta_dot)
        obs = np.array(self.state, dtype=np.float32)
        return obs, {}

    def _init_pygame(self):
        if self._pygame_inited:
            return
        import pygame
        from pygame import gfxdraw  # noqa: F401

        pygame.init()
        self._pygame_inited = True
        self._clock = pygame.time.Clock()
        if self.render_mode == "human":
            self._screen = pygame.display.set_mode((600, 400))
            pygame.display.set_caption("Continuous CartPole Swing-Up")
        self._surf = pygame.Surface((600, 400))

    def render(self):
        if self.render_mode is None:
            return None
        if self.state is None:
            return None

        import pygame
        from pygame import gfxdraw

        self._init_pygame()

        W, H = 600, 400
        surf = self._surf
        surf.fill((250, 250, 250))

        x, x_dot, theta, theta_dot = self.state

        world_width = self.x_threshold * 2.0
        scale = W / world_width
        carty = 260
        cartw = 60
        carth = 30
        polelen = scale * (2.0 * self.length)
        polew = 8

        track_y = carty + carth // 2
        pygame.draw.line(surf, (50, 50, 50), (0, track_y), (W, track_y), 2)

        cartx = int(W / 2.0 + x * scale)

        l = cartx - cartw // 2
        r = cartx + cartw // 2
        t = carty - carth // 2
        b = carty + carth // 2

        pygame.draw.rect(surf, (20, 20, 20), pygame.Rect(l, t, cartw, carth), 2)

        pivot = (cartx, t)
        tip = (cartx + polelen * math.sin(theta), t - polelen * math.cos(theta))

        def thick_line(a, b, width, color):
            ax, ay = a
            bx, by = b
            vx, vy = bx - ax, by - ay
            norm = math.hypot(vx, vy) + 1e-9
            nx, ny = -vy / norm, vx / norm
            w2 = width / 2.0
            pts = [
                (ax + nx * w2, ay + ny * w2),
                (ax - nx * w2, ay - ny * w2),
                (bx - nx * w2, by - ny * w2),
                (bx + nx * w2, by + ny * w2),
            ]
            pygame.draw.polygon(surf, color, pts)

        thick_line(pivot, tip, polew, (80, 160, 255))
        gfxdraw.filled_circle(surf, pivot[0], pivot[1], 6, (220, 80, 80))
        gfxdraw.aacircle(surf, pivot[0], pivot[1], 6, (220, 80, 80))

        if self.render_mode == "human":
            for _ in pygame.event.get():
                pass
            self._screen.blit(surf, (0, 0))
            pygame.display.flip()
            self._clock.tick(self.metadata.get("render_fps", 50))
            return None

        arr = np.transpose(np.array(pygame.surfarray.pixels3d(surf)), (1, 0, 2))
        return arr.copy()

    def close(self):
        if self._pygame_inited:
            import pygame

            pygame.quit()
        self._pygame_inited = False
        self._screen = None
        self._clock = None
        self._surf = None


class EdgeRespawnWrapper(gym.Wrapper):
    def __init__(self, env, respawn_penalty: float = -2.0, reset_seed_mode: str = "random", seed: int = 0):
        super().__init__(env)
        self.respawn_penalty = float(respawn_penalty)
        self.reset_seed_mode = str(reset_seed_mode)
        self._rng = np.random.default_rng(seed)

    def reset(self, *, seed: Optional[int] = None, options: Optional[dict] = None):
        if seed is not None:
            self._rng = np.random.default_rng(seed)
        return self.env.reset(seed=seed, options=options)

    def step(self, action):
        obs, reward, terminated, truncated, info = self.env.step(action)
        if terminated:
            info = dict(info)
            info["respawned"] = True
            reward = float(reward) + self.respawn_penalty
            seed = int(self._rng.integers(0, 10**9)) if self.reset_seed_mode == "random" else None
            obs, _ = self.env.reset(seed=seed)
            terminated = False
            truncated = False
        return obs, float(reward), bool(terminated), bool(truncated), info


def make_env(
    render_mode: Optional[str] = None,
    seed: int = 0,
    max_episode_steps: int = MAX_EPISODE_STEPS,
    start_down: bool = True,
    edge_respawn: bool = True,
    respawn_penalty: float = -2.0,
):
    env = ContinuousCartPoleSwingUpEnv(render_mode=render_mode, start_down=start_down)
    env = TimeLimit(env, max_episode_steps=max_episode_steps)
    if edge_respawn:
        env = EdgeRespawnWrapper(env, respawn_penalty=respawn_penalty, seed=seed)
    env.reset(seed=seed)
    return env
