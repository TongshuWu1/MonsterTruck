# mppi.py

from __future__ import annotations

from dataclasses import dataclass
from typing import Callable, Dict, Optional, Tuple

import numpy as np

from cartpole_env import U_MIN, U_MAX, wrap_pi


@dataclass
class MPPIParams:
    horizon: int = 40
    K: int = 96
    sigma: float = 0.6
    lam: float = 1.0


@dataclass
class CostParams:
    upright_w: float = 2.0
    center_w: float = 1.0
    u_w: float = 0.005
    upright_cos: float = 0.85
    x_band: float = 0.70


@dataclass
class ExploreParams:
    explore_steps: int = 200
    unc_w_max: float = 5.0
    unc_w_min: float = 0.0


def batch_state_to_features(
    S: np.ndarray,
    U: np.ndarray,
    *,
    x_scale: float = 2.4,
    v_scale: float = 3.0,
    w_scale: float = 8.0,
) -> np.ndarray:
    S = np.asarray(S, dtype=np.float64)
    U = np.asarray(U, dtype=np.float64).reshape(-1)
    x = S[:, 0]
    xdot = S[:, 1]
    th = S[:, 2]
    thdot = S[:, 3]
    X = np.empty((S.shape[0], 6), dtype=np.float64)
    X[:, 0] = np.tanh(x / x_scale)
    X[:, 1] = np.tanh(xdot / v_scale)
    X[:, 2] = np.sin(th)
    X[:, 3] = np.cos(th)
    X[:, 4] = np.tanh(thdot / w_scale)
    X[:, 5] = np.clip(U, U_MIN, U_MAX)
    return X


def exploration_weight(t_global: int, p: ExploreParams) -> float:
    if p.explore_steps <= 0:
        return float(p.unc_w_min)
    a = np.clip(1.0 - float(t_global) / float(p.explore_steps), 0.0, 1.0)
    return float(p.unc_w_min + (p.unc_w_max - p.unc_w_min) * a)


def stage_cost_cartpole(
    S: np.ndarray,
    U: np.ndarray,
    *,
    x_init: float,
    cp: CostParams,
    unc_bonus: Optional[np.ndarray] = None,
    unc_w: float = 0.0,
) -> np.ndarray:
    th = S[:, 2]
    x = S[:, 0]
    c = (
        cp.upright_w * (1.0 - np.cos(th))
        + cp.center_w * ((x - float(x_init)) ** 2)
        + cp.u_w * (np.asarray(U, dtype=np.float64) ** 2)
    )
    if (unc_bonus is not None) and (unc_w > 0.0):
        c = c - float(unc_w) * np.asarray(unc_bonus, dtype=np.float64)
    return c


def terminal_cost_hold_like(S: np.ndarray, *, x_init: float, cp: CostParams) -> np.ndarray:
    th = S[:, 2]
    x = S[:, 0]
    cT = np.zeros_like(th, dtype=np.float64)
    good = (np.cos(th) >= cp.upright_cos) & (np.abs(x - float(x_init)) <= cp.x_band)
    cT[good] -= 5.0
    return cT


def is_success(obs: np.ndarray, *, x_init: float, cp: CostParams) -> bool:
    x = float(obs[0])
    th = float(obs[2])
    return (np.cos(th) >= cp.upright_cos) and (abs(x - float(x_init)) <= cp.x_band)


def dynamics_step_batch(
    S: np.ndarray,
    U: np.ndarray,
    predict_fn: Callable[[np.ndarray], Tuple[np.ndarray, Optional[np.ndarray]]],
) -> Tuple[np.ndarray, np.ndarray, Optional[np.ndarray]]:
    Xfeat = batch_state_to_features(S, U)
    mu, var = predict_fn(Xfeat)  # mu: (B,4)
    mu = np.asarray(mu, dtype=np.float64)
    S2 = np.empty_like(S, dtype=np.float64)
    S2[:, 0] = S[:, 0] + mu[:, 0]
    S2[:, 1] = S[:, 1] + mu[:, 1]
    S2[:, 2] = np.vectorize(wrap_pi)(S[:, 2] + mu[:, 2])
    S2[:, 3] = S[:, 3] + mu[:, 3]
    return S2, Xfeat, var


def rollout_tube_features(
    state: np.ndarray,
    u_seq: np.ndarray,
    predict_fn: Callable[[np.ndarray], Tuple[np.ndarray, Optional[np.ndarray]]],
) -> np.ndarray:
    state = np.asarray(state, dtype=np.float64).reshape(4,)
    u_seq = np.asarray(u_seq, dtype=np.float64).reshape(-1)
    H = u_seq.shape[0]
    tubeX = np.empty((H, 6), dtype=np.float64)

    s = state.copy()
    for t in range(H):
        u = float(np.clip(u_seq[t], U_MIN, U_MAX))
        X = batch_state_to_features(s[None, :], np.array([u], dtype=np.float64))
        mu, _ = predict_fn(X)
        mu = np.asarray(mu, dtype=np.float64).reshape(4,)
        tubeX[t] = X[0]
        s[0] = s[0] + mu[0]
        s[1] = s[1] + mu[1]
        s[2] = wrap_pi(s[2] + mu[2])
        s[3] = s[3] + mu[3]
    return tubeX


def mppi_plan(
    state: np.ndarray,
    *,
    x_init: float,
    predict_fn: Callable[[np.ndarray], Tuple[np.ndarray, Optional[np.ndarray]]],
    mppi: MPPIParams,
    cp: CostParams,
    u_init: Optional[np.ndarray] = None,
    rng: Optional[np.random.Generator] = None,
    t_global: int = 0,
    unc_fn: Optional[Callable[[np.ndarray], np.ndarray]] = None,
    ep: Optional[ExploreParams] = None,
) -> Tuple[float, np.ndarray, np.ndarray, Dict[str, float]]:
    if rng is None:
        rng = np.random.default_rng(0)
    if ep is None:
        ep = ExploreParams()

    state = np.asarray(state, dtype=np.float64).reshape(4,)
    H = int(mppi.horizon)
    K = int(mppi.K)

    u_mean = np.zeros((H,), dtype=np.float64) if u_init is None else np.asarray(u_init, dtype=np.float64).reshape(H,).copy()

    eps = rng.normal(0.0, float(mppi.sigma), size=(K, H)).astype(np.float64)
    U = np.clip(u_mean[None, :] + eps, U_MIN, U_MAX)

    S = np.repeat(state[None, :], K, axis=0).astype(np.float64)
    total_cost = np.zeros((K,), dtype=np.float64)

    unc_w = exploration_weight(int(t_global), ep)
    pred_ms = 0.0

    for tt in range(H):
        Ut = U[:, tt]
        t0 = time_now()
        S2, Xfeat, _ = dynamics_step_batch(S, Ut, predict_fn)
        pred_ms += (time_now() - t0) * 1000.0

        unc = unc_fn(Xfeat) if (unc_fn is not None) else None
        total_cost += stage_cost_cartpole(S, Ut, x_init=float(x_init), cp=cp, unc_bonus=unc, unc_w=unc_w)
        S = S2

    total_cost += terminal_cost_hold_like(S, x_init=float(x_init), cp=cp)

    cmin = float(np.min(total_cost))
    w = np.exp(-(total_cost - cmin) / float(mppi.lam))
    wsum = float(np.sum(w)) + 1e-12

    u_mean = u_mean + (w[:, None] * eps).sum(axis=0) / wsum
    u_mean = np.clip(u_mean, U_MIN, U_MAX)

    tubeX = rollout_tube_features(state, u_mean, predict_fn)

    info = {
        "pred_ms": float(pred_ms),
        "unc_w": float(unc_w),
        "cost_min": float(np.min(total_cost)),
        "cost_mean": float(np.mean(total_cost)),
    }
    return float(u_mean[0]), u_mean, tubeX, info


def shift_u(u_seq: np.ndarray) -> np.ndarray:
    u_seq = np.asarray(u_seq, dtype=np.float64).reshape(-1)
    if u_seq.size == 0:
        return u_seq
    out = np.empty_like(u_seq)
    out[:-1] = u_seq[1:]
    out[-1] = 0.0
    return out


def time_now() -> float:
    import time
    return time.perf_counter()
