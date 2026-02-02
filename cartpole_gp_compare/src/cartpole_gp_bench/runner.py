# runner.py

import time
import numpy as np

from cartpole_env import obs_to_state, wrap_pi, state_to_features
from mppi import MPPIParams, CostParams, ExploreParams, mppi_plan, shift_u, is_success


def _delta_state(s_next, s):
    dx = s_next[0] - s[0]
    dxd = s_next[1] - s[1]
    dth = wrap_pi(s_next[2] - s[2])
    dthd = s_next[3] - s[3]
    return np.array([dx, dxd, dth, dthd], dtype=np.float64)


def collect_random_init(env, n_steps: int, seed: int = 0):
    rng = np.random.default_rng(seed)
    obs, _ = env.reset(seed=seed)
    s = np.array(obs_to_state(obs), dtype=np.float64)

    Xs, Ys = [], []
    x_init = float(s[0])

    for _ in range(int(n_steps)):
        u = float(rng.uniform(-1.0, 1.0))
        obs2, r, term, trunc, info = env.step(np.array([u], dtype=np.float32))
        s2 = np.array(obs_to_state(obs2), dtype=np.float64)

        Xs.append(state_to_features(s[0], s[1], s[2], s[3], u))
        Ys.append(_delta_state(s2, s))

        if term or trunc:
            obs2, _ = env.reset()
            s2 = np.array(obs_to_state(obs2), dtype=np.float64)

        s = s2

    X = np.stack(Xs, axis=0)
    Y = np.stack(Ys, axis=0)
    return X, Y, x_init


def _maybe_rebuild_local(model, state, u_seq):
    if hasattr(model, "rebuild_local"):
        model.rebuild_local(state, u_seq)


def _maybe_unc_fn(model):
    if hasattr(model, "get_uncertainty_fn"):
        return model.get_uncertainty_fn()
    return None


def run_mppi_retrain(
    env,
    model,
    *,
    seed: int = 0,
    init_random_steps: int = 300,
    max_steps: int = 800,
    update_every: int = 10,
    update_batch: int = 64,
    hold_steps: int = 25,
    mppi_params: MPPIParams = MPPIParams(),
    cost_params: CostParams = CostParams(),
    explore_params: ExploreParams = ExploreParams(),
):
    rng = np.random.default_rng(seed)

    # identical init dataset for all models
    X0, Y0, x_init = collect_random_init(env, n_steps=init_random_steps, seed=seed)
    model.fit_init(X0, Y0)

    obs, _ = env.reset(seed=seed + 1)
    s = np.array(obs_to_state(obs), dtype=np.float64)

    u_seq = np.zeros((mppi_params.horizon,), dtype=np.float64)

    # PALSGP: build local predictor once after init, exactly like your notebook pattern
    _maybe_rebuild_local(model, s, u_seq)

    bufX, bufY = [], []
    success_count = 0
    t_global = 0

    t_start = time.perf_counter()
    total_pred_ms = 0.0

    for step in range(int(max_steps)):
        unc_fn = _maybe_unc_fn(model)

        u0, u_seq_new, tubeX, info = mppi_plan(
            s,
            x_init=x_init,
            predict_fn=model.predict,
            mppi=mppi_params,
            cp=cost_params,
            u_init=u_seq,
            rng=rng,
            t_global=t_global,
            unc_fn=unc_fn,
            ep=explore_params,
        )
        total_pred_ms += float(info.get("pred_ms", 0.0))

        obs2, r, term, trunc, inf = env.step(np.array([u0], dtype=np.float32))
        s2 = np.array(obs_to_state(obs2), dtype=np.float64)

        bufX.append(state_to_features(s[0], s[1], s[2], s[3], float(u0)))
        bufY.append(_delta_state(s2, s))

        s = s2
        u_seq = shift_u(u_seq_new)
        t_global += 1

        if is_success(obs2, x_init=x_init, cp=cost_params):
            success_count += 1
        else:
            success_count = 0

        if success_count >= int(hold_steps):
            wall = time.perf_counter() - t_start
            return {
                "success": True,
                "steps": step + 1,
                "wall_s": float(wall),
                "mppi_pred_ms_total": float(total_pred_ms),
                "model_stats": model.stats(),
            }

        if term or trunc:
            obs, _ = env.reset()
            s = np.array(obs_to_state(obs), dtype=np.float64)
            success_count = 0
            # keep u_seq as-is (same across models); or reset if you prefer:
            # u_seq[:] = 0.0

        # identical retrain schedule for ALL models
        if (t_global % int(update_every) == 0) and (len(bufX) >= int(update_batch)):
            Xb = np.stack(bufX[-update_batch:], axis=0)
            Yb = np.stack(bufY[-update_batch:], axis=0)
            model.update(Xb, Yb)

            # PALSGP: rebuild local predictor only after updates (not every step)
            _maybe_rebuild_local(model, s, u_seq)

    wall = time.perf_counter() - t_start
    return {
        "success": False,
        "steps": int(max_steps),
        "wall_s": float(wall),
        "mppi_pred_ms_total": float(total_pred_ms),
        "model_stats": model.stats(),
    }
