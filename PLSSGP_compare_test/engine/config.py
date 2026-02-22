# ===========================
# Cell 5 — Experiment config + Eval metrics + Registry helpers
#
# Purpose:
#   - Define global knobs shared by ALL methods (PALSGP / OSGPR_GLOBAL / SVGP_GLOBAL)
#   - Define richer evaluation metrics you asked for
#   - Provide a consistent EVAL_REGISTRY schema used by Cell 9 plotting
#
# Notes:
#   - Env obs/action can stay float32; GP features & TF compute are float64.
#   - Methods (Cells 6/7/8) should log into this registry using the helpers below.
# ===========================

import numpy as np
import tensorflow as tf
import time
from collections import defaultdict

# ============================================================
# Global experiment knobs (shared)
# ============================================================
# Change ENV_NAME to switch tasks/environments without touching the launcher.
ENV_NAME = "cartpole_swingup"   # choices: "cartpole_swingup", "mountaincar_hold"
ENV_MODULE_BY_NAME = {
    "cartpole_swingup": "envs/cartpole_swingup/env.py",
    "mountaincar_hold": "envs/mountaincar_hold/env.py",
}
ENV_MODULE = ENV_MODULE_BY_NAME.get(ENV_NAME, ENV_NAME)

# Worker-overridden runtime defaults (run_suite.py -> worker patches these before execution)
LIVE_RENDER = True  # default only; usually injected by worker
LIVE_EVERY_STEPS = 50
PROGRESS_EVERY_STEPS = 50
LIVE_ONLY_FIRST_EP = False
LIVE_ONLY_FIRST_RUN = True

SEED_BASE = 0  # default only; overridden per worker
N_RUNS = 1
N_EPISODES_PER_RUN = 2
MAX_STEPS_PER_EP = 600  # should match MAX_EPISODE_STEPS

# Method toggles (launcher reads these; single source of truth)
ENABLE_PALSGP = True
ENABLE_SVGP_GLOBAL = True
ENABLE_OSGPR_GLOBAL = True
ENABLE_EXACTGP_GLOBAL = False
METHOD_ORDER = ["PALSGP", "SVGP_GLOBAL", "OSGPR_GLOBAL", "EXACTGP_GLOBAL"]
SHOW_SKIP_MESSAGES = False

# MPPI
H = 45
K = 2048
MPPI_SIGMA = 0.35
MPPI_LAMBDA = 1.0

U_MIN, U_MAX = -1.0, 1.0

# Online update schedule
UPDATE_EVERY = 50
OSGPR_ITERS_PER_UPDATE = 100
OSGPR_LR = 0.02
NOISE_INIT = 1e-4

# Inducing / local subset sizes
M_GLOBAL_INIT = 256
M_GLOBAL_MAX  = 256
M_ANCHORS     = 16
M_LOCAL       = 52

# Local rebuild controls (PALSGP)
LOCAL_REBUILD_EVERY = 20          # force rebuild at least this often
LOCAL_OVERLAP_TRIG  = 0.70        # rebuild if overlap < threshold

# Corridor thresholds (whitened squared distance)
CORR_R0 = 1.0
CORR_R1 = 4.0
TUBE_INVVAR_EPS = 1e-4

# Uncertainty penalty in MPPI (set 0.0 to disable)
UNC_LAMBDA = 0.0

# Success thresholds (used in run loops)
SUCCESS_COS_TH_MIN = 0.98
SUCCESS_X_ABS_MAX = 0.35
SUCCESS_XDOT_ABS_MAX = 1.0
SUCCESS_THDOT_ABS_MAX = 2.5
SUCCESS_HOLD_STEPS_BY_ENV = {
    "cartpole_swingup": 200,
    "mountaincar_hold": 100,
}
SUCCESS_HOLD_STEPS = int(SUCCESS_HOLD_STEPS_BY_ENV.get(ENV_NAME, 200))

# Exact GP baseline (full replay exact retraining)
EXACT_MIN_MEM_FOR_UPDATE = 200
EXACT_ITERS_INIT = 220
EXACT_ITERS_UPDATE = 60
EXACT_LR = 0.02
EXACT_GRAD_CLIP = 10.0


print("✅ Cell 5 config loaded.")
print(f"Env: ENV_NAME={ENV_NAME}, ENV_MODULE={ENV_MODULE}")
print(f"N_RUNS={N_RUNS}, EP/RUN={N_EPISODES_PER_RUN}, MAX_STEPS={MAX_STEPS_PER_EP}")
print(f"Methods: PALSGP={ENABLE_PALSGP}, SVGP={ENABLE_SVGP_GLOBAL}, OSGPR={ENABLE_OSGPR_GLOBAL}, EXACT={ENABLE_EXACTGP_GLOBAL}")
print(f"MPPI: H={H}, K={K}, sigma={MPPI_SIGMA}, lambda={MPPI_LAMBDA}")
print(f"Update: every {UPDATE_EVERY} steps, iters={OSGPR_ITERS_PER_UPDATE}, lr={OSGPR_LR}")
print(f"Inducing: M_INIT={M_GLOBAL_INIT}, M_MAX={M_GLOBAL_MAX}, anchors={M_ANCHORS}, local={M_LOCAL}")

# ============================================================
# Numeric helpers
# ============================================================
def rmse_np(a, b):
    a = np.asarray(a).reshape(-1)
    b = np.asarray(b).reshape(-1)
    return float(np.sqrt(np.mean((a - b) ** 2)))

def running_average(x):
    x = np.asarray(x, dtype=np.float64)
    if x.size == 0:
        return x
    cs = np.cumsum(x)
    return cs / (np.arange(len(x)) + 1.0)

def safe_mean(x):
    x = np.asarray(x, dtype=np.float64)
    return float(np.nanmean(x)) if x.size else float("nan")

# ============================================================
# Batch eval (valuable metrics): RMSE + NLPD + 2σ coverage + mean std
# ============================================================
def _likelihood_var(model):
    try:
        return float(model.likelihood.variance.numpy())
    except Exception:
        return 0.0

def eval_regression_batch(model, X, y, add_likelihood_var=True, eps=1e-12):
    """
    Evaluate a GP regression model on batch:
      rmse, nlpd, cover2 (|err|<=2σ), std_mean, var_mean
    """
    X = np.asarray(X, dtype=np.float64)
    y = np.asarray(y, dtype=np.float64).reshape(-1)

    Xt = tf.convert_to_tensor(X, dtype=tf.float64)

    if hasattr(model, "predict_f_cached"):
        mu_tf, var_tf = model.predict_f_cached(Xt, full_cov=False)
    else:
        mu_tf, var_tf = model.predict_f(Xt, full_cov=False)

    mu = mu_tf.numpy().reshape(-1)
    var = var_tf.numpy().reshape(-1)

    lik = _likelihood_var(model) if add_likelihood_var else 0.0
    var_y = np.maximum(var + lik, eps)
    std_y = np.sqrt(var_y)

    err = mu - y
    rmse = float(np.sqrt(np.mean(err * err)))
    nlpd = float(np.mean(0.5*np.log(2.0*np.pi*var_y) + 0.5*(err*err)/var_y))
    cover2 = float(np.mean(np.abs(err) <= 2.0*std_y))

    return dict(
        rmse=rmse,
        nlpd=nlpd,
        cover2=cover2,
        std_mean=float(np.mean(std_y)),
        var_mean=float(np.mean(var_y)),
    )

def eval_multihead_batch(models, X, ys, add_likelihood_var=True):
    mets = [eval_regression_batch(m, X, y, add_likelihood_var=add_likelihood_var) for m, y in zip(models, ys)]
    out = {}
    for k in mets[0].keys():
        out[k + "_mean"] = float(np.mean([d[k] for d in mets]))
        out[k + "_std"]  = float(np.std([d[k] for d in mets]))
    out["rmse_heads"]   = [float(d["rmse"]) for d in mets]
    out["nlpd_heads"]   = [float(d["nlpd"]) for d in mets]
    out["cover2_heads"] = [float(d["cover2"]) for d in mets]
    return out

# ============================================================
# EVAL_REGISTRY schema + helpers
# ============================================================
def make_empty_method_registry(method_name):
    """
    EVAL_REGISTRY[method] = {
      "method": str,
      "run_stats": [dict per run],
      "run_traces": [dict per run],
      # optional aggregated time-series (filled by finalize_method_registry)
      "running_avg_wall_per_step": np.ndarray,
      "pred_time_mean": np.ndarray,
      "u_abs_mean": np.ndarray,
      "unc_mean": np.ndarray,
      ...
    }
    """
    return dict(
        method=method_name,
        run_stats=[],
        run_traces=[],
    )

EVAL_REGISTRY = {}

def registry_get(method):
    if method not in EVAL_REGISTRY:
        EVAL_REGISTRY[method] = make_empty_method_registry(method)
    return EVAL_REGISTRY[method]

def registry_add_run(method, run_stats, run_traces):
    """
    run_stats: dict (scalar summaries + per-episode arrays)
    run_traces: dict (per-step arrays + update-event arrays)
    """
    reg = registry_get(method)
    reg["run_stats"].append(run_stats)
    reg["run_traces"].append(run_traces)

def _pad_to_length(arrs, L, fill=np.nan):
    mats = []
    for a in arrs:
        a = np.asarray(a, dtype=np.float64).reshape(-1)
        b = np.full((L,), fill, dtype=np.float64)
        n = min(L, len(a))
        if n > 0:
            b[:n] = a[:n]
        mats.append(b)
    if len(mats) == 0:
        return np.zeros((0, L), dtype=np.float64)
    return np.vstack(mats)

def finalize_method_registry(method):
    """
    Build aggregated time-series for plotting in Cell 9.
    Expects each run_trace to contain some of these keys:
      - wall_time_step, pred_time_step, train_time_step, rebuild_time_step, vis_time_step
      - u_exec_step, unc_exec_step
    """
    reg = registry_get(method)
    traces = reg["run_traces"]
    if len(traces) == 0:
        return

    # concatenate across runs for "running avg wall/step" like your original plot
    wall_all = []
    pred_all = []

    # also compute mean±std time series (aligned by timestep within run)
    # choose a conservative max len
    maxlen = 0
    for tr in traces:
        maxlen = max(maxlen, len(tr.get("pred_time_step", [])))

    # per-step mean curves (across runs, aligned)
    def mean_curve(key):
        arrs = [tr.get(key, []) for tr in traces]
        M = _pad_to_length(arrs, maxlen)
        return np.nanmean(M, axis=0)

    # concatenated (for running avg)
    for tr in traces:
        if "wall_time_step" in tr:
            wall_all.append(np.asarray(tr["wall_time_step"], dtype=np.float64))
        if "pred_time_step" in tr:
            pred_all.append(np.asarray(tr["pred_time_step"], dtype=np.float64))

    wall_cat = np.concatenate(wall_all) if len(wall_all) else np.array([], dtype=np.float64)
    pred_cat = np.concatenate(pred_all) if len(pred_all) else np.array([], dtype=np.float64)

    reg["running_avg_wall_per_step"] = running_average(wall_cat)
    reg["pred_time_mean"] = mean_curve("pred_time_step")

    # extra curves if present
    if any("u_exec_step" in tr for tr in traces):
        reg["u_abs_mean"] = mean_curve("u_exec_step")  # (already scalar u0; Cell 9 can abs if desired)
    if any("unc_exec_step" in tr for tr in traces):
        reg["unc_mean"] = mean_curve("unc_exec_step")

def finalize_all_registries():
    for m in list(EVAL_REGISTRY.keys()):
        finalize_method_registry(m)

print("✅ Eval helpers + registry ready. (EVAL_REGISTRY will be filled by Cells 6/7/8)")
