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
#   - Edit PROJECT_CONFIG below as the single source of truth for everything
#     except the environment module files themselves.
# ===========================

# ============================================================
# Single config (edit this) — everything except environment dynamics/reward code
# ============================================================
PROJECT_CONFIG = {
    "meta": {
        "method_order_default": ["PALSGP", "SVGP_GLOBAL", "OSGPR_GLOBAL", "EXACTGP_GLOBAL"],
    },

    "suite": {
        # Process / paths
        "python_unbuffered": True,
        "output_root": "outputs",
        "runs_subdir": "run_artifacts",
        "plots_subdir": "plots",
        "worker_tmp_subdir": "_worker_tmp",
        "stop_on_first_worker_failure": True,

        # What to run (launcher defaults)
        "env_module": "envs/cartpole_swingup/env.py",  # fallback; suite_config.py prefers ENV_NAME mapping below
        "methods": ["PALSGP", "SVGP_GLOBAL", "OSGPR_GLOBAL"],
        "n_runs": 1,
        "episodes_per_run": 2,
        "max_steps_per_ep": 600,
        "seed_base": 0,

        # Live progress / render cadence (forwarded into workers)
        "live_render": True,
        "live_every_steps": 50,
        "progress_every_steps": 50,
        "live_only_first_ep": False,
        "live_only_first_run": True,

        # Dashboard mode: 'unified' | 'per_worker' | 'off'
        "dashboard_mode": "unified",

        # Unified dashboard window (used when dashboard_mode == 'unified')
        "dashboard_window_size": (1600, 980),
        "dashboard_caption": "GP Comparison Dashboard",
    },

    "runtime_defaults": {
        # Worker-overridden runtime defaults (experiments.py standalone fallbacks)
        "live_render": True,
        "live_every_steps": 50,
        "progress_every_steps": 50,
        "live_only_first_ep": False,
        "live_only_first_run": True,
        "live_min_dt": 0.05,
        "live_size": (720, 450),

        "seed_base": 0,
        "n_runs": 1,
        "episodes_per_run": 2,
        "max_steps_per_ep": 600,
    },

    "methods": {
        "enable": {
            "PALSGP": True,
            "SVGP_GLOBAL": True,
            "OSGPR_GLOBAL": True,
            "EXACTGP_GLOBAL": True,
        },
        "order": ["PALSGP", "SVGP_GLOBAL", "OSGPR_GLOBAL", "EXACTGP_GLOBAL"],
        "show_skip_messages": False,
    },

    "mppi": {
        "horizon": 50,
        "samples": 1000,
        "sigma": 0.35,
        "lambda": 1.0,
    },

    "action_bounds": {
        "u_min": -1.0,
        "u_max": 1.0,
    },

    "online_update": {
        "update_every": 50,
        "osgpr_iters_per_update": 100,
        "osgpr_lr": 0.02,
        "noise_init": 1e-4,
    },

    "inducing": {
        "m_global_init": 256,
        "m_global_max": 256,
        "m_anchors": 16,
        "m_local": 52,
    },

    "local_rebuild": {
        "every": 20,
        "overlap_trig": 0.70,
    },

    "corridor": {
        "r0": 1.0,
        "r1": 4.0,
        "tube_invvar_eps": 1e-4,
    },

    "uncertainty": {
        "unc_lambda": 0.0,
    },

    "success": {
        "cos_th_min": 0.98,
        "x_abs_max": 0.35,
        "xdot_abs_max": 1.0,
        "thdot_abs_max": 2.5,
        "hold_steps_by_env": {
            "cartpole_swingup": 200,
        },
        "stop_on_hold_success": True,
    },

    "logging": {
        # Print/log hygiene (keep output readable in suite + workers)
        "module_banners": False,          # import-time 'Cell ready' messages
        "startup_summary": True,          # one concise config summary
        "data_bootstrap_summary": True,   # X0/targets dataset summary at startup
        "method_banner": True,            # per-method header blocks
        "method_setup": True,             # method setup / init lines
        "episode_summary": True,          # per-episode summary lines (worker-injected)
        "run_summary": True,              # per-run + method-finished summaries
        "replay_messages": True,          # replay/no-frames messages
        "update_events": False,           # per-update timing lines (very noisy)
        "update_event_every": 10,
        "eval_tables": True,
        "suite_worker_stdout": True,
        "suite_progress_lines": False,
    },

    "svgp": {
        # Make SVGP baseline closer to a standard/fair sparse-GP baseline
        # (compared to PALSGP/OSGPR global models) rather than a very light fixed-hyperparam variant.
        "train_hypers": True,
        "train_inducing": True,      # keep shared global-Z semantics by default
        "steps_init": 250,
        "steps_per_update": 100,
        "lr": 0.02,
        "replay_train_max": 2500,
        "refit_global_each_episode": True,
        "refit_replay_max": 4000,
        "refit_steps": 250,
        "refit_lr": 0.02,
    },

    "exactgp": {
        "min_mem_for_update": 200,
        "iters_init": 220,
        "iters_update": 60,
        "lr": 0.02,
        "grad_clip": 10.0,
    },
}

# ============================================================
# Environment selection (kept separate from the single config above)
# ============================================================
# Change ENV_NAME to switch tasks/environments without touching the launcher.
ENV_NAME = "cartpole_swingup"   # choices: "cartpole_swingup"
ENV_MODULE_BY_NAME = {
    "cartpole_swingup": "envs/cartpole_swingup/env.py",
}
ENV_MODULE = ENV_MODULE_BY_NAME.get(ENV_NAME, ENV_NAME)

import numpy as np
import tensorflow as tf
import time
from collections import defaultdict

# ============================================================
# Backward-compatible flat aliases (experiments.py still uses these names)
# ============================================================
_METHODS = dict(PROJECT_CONFIG.get("methods", {}))
_ENABLE = dict(_METHODS.get("enable", {})) if isinstance(_METHODS.get("enable", {}), dict) else {}
_SUITE = dict(PROJECT_CONFIG.get("suite", {}))
_RT = dict(PROJECT_CONFIG.get("runtime_defaults", {}))
_MPPI = dict(PROJECT_CONFIG.get("mppi", {}))
_AB = dict(PROJECT_CONFIG.get("action_bounds", {}))
_UPD = dict(PROJECT_CONFIG.get("online_update", {}))
_IND = dict(PROJECT_CONFIG.get("inducing", {}))
_LOC = dict(PROJECT_CONFIG.get("local_rebuild", {}))
_COR = dict(PROJECT_CONFIG.get("corridor", {}))
_UNC = dict(PROJECT_CONFIG.get("uncertainty", {}))
_SUC = dict(PROJECT_CONFIG.get("success", {}))
_EXA = dict(PROJECT_CONFIG.get("exactgp", {}))
_LOG = dict(PROJECT_CONFIG.get("logging", {}))
_SVG = dict(PROJECT_CONFIG.get("svgp", {}))

METHOD_ORDER_DEFAULT = list(PROJECT_CONFIG.get("meta", {}).get("method_order_default", ["PALSGP", "SVGP_GLOBAL", "OSGPR_GLOBAL", "EXACTGP_GLOBAL"]))
SUITE_CONFIG = dict(_SUITE)

# Worker-overridden runtime defaults (run_suite.py -> worker patches these before execution)
LIVE_RENDER = bool(_RT.get("live_render", True))  # default only; usually injected by worker
LIVE_EVERY_STEPS = int(_RT.get("live_every_steps", 50))
PROGRESS_EVERY_STEPS = int(_RT.get("progress_every_steps", 50))
LIVE_ONLY_FIRST_EP = bool(_RT.get("live_only_first_ep", False))
LIVE_ONLY_FIRST_RUN = bool(_RT.get("live_only_first_run", True))
LIVE_MIN_DT = float(_RT.get("live_min_dt", 0.05))
LIVE_SIZE = tuple(_RT.get("live_size", (720, 450)))

SEED_BASE = 0
N_RUNS = 1
N_EPISODES_PER_RUN = 2
MAX_STEPS_PER_EP = 600

# Method toggles (launcher + experiments use these; single source of truth)
ENABLE_PALSGP = bool(_ENABLE.get("PALSGP", True))
ENABLE_SVGP_GLOBAL = bool(_ENABLE.get("SVGP_GLOBAL", True))
ENABLE_OSGPR_GLOBAL = bool(_ENABLE.get("OSGPR_GLOBAL", True))
ENABLE_EXACTGP_GLOBAL = bool(_ENABLE.get("EXACTGP_GLOBAL", False))
METHOD_ORDER = [str(m).strip().upper() for m in _METHODS.get("order", METHOD_ORDER_DEFAULT) if str(m).strip()]
SHOW_SKIP_MESSAGES = bool(_METHODS.get("show_skip_messages", False))

# MPPI
H = int(_MPPI.get("horizon", 45))
K = int(_MPPI.get("samples", 2048))
MPPI_SIGMA = float(_MPPI.get("sigma", 0.35))
MPPI_LAMBDA = float(_MPPI.get("lambda", 1.0))

U_MIN = float(_AB.get("u_min", -1.0))
U_MAX = float(_AB.get("u_max", 1.0))

# Online update schedule
UPDATE_EVERY = int(_UPD.get("update_every", 50))
OSGPR_ITERS_PER_UPDATE = int(_UPD.get("osgpr_iters_per_update", 100))
OSGPR_LR = float(_UPD.get("osgpr_lr", 0.02))
NOISE_INIT = float(_UPD.get("noise_init", 1e-4))

# SVGP baseline knobs (fairer/more standard baseline controls)
SVGP_TRAIN_HYPERS = bool(_SVG.get("train_hypers", True))
SVGP_TRAIN_INDUCING = bool(_SVG.get("train_inducing", False))
SVGP_STEPS_INIT = int(_SVG.get("steps_init", 250))
SVGP_STEPS_PER_UPDATE = int(_SVG.get("steps_per_update", max(1, OSGPR_ITERS_PER_UPDATE)))
SVGP_LR = float(_SVG.get("lr", OSGPR_LR))
SVGP_REPLAY_TRAIN_MAX = int(_SVG.get("replay_train_max", 2500))
SVGP_REFIT_GLOBAL_EACH_EP = bool(_SVG.get("refit_global_each_episode", True))
SVGP_REFIT_REPLAY_MAX = int(_SVG.get("refit_replay_max", 4000))
SVGP_REFIT_STEPS = int(_SVG.get("refit_steps", 250))
SVGP_REFIT_LR = float(_SVG.get("refit_lr", SVGP_LR))

# Inducing / local subset sizes
M_GLOBAL_INIT = int(_IND.get("m_global_init", 256))
M_GLOBAL_MAX  = int(_IND.get("m_global_max", 256))
M_ANCHORS     = int(_IND.get("m_anchors", 16))
M_LOCAL       = int(_IND.get("m_local", 52))

# Local rebuild controls (PALSGP)
LOCAL_REBUILD_EVERY = int(_LOC.get("every", 20))
LOCAL_OVERLAP_TRIG  = float(_LOC.get("overlap_trig", 0.70))

# Corridor thresholds (whitened squared distance)
CORR_R0 = float(_COR.get("r0", 1.0))
CORR_R1 = float(_COR.get("r1", 4.0))
TUBE_INVVAR_EPS = float(_COR.get("tube_invvar_eps", 1e-4))

# Uncertainty penalty in MPPI (set 0.0 to disable)
UNC_LAMBDA = float(_UNC.get("unc_lambda", 0.0))

# Success thresholds (used in run loops)
SUCCESS_COS_TH_MIN = float(_SUC.get("cos_th_min", 0.98))
SUCCESS_X_ABS_MAX = float(_SUC.get("x_abs_max", 0.35))
SUCCESS_XDOT_ABS_MAX = float(_SUC.get("xdot_abs_max", 1.0))
SUCCESS_THDOT_ABS_MAX = float(_SUC.get("thdot_abs_max", 2.5))
SUCCESS_HOLD_STEPS_BY_ENV = dict(_SUC.get("hold_steps_by_env", {"cartpole_swingup": 200}))
SUCCESS_HOLD_STEPS = int(SUCCESS_HOLD_STEPS_BY_ENV.get(ENV_NAME, 200))
STOP_ON_HOLD_SUCCESS = bool(_SUC.get("stop_on_hold_success", True))

# Exact GP baseline (full replay exact retraining)
EXACT_MIN_MEM_FOR_UPDATE = int(_EXA.get("min_mem_for_update", 200))
EXACT_ITERS_INIT = int(_EXA.get("iters_init", 220))
EXACT_ITERS_UPDATE = int(_EXA.get("iters_update", 60))
EXACT_LR = float(_EXA.get("lr", 0.02))
EXACT_GRAD_CLIP = float(_EXA.get("grad_clip", 10.0))

# Logging toggles + helpers
LOG_MODULE_BANNERS = bool(_LOG.get("module_banners", False))
LOG_STARTUP_SUMMARY = bool(_LOG.get("startup_summary", True))
LOG_DATA_BOOTSTRAP_SUMMARY = bool(_LOG.get("data_bootstrap_summary", True))
LOG_METHOD_BANNER = bool(_LOG.get("method_banner", True))
LOG_METHOD_SETUP = bool(_LOG.get("method_setup", True))
LOG_EPISODE_SUMMARY = bool(_LOG.get("episode_summary", True))
LOG_RUN_SUMMARY = bool(_LOG.get("run_summary", True))
LOG_REPLAY_MESSAGES = bool(_LOG.get("replay_messages", True))
LOG_UPDATE_EVENTS = bool(_LOG.get("update_events", False))
LOG_UPDATE_EVENT_EVERY = int(_LOG.get("update_event_every", 10))
LOG_EVAL_TABLES = bool(_LOG.get("eval_tables", True))
LOG_SUITE_WORKER_STDOUT = bool(_LOG.get("suite_worker_stdout", True))
LOG_SUITE_PROGRESS_LINES = bool(_LOG.get("suite_progress_lines", False))


def _log_flag(category, default=True):
    key = str(category or "").strip().lower()
    mapping = {
        "module_banners": LOG_MODULE_BANNERS,
        "startup_summary": LOG_STARTUP_SUMMARY,
        "data_bootstrap_summary": LOG_DATA_BOOTSTRAP_SUMMARY,
        "method_banner": LOG_METHOD_BANNER,
        "method_setup": LOG_METHOD_SETUP,
        "episode_summary": LOG_EPISODE_SUMMARY,
        "run_summary": LOG_RUN_SUMMARY,
        "replay_messages": LOG_REPLAY_MESSAGES,
        "update_events": LOG_UPDATE_EVENTS,
        "eval_tables": LOG_EVAL_TABLES,
    }
    return bool(mapping.get(key, default))


def log_print(category, *args, **kwargs):
    if _log_flag(category, default=True):
        print(*args, **kwargs)


def log_method_banner(method_name, hold_steps=None):
    if not _log_flag("method_banner", True):
        return
    if hold_steps is None:
        print(f"\n=== {method_name} ===")
    else:
        print(f"\n=== {method_name} | hold={int(hold_steps)} ===")


def should_log_update_event(update_idx=None):
    if not _log_flag("update_events", False):
        return False
    if update_idx is None:
        return True
    try:
        return (int(update_idx) % int(LOG_UPDATE_EVENT_EVERY)) == 0
    except Exception:
        return True

if LOG_STARTUP_SUMMARY:
    print("[config] loaded engine/config.py (PROJECT_CONFIG + legacy aliases)")
    print(f"[config] env={ENV_NAME} module={ENV_MODULE} | runs={N_RUNS} ep/run={N_EPISODES_PER_RUN} max_steps={MAX_STEPS_PER_EP}")
    print(f"[config] methods PALSGP={ENABLE_PALSGP} SVGP={ENABLE_SVGP_GLOBAL} OSGPR={ENABLE_OSGPR_GLOBAL} EXACT={ENABLE_EXACTGP_GLOBAL} | order={METHOD_ORDER}")
    print(f"[config] mppi H={H} K={K} sigma={MPPI_SIGMA} lambda={MPPI_LAMBDA} | update_every={UPDATE_EVERY} iters={OSGPR_ITERS_PER_UPDATE}")
    print(f"[config] inducing init/max={M_GLOBAL_INIT}/{M_GLOBAL_MAX} anchors={M_ANCHORS} local={M_LOCAL} | live_every={LIVE_EVERY_STEPS} prog_every={PROGRESS_EVERY_STEPS}")

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

log_print("module_banners", "✅ Eval helpers + registry ready. (EVAL_REGISTRY will be filled by Cells 6/7/8)")

# worker-injected runtime knobs
LIVE_RENDER = True
LIVE_ONLY_FIRST_RUN = True
LIVE_ONLY_FIRST_EP = False
LIVE_EVERY_STEPS = 50
PROGRESS_EVERY_STEPS = 50
