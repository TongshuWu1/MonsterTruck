import argparse
from pathlib import Path

def _exec_file(path, g):
    code = Path(path).read_text(encoding="utf-8")
    exec(compile(code, str(path), "exec"), g, g)

here = Path(__file__).resolve().parent
ap = argparse.ArgumentParser()
ap.add_argument("--env-module", default="envs/cartpole_swingup/env.py")
ap.add_argument("--skip-eval", action="store_true")
args = ap.parse_args()

g = globals()
_exec_file(here / args.env_module, g)

# ============================
# Cell 2 — Render the RANDOM collection path (and collect X,Y)  ✅ CartPole version
#
# Fix:
#   ✅ Adds render_cartpole_frame_from_state() (pure PIL) so the line:
#        frame = render_cartpole_frame_from_state(...)
#      actually works.
#
# What you get:
#   - Runs random actions for n_steps (with resets as needed)
#   - Collects executed transitions:
#       X0: (N,6)  = [x_feat, xdot_feat, sinθ, cosθ, w_feat, u]
#       Ydx0, Ydxdot0, Ydth0, Ydthdot0  (each (N,1))
#   - Records frames (rgb) and displays an inline animation (JS HTML)
#   - Plots trajectories:
#       x(t), xdot(t), theta(t), thetadot(t), action(t)
#       phase plot: x vs xdot
#
# IMPORTANT:
#   - We DO NOT rely on env.render() / pygame.
#   - We render from state using a self-contained PIL renderer below.
#   - If your env does edge respawn and sets info["respawned"]=True, we DROP that transition.
# ============================

import numpy as np
import matplotlib.pyplot as plt
from matplotlib import animation
from IPython.display import HTML, display

from PIL import Image, ImageDraw

# ------------------------------------------------------------
# Pure-PIL renderer: (x, theta) -> RGB frame
# ------------------------------------------------------------
def render_cartpole_frame_from_state(
    x, theta,
    x_threshold=2.4,
    W=720, H=450,
    cart_width=70,
    cart_height=35,
    pole_length_px=160,
    pole_width=10,
):
    """
    Minimal CartPole render (rgb uint8) from state.
    Coordinates:
      - x in [-x_threshold, x_threshold] maps to screen track
      - theta: 0 = upright, pi = down (standard)
    """
    # background
    img = Image.new("RGB", (W, H), (245, 245, 245))
    dr = ImageDraw.Draw(img)

    # track line
    y_track = int(0.72 * H)
    dr.line([(0, y_track), (W, y_track)], fill=(210, 210, 210), width=4)

    # map x -> screen px
    # keep margins so cart stays visible
    margin = 40
    x_clamped = float(np.clip(x, -x_threshold, x_threshold))
    x_norm = (x_clamped + x_threshold) / (2 * x_threshold)  # [0,1]
    cx = int(margin + x_norm * (W - 2 * margin))
    cy = y_track - cart_height // 2

    # cart rectangle
    x0 = cx - cart_width // 2
    y0 = cy - cart_height // 2
    x1 = cx + cart_width // 2
    y1 = cy + cart_height // 2
    dr.rounded_rectangle([x0, y0, x1, y1], radius=8, fill=(60, 90, 160), outline=(30, 30, 30), width=2)

    # axle point on top of cart
    ax = cx
    ay = y0 + 6

    # pole end point
    # IMPORTANT: theta=0 is upright -> pole points UP, so use -cos/sin appropriately
    # Standard cartpole uses theta from vertical; here we draw:
    #   dx =  L * sin(theta)
    #   dy = -L * cos(theta)
    dx = pole_length_px * float(np.sin(theta))
    dy = -pole_length_px * float(np.cos(theta))
    px = ax + dx
    py = ay + dy

    # pole (as thick line)
    dr.line([(ax, ay), (px, py)], fill=(180, 50, 50), width=pole_width)

    # axle circle
    r = 8
    dr.ellipse([ax - r, ay - r, ax + r, ay + r], fill=(30, 30, 30))

    return np.asarray(img, dtype=np.uint8)


# ------------------------------------------------------------
# Random collection (rendered)
# ------------------------------------------------------------
def collect_random_transitions_rendered_cartpole(
    n_steps=500,
    seed=0,
    max_episode_steps=500,
    start_down=True,
    # rendering controls
    frame_stride=2,
    resize=(720, 450),
    fps=20,
    # edge-respawn handling (relies on your Cell 1 env.step() putting info["respawned"]=True)
    edge_respawn=True,
    respawn_penalty=-2.0,
    drop_respawn_transitions=True,
    verbose=True,
):
    rng = np.random.default_rng(seed)

    env_vis = make_env(
        render_mode=None,   # we render ourselves
        seed=seed,
        max_episode_steps=max_episode_steps,
        start_down=start_down,
        edge_respawn=edge_respawn,
        respawn_penalty=respawn_penalty,
    )

    obs, _ = env_vis.reset(seed=seed)
    x, xdot, th, thdot = obs_to_state(obs)

    frames = []
    traj = dict(
        x=[], xdot=[], theta=[], thetadot=[], u=[],
        kept=0, dropped_respawn=0, resets=0
    )

    X_list = []
    Ydx_list = []
    Ydxdot_list = []
    Ydth_list = []
    Ydthdot_list = []

    for t in range(n_steps):
        # sample random control
        u = float(rng.uniform(U_MIN, U_MAX))
        act = np.array([u], dtype=np.float32)

        # store previous state (s_t)
        x0, xdot0, th0, thdot0 = x, xdot, th, thdot

        obs1, r, terminated, truncated, info = env_vis.step(act)
        x1, xdot1, th1, thdot1 = obs_to_state(obs1)

        # if edge-respawn happened, optionally drop that transition
        respawned = bool(info.get("respawned", False))
        if respawned and drop_respawn_transitions:
            traj["dropped_respawn"] += 1
        else:
            # features from s_t and executed u
            feat = state_to_features(x0, xdot0, th0, thdot0, u).reshape(1, -1)

            # delta targets (s_{t+1} - s_t)
            # NOTE: theta is wrapped already in obs_to_state; this is consistent with your pipeline
            d_x = (x1 - x0)
            d_xdot = (xdot1 - xdot0)
            d_th = wrap_pi(th1 - th0)
            d_thdot = (thdot1 - thdot0)

            X_list.append(feat)
            Ydx_list.append([d_x])
            Ydxdot_list.append([d_xdot])
            Ydth_list.append([d_th])
            Ydthdot_list.append([d_thdot])

            traj["kept"] += 1

        # store trajectory for plots (always store executed path)
        traj["x"].append(x1)
        traj["xdot"].append(xdot1)
        traj["theta"].append(th1)
        traj["thetadot"].append(thdot1)
        traj["u"].append(u)

        # render occasionally
        if (t % frame_stride) == 0:
            _custom_render = globals().get("render_state_frame_from_state", None)
            if callable(_custom_render):
                frame = _custom_render(
                    x1, th1,
                    x_threshold=2.4,
                    W=resize[0], H=resize[1],
                )
            else:
                frame = render_cartpole_frame_from_state(
                    x1, th1,
                    x_threshold=2.4,
                    W=resize[0], H=resize[1],
                )
            if frame is not None:
                frames.append(frame)

        # handle resets
        if terminated or truncated:
            traj["resets"] += 1
            obs, _ = env_vis.reset(seed=int(rng.integers(0, 10**9)))
            x, xdot, th, thdot = obs_to_state(obs)
        else:
            x, xdot, th, thdot = x1, xdot1, th1, thdot1

    env_vis.close()

    # stack outputs
    X0 = np.concatenate(X_list, axis=0).astype(np.float64)
    Ydx0 = np.asarray(Ydx_list, dtype=np.float64)
    Ydxdot0 = np.asarray(Ydxdot_list, dtype=np.float64)
    Ydth0 = np.asarray(Ydth_list, dtype=np.float64)
    Ydthdot0 = np.asarray(Ydthdot_list, dtype=np.float64)

    # ----------------------------
    # 1) animate frames
    # ----------------------------
    if len(frames) > 0:
        fig, ax = plt.subplots(figsize=(8, 5))
        im = ax.imshow(frames[0])
        ax.axis("off")

        def update(i):
            im.set_data(frames[i])
            return [im]

        ani = animation.FuncAnimation(fig, update, frames=len(frames), interval=1000.0 / fps, blit=True)
        plt.close(fig)
        display(HTML(ani.to_jshtml()))
    else:
        print("⚠️ No frames collected (check frame_stride / resize).")

    # ----------------------------
    # 2) trajectory plots
    # ----------------------------
    tgrid = np.arange(len(traj["x"]))

    plt.figure(figsize=(9, 3.8))
    plt.plot(tgrid, traj["x"], linewidth=2)
    plt.xlabel("t"); plt.ylabel("x")
    plt.title("Random collection: cart position x(t)")
    plt.grid(True, alpha=0.25); plt.show()

    plt.figure(figsize=(9, 3.8))
    plt.plot(tgrid, traj["xdot"], linewidth=2)
    plt.xlabel("t"); plt.ylabel("xdot")
    plt.title("Random collection: cart velocity xdot(t)")
    plt.grid(True, alpha=0.25); plt.show()

    plt.figure(figsize=(9, 3.8))
    plt.plot(tgrid, traj["theta"], linewidth=2)
    plt.xlabel("t"); plt.ylabel("theta (rad)")
    plt.title("Random collection: pole angle theta(t) (wrapped)")
    plt.grid(True, alpha=0.25); plt.show()

    plt.figure(figsize=(9, 3.8))
    plt.plot(tgrid, traj["thetadot"], linewidth=2)
    plt.xlabel("t"); plt.ylabel("thetadot (rad/s)")
    plt.title("Random collection: pole angular velocity thetadot(t)")
    plt.grid(True, alpha=0.25); plt.show()

    plt.figure(figsize=(9, 3.4))
    plt.plot(tgrid, traj["u"], linewidth=2)
    plt.xlabel("t"); plt.ylabel("u")
    plt.title("Random actions u(t)")
    plt.grid(True, alpha=0.25); plt.show()

    plt.figure(figsize=(6.8, 5.2))
    plt.scatter(traj["x"], traj["xdot"], s=10, alpha=0.55)
    plt.xlabel("x"); plt.ylabel("xdot")
    plt.title("Random collection path in state space (x vs xdot)")
    plt.grid(True, alpha=0.25); plt.show()

    print("Collected X0 shape:", X0.shape, " (6D features)")
    print("Targets shapes:", Ydx0.shape, Ydxdot0.shape, Ydth0.shape, Ydthdot0.shape)
    print(f"Kept={traj['kept']}  Dropped(respawn)={traj['dropped_respawn']}  Resets={traj['resets']}")

    return X0, Ydx0, Ydxdot0, Ydth0, Ydthdot0, frames, traj


# ---- run it ----
SEED = 0
X0, Ydx0, Ydxdot0, Ydth0, Ydthdot0, frames0, traj0 = collect_random_transitions_rendered_cartpole(
    n_steps=400,
    seed=SEED,
    max_episode_steps=500,
    start_down=True,
    frame_stride=2,
    resize=(720, 450),
    fps=20,
    edge_respawn=True,
    respawn_penalty=-2.0,
    drop_respawn_transitions=True,
    verbose=False,
)

# Optional: quick distribution sanity
import plotly.graph_objects as go
fig = go.Figure()
fig.add_trace(go.Histogram(x=Ydth0.flatten(), nbinsx=60, name="Δtheta"))
fig.add_trace(go.Histogram(x=Ydthdot0.flatten(), nbinsx=60, name="Δthetadot"))
fig.update_layout(
    title="Initial random dataset: distribution of Δtheta and Δtheta_dot",
    barmode="overlay",
    xaxis_title="delta value",
    yaxis_title="count",
)
fig.update_traces(opacity=0.55)
fig.show()

_exec_file(here / "gp_models.py", g)
_exec_file(here / "controllers.py", g)
_exec_file(here / "config.py", g)

# ===========================
# Cell 6 — PALSGP (MPPI + Online OSGPR-VFE global, fixed-Z during episode)
#         + Local corridor subset (k-center) + anchors
#         + Success = hold upright for SUCCESS_HOLD_STEPS consecutive steps
#
# FIXES:
#   - tubeX uses tf.TensorArray (no InaccessibleTensorError)
#   - plan_tf factory is plain Python (no "returns Function" error)
#   - reduce_retracing=True
#
# Progress: TextProgress (PyCharm-friendly)
# Live PIL + concatenated replay per run (JS animation; best in notebook UI)
# ===========================

import numpy as np
import tensorflow as tf
import gpflow
import time
import sys

from IPython.display import HTML, display
import matplotlib.pyplot as plt
from matplotlib import animation
from PIL import Image, ImageDraw

# ---------------------------
# Preconditions
# ---------------------------
for name in ["X0", "Ydx0", "Ydxdot0", "Ydth0", "Ydthdot0"]:
    assert name in globals(), f"Run Cell 2 first: missing {name}"
for name in ["OSGPR_VFE", "train_osgpr", "prior_summary", "extract_summary_from_model", "greedy_dopt_anchors_from_K"]:
    assert name in globals(), f"Run Cell 3 first: missing {name}"
for name in ["build_pack_from_model", "kcenter_farthest_first",
             "stage_cost_tf", "tf_feature_map", "tf_wrap_angle", "is_success_state"]:
    assert name in globals(), f"Run Cell 4 first: missing {name}"
for name in ["registry_add_run", "eval_multihead_batch", "EVAL_REGISTRY"]:
    assert name in globals(), f"Run Cell 5 first: missing {name}"
assert "render_cartpole_frame_from_state" in globals(), "Run Cell 2 first (PIL renderer)"

tf.keras.backend.set_floatx("float64")
gpflow.config.set_default_float(np.float64)
gpflow.config.set_default_jitter(1e-6)
DTF = tf.float64

# ---------------------------
# Success-hold config
# ---------------------------
SUCCESS_HOLD_STEPS = int(globals().get("SUCCESS_HOLD_STEPS", 200))
STOP_ON_HOLD_SUCCESS = bool(globals().get("STOP_ON_HOLD_SUCCESS", True))

# ---------------------------
# Fairness: fixed episode reset seeds shared across all methods
# ---------------------------
_EP_RESET_SEED_TABLE = globals().get("_EP_RESET_SEED_TABLE", None)
if _EP_RESET_SEED_TABLE is None:
    _seed_rng = np.random.default_rng(int(SEED_BASE) + 20260221)
    _EP_RESET_SEED_TABLE = _seed_rng.integers(0, 10**9, size=(int(N_RUNS), int(N_EPISODES_PER_RUN)), dtype=np.int64)
    globals()["_EP_RESET_SEED_TABLE"] = _EP_RESET_SEED_TABLE

def _shared_ep_reset_seed(run_i, ep_i):
    return int(np.asarray(_EP_RESET_SEED_TABLE, dtype=np.int64)[int(run_i), int(ep_i)])

# ============================================================
# Text progress (PyCharm-friendly)
# ============================================================
class TextProgress:
    def __init__(self, total, prefix="", every=10):
        self.total = int(total)
        self.prefix = str(prefix)
        self.every = int(max(1, every))
        self.t0 = time.perf_counter()
        self.last_i = -1

    def update(self, i, extra=""):
        i = int(i)
        if (i == self.total - 1) or ((i % self.every) == 0 and i != self.last_i):
            dt = time.perf_counter() - self.t0
            rate = (i + 1) / max(dt, 1e-9)
            sys.stdout.write(f"\r{self.prefix}{i+1}/{self.total} | {rate:6.1f} it/s | {dt:6.1f}s {extra}")
            sys.stdout.flush()
            self.last_i = i

    def close(self):
        sys.stdout.write("\n")
        sys.stdout.flush()

# ============================================================
# Live + Replay knobs
# ============================================================
LIVE_RENDER = globals().get("LIVE_RENDER", True)
LIVE_ONLY_FIRST_RUN = globals().get("LIVE_ONLY_FIRST_RUN", True)
LIVE_ONLY_FIRST_EP  = globals().get("LIVE_ONLY_FIRST_EP", False)
LIVE_EVERY_STEPS = globals().get("LIVE_EVERY_STEPS", 3)
LIVE_MIN_DT = globals().get("LIVE_MIN_DT", 0.05)
LIVE_SIZE = globals().get("LIVE_SIZE", (720, 450))

CAPTURE_REPLAY = False
REPLAY_EVERY_STEPS = 2
REPLAY_FPS = 20
REPLAY_SIZE = (720, 450)
TITLE_HOLD_FRAMES = 12

if "LIVE_VIEWER" not in globals():
    class _LiveViewerFallback:
        def __init__(self, enabled=True, every_steps=3, min_dt=0.05, size=(720, 450)):
            self.enabled = bool(enabled)
            self.every_steps = int(every_steps)
            self.min_dt = float(min_dt)
            self.size = tuple(size)
            self._img_handle = None
            self._txt_handle = None
            self._last_step = -10**9
            self._last_t = 0.0
        def reset(self):
            self._last_step = -10**9
            self._last_t = 0.0
        def maybe_update(self, step_i, x, theta, text=""):
            if not self.enabled:
                return
            now = time.perf_counter()
            if (step_i - self._last_step) < self.every_steps:
                return
            if (now - self._last_t) < self.min_dt:
                return
            frame = render_cartpole_frame_from_state(x, theta, x_threshold=2.4, W=self.size[0], H=self.size[1])
            img = Image.fromarray(frame)
            if self._img_handle is None:
                self._img_handle = display(img, display_id=True)
                self._txt_handle = display(text, display_id=True)
            else:
                self._img_handle.update(img)
                self._txt_handle.update(text)
            self._last_step = int(step_i)
            self._last_t = now
    LIVE_VIEWER = _LiveViewerFallback(enabled=LIVE_RENDER, every_steps=LIVE_EVERY_STEPS, min_dt=LIVE_MIN_DT, size=LIVE_SIZE)

def _render_with_text(x, theta, text, W=720, H=450):
    frame = render_cartpole_frame_from_state(x, theta, x_threshold=2.4, W=W, H=H)
    img = Image.fromarray(frame)
    dr = ImageDraw.Draw(img)
    dr.rectangle([8, 8, W - 8, 44], fill=(255, 255, 255))
    dr.text((14, 14), text, fill=(0, 0, 0))
    return np.asarray(img, dtype=np.uint8)

def _title_frame(text, W=720, H=450):
    img = Image.new("RGB", (W, H), (245, 245, 245))
    dr = ImageDraw.Draw(img)
    dr.text((18, 18), text, fill=(0, 0, 0))
    return np.asarray(img, dtype=np.uint8)

def _show_replay(frames, fps=20):
    if len(frames) == 0:
        print("⚠️ No replay frames.")
        return
    fig, ax = plt.subplots(figsize=(8, 5))
    im = ax.imshow(frames[0])
    ax.axis("off")
    def update(i):
        im.set_data(frames[i])
        return [im]
    ani = animation.FuncAnimation(fig, update, frames=len(frames), interval=1000.0 / fps, blit=True)
    plt.close(fig)
    display(HTML(ani.to_jshtml()))

# ============================================================
# Kernels (one per head)
# ============================================================
def make_kernel_6d():
    return gpflow.kernels.Matern52(lengthscales=np.ones(6, dtype=np.float64), variance=1.0)

k_dx     = make_kernel_6d()
k_dxdot  = make_kernel_6d()
k_dth    = make_kernel_6d()
k_dthdot = make_kernel_6d()

# ============================================================
# OSGPR init + fixed-Z streaming update
# ============================================================
def init_osgpr_from_prior(X, Y, Z, kernel, noise=NOISE_INIT, iters=650, lr=0.02, freeze_kernel=False):
    mu0, Su0, Kaa0, Z0 = prior_summary(kernel, Z)
    m = OSGPR_VFE(
        data=(np.asarray(X, dtype=np.float64), np.asarray(Y, dtype=np.float64)),
        kernel=kernel,
        mu_old=mu0, Su_old=Su0, Kaa_old=Kaa0, Z_old=Z0,
        Z=np.asarray(Z, dtype=np.float64),
    )
    m.likelihood.variance.assign(float(noise))
    if freeze_kernel:
        try:
            m.kernel.variance.trainable = False
            m.kernel.lengthscales.trainable = False
        except Exception:
            pass
    t_sec, _ = train_osgpr(m, iters=iters, lr=lr, clip_norm=10.0)
    m.build_predict_cache()
    return m, float(t_sec)

def osgpr_stream_update_fixedZ(model_old, X_new, Y_new, Z_fixed,
                               iters=OSGPR_ITERS_PER_UPDATE, lr=OSGPR_LR,
                               noise=NOISE_INIT, freeze_kernel=False):
    mu_old, Su_old, Kaa_old, Z_old = extract_summary_from_model(model_old)
    m = OSGPR_VFE(
        data=(np.asarray(X_new, dtype=np.float64), np.asarray(Y_new, dtype=np.float64)),
        kernel=model_old.kernel,   # reuse kernel object
        mu_old=mu_old, Su_old=Su_old, Kaa_old=Kaa_old, Z_old=Z_old,
        Z=np.asarray(Z_fixed, dtype=np.float64),
    )
    m.likelihood.variance.assign(float(noise))
    if freeze_kernel:
        try:
            m.kernel.variance.trainable = False
            m.kernel.lengthscales.trainable = False
        except Exception:
            pass
    t_sec, _ = train_osgpr(m, iters=iters, lr=lr, clip_norm=10.0)
    m.build_predict_cache()
    return m, float(t_sec)

# ============================================================
# Local Z selection: anchors + corridor k-center
# ============================================================
def select_local_Z_from_corridor(
    Xpool, pool_ids, r2_pool,
    Z_global, anchor_idx,
    m_local=M_LOCAL, m_anchors=M_ANCHORS,
    r0=CORR_R0, r1=CORR_R1,
    rng=None
):
    if rng is None:
        rng = np.random.default_rng(0)

    Z_global = np.asarray(Z_global, dtype=np.float64)
    Z_anchor = Z_global[np.asarray(anchor_idx, dtype=np.int64)]

    N = int(Xpool.shape[0])
    r2 = np.asarray(r2_pool, dtype=np.float64).reshape(-1) if N else np.zeros((0,), dtype=np.float64)
    used_fallback = False

    idx0 = np.where(r2 <= float(r0))[0]
    idx1 = np.where(r2 <= float(r1))[0]

    need = int(m_local - m_anchors)
    if N == 0:
        Z_fill = Z_global[rng.choice(Z_global.shape[0], size=need, replace=True)]
        sel_ids = np.full((need,), -1, dtype=np.int64)
        dbg = dict(inside0=0, inside1=0, used_fallback=True)
        return np.concatenate([Z_anchor, Z_fill], axis=0)[:m_local], sel_ids, dbg

    if len(idx0) >= need:
        cand = idx0
    elif len(idx1) >= need:
        cand = idx1
    else:
        cand = idx1 if len(idx1) else np.arange(N)
        used_fallback = True

    Xcand = Xpool[cand]
    idcand = pool_ids[cand]

    start_idx = int(rng.integers(0, len(Xcand)))
    sel_local = kcenter_farthest_first(Xcand, need, start_idx=start_idx)

    Z_fill = Xcand[sel_local]
    sel_ids = idcand[sel_local]

    if Z_fill.shape[0] < need:
        pad_n = need - Z_fill.shape[0]
        pad_idx = rng.choice(N, size=pad_n, replace=True)
        Z_fill = np.concatenate([Z_fill, Xpool[pad_idx]], axis=0)
        sel_ids = np.concatenate([sel_ids, pool_ids[pad_idx]], axis=0)

    dbg = dict(inside0=int(len(idx0)), inside1=int(len(idx1)), used_fallback=bool(used_fallback))
    Z_sub = np.concatenate([Z_anchor, Z_fill], axis=0)[:m_local]
    return Z_sub, sel_ids[:need], dbg

# ============================================================
# MPPI plan_tf with corridor + tubeX (TensorArray fix)
# ============================================================
@tf.function
def _predict_pack_mu_var_tf_tensors(Z, L, mw, CT, kernel, Xfeat):
    Kzx = kernel(Z, Xfeat)
    A = tf.linalg.triangular_solve(L, Kzx, lower=True)
    mu = tf.matmul(A, mw, transpose_a=True)
    mu = tf.reshape(mu, (-1,))
    kxx = kernel(Xfeat, full_cov=False)
    q = tf.reduce_sum(tf.square(A), axis=0)
    E = tf.matmul(CT, A)
    var_add = tf.reduce_sum(tf.square(E), axis=0)
    var = tf.maximum(kxx - q + var_add, tf.constant(1e-12, DTF))
    return mu, var

@tf.function
def _gp_step_batch_tf_packs(s, u,
                           Zdx,Ldx,mwdx,CTdx, likdx,
                           Zdxd,Ldxd,mwdxd,CTdxd, likdxd,
                           Zdth,Ldth,mwdth,CTdth, likdth,
                           Zdthd,Ldthd,mwdthd,CTdthd, likdthd,
                           unc_lambda):
    Xfeat = tf_feature_map(s, u)
    mu_dx,  _       = _predict_pack_mu_var_tf_tensors(Zdx,   Ldx,   mwdx,   CTdx,   k_dx,     Xfeat)
    mu_dxd, var_dxd = _predict_pack_mu_var_tf_tensors(Zdxd,  Ldxd,  mwdxd,  CTdxd,  k_dxdot,  Xfeat)
    mu_dt,  _       = _predict_pack_mu_var_tf_tensors(Zdth,  Ldth,  mwdth,  CTdth,  k_dth,    Xfeat)
    mu_dtd, _       = _predict_pack_mu_var_tf_tensors(Zdthd, Ldthd, mwdthd, CTdthd, k_dthdot, Xfeat)

    x     = s[:, 0] + mu_dx
    xdot  = s[:, 1] + mu_dxd
    th    = tf_wrap_angle(s[:, 2] + mu_dt)
    thdot = s[:, 3] + mu_dtd
    s_next = tf.stack([x, xdot, th, thdot], axis=-1)

    unc = tf.sqrt(tf.maximum(var_dxd + likdxd, tf.constant(1e-12, DTF)))
    return s_next, unc_lambda * unc, unc

@tf.function
def _corridor_min_dist_tf(Xpool, tubeX, inv_var):
    diff = tf.expand_dims(Xpool, 1) - tf.expand_dims(tubeX, 0)
    d2 = tf.reduce_sum(tf.square(diff) * tf.reshape(inv_var, (1, 1, -1)), axis=-1)
    return tf.reduce_min(d2, axis=1)

# factory MUST be plain python
def make_plan_tf(H=H, K=K, sigma=MPPI_SIGMA, lam=MPPI_LAMBDA,
                 u_min=U_MIN, u_max=U_MAX, unc_lambda=UNC_LAMBDA,
                 tube_invvar_eps=TUBE_INVVAR_EPS):

    sigma = float(sigma); lam = float(lam)
    u_min = float(u_min); u_max = float(u_max)
    unc_lambda = float(unc_lambda)
    tube_invvar_eps = float(tube_invvar_eps)

    @tf.function(reduce_retracing=True)
    def plan_tf(s0, u_mean, Xpool, corr_r0, corr_r1,
                Zdx,Ldx,mwdx,CTdx, likdx,
                Zdxd,Ldxd,mwdxd,CTdxd, likdxd,
                Zdth,Ldth,mwdth,CTdth, likdth,
                Zdthd,Ldthd,mwdthd,CTdthd, likdthd):

        s0 = tf.cast(tf.reshape(s0, (4,)), DTF)
        u_mean = tf.cast(tf.reshape(u_mean, (H,)), DTF)
        Xpool = tf.cast(Xpool, DTF)

        eps = tf.random.normal((K, H), mean=0.0, stddev=sigma, dtype=DTF)
        U = tf.clip_by_value(tf.expand_dims(u_mean, 0) + eps, u_min, u_max)

        s = tf.repeat(tf.expand_dims(s0, 0), repeats=K, axis=0)
        cost = tf.zeros((K,), dtype=DTF)
        unc_sum = tf.zeros((K,), dtype=DTF)

        for t in tf.range(H):
            u_t = U[:, t]
            s, unc_bonus, unc = _gp_step_batch_tf_packs(
                s, u_t,
                Zdx,Ldx,mwdx,CTdx, likdx,
                Zdxd,Ldxd,mwdxd,CTdxd, likdxd,
                Zdth,Ldth,mwdth,CTdth, likdth,
                Zdthd,Ldthd,mwdthd,CTdthd, likdthd,
                tf.constant(unc_lambda, DTF),
            )
            cost = cost + stage_cost_tf(s, u_t) + unc_bonus
            unc_sum = unc_sum + unc

        beta = tf.reduce_min(cost)
        w = tf.exp(-(cost - beta) / tf.constant(lam, DTF))
        w_sum = tf.reduce_sum(w) + tf.constant(1e-12, DTF)

        u_mean_new = tf.reduce_sum(tf.expand_dims(w, 1) * U, axis=0) / w_sum
        u0 = u_mean_new[0]
        unc_w = tf.reduce_sum(w * (unc_sum / tf.cast(H, DTF))) / w_sum

        # tubeX via TensorArray (scope-safe)
        sT = tf.reshape(s0, (1, 4))
        ta = tf.TensorArray(DTF, size=H, element_shape=(6,))
        for t in tf.range(H):
            u_t = tf.reshape(u_mean_new[t], (1,))
            Xft = tf_feature_map(sT, u_t)
            ta = ta.write(t, tf.reshape(Xft, (6,)))
            sT, _, _ = _gp_step_batch_tf_packs(
                sT, u_t,
                Zdx,Ldx,mwdx,CTdx, likdx,
                Zdxd,Ldxd,mwdxd,CTdxd, likdxd,
                Zdth,Ldth,mwdth,CTdth, likdth,
                Zdthd,Ldthd,mwdthd,CTdthd, likdthd,
                tf.constant(0.0, DTF),
            )
        tubeX = ta.stack()  # (H,6)

        tube_var = tf.math.reduce_variance(tubeX, axis=0)
        inv_var = 1.0 / (tube_var + tf.constant(tube_invvar_eps, DTF))

        Npool = tf.shape(Xpool)[0]
        r2_pool = tf.cond(
            Npool > 0,
            lambda: _corridor_min_dist_tf(Xpool, tubeX, inv_var),
            lambda: tf.zeros((0,), dtype=DTF),
        )

        inside0 = tf.reduce_sum(tf.cast(r2_pool <= tf.cast(corr_r0, DTF), tf.int32))
        inside1 = tf.reduce_sum(tf.cast(r2_pool <= tf.cast(corr_r1, DTF), tf.int32))
        corr_dbg = tf.stack([tf.cast(Npool, tf.float64),
                             tf.cast(inside0, tf.float64),
                             tf.cast(inside1, tf.float64)], axis=0)

        return u0, u_mean_new, tubeX, unc_w, r2_pool, inv_var, corr_dbg

    return plan_tf

plan_tf = make_plan_tf()

# ============================================================
# Packs at Z_sub for all 4 heads
# ============================================================
def packs_from_Zsub(models, Z_sub):
    m_dx, m_dxdot, m_dth, m_dthdot = models
    p_dx     = build_pack_from_model(m_dx,     Z_sub, jitter=1e-6, use_cached=True)
    p_dxdot  = build_pack_from_model(m_dxdot,  Z_sub, jitter=1e-6, use_cached=True)
    p_dth    = build_pack_from_model(m_dth,    Z_sub, jitter=1e-6, use_cached=True)
    p_dthdot = build_pack_from_model(m_dthdot, Z_sub, jitter=1e-6, use_cached=True)
    return (
        p_dx.Z, p_dx.L, p_dx.m_w, p_dx.C_T, p_dx.lik_var,
        p_dxdot.Z, p_dxdot.L, p_dxdot.m_w, p_dxdot.C_T, p_dxdot.lik_var,
        p_dth.Z, p_dth.L, p_dth.m_w, p_dth.C_T, p_dth.lik_var,
        p_dthdot.Z, p_dthdot.L, p_dthdot.m_w, p_dthdot.C_T, p_dthdot.lik_var,
    )

# ============================================================
# Episode-end refit Z_GLOBAL (D-opt) on replay candidates
# ============================================================
def refit_Z_global_multihead(Xcand, M_new, kernels, lam=1e-6, rng=None):
    if rng is None:
        rng = np.random.default_rng(0)
    Xcand = np.asarray(Xcand, dtype=np.float64)
    N = Xcand.shape[0]
    if N <= M_new:
        return Xcand.copy()

    Ncap = min(N, 1800)
    if N > Ncap:
        idx = rng.choice(N, size=Ncap, replace=False)
        Xuse = Xcand[idx]
    else:
        Xuse = Xcand

    Ksum = None
    for k in kernels:
        Kk = k.K(Xuse).numpy()
        Ksum = Kk if Ksum is None else (Ksum + Kk)
    Kavg = Ksum / float(len(kernels))
    Kavg = 0.5 * (Kavg + Kavg.T) + float(lam) * np.eye(Kavg.shape[0], dtype=np.float64)

    sel = greedy_dopt_anchors_from_K(Kavg, m_anchors=int(M_new), lam=lam)
    return Xuse[sel]

# ============================================================
# MAIN: PALSGP
# ============================================================
METHOD = "PALSGP"
print(f"\n===========================\nRunning {METHOD} (hold {SUCCESS_HOLD_STEPS} steps)\n===========================\n")

POOL_MAX = 5000
REFIT_REPLAY_MAX = 4000

# Init Z_GLOBAL from random data
rng0 = np.random.default_rng(SEED_BASE)
M_GLOBAL = int(min(M_GLOBAL_INIT, X0.shape[0]))
Z_GLOBAL = np.asarray(X0[rng0.choice(X0.shape[0], size=M_GLOBAL, replace=False)], dtype=np.float64)

# anchors from dxdot kernel
Kzz_init = k_dxdot.K(Z_GLOBAL).numpy()
anchor_idx = greedy_dopt_anchors_from_K(Kzz_init, m_anchors=M_ANCHORS, lam=1e-6)
print("✅ Z_GLOBAL init:", Z_GLOBAL.shape, "anchors:", anchor_idx.shape)

# initial global training
print("Training initial global OSGPR models...")
m_dx,     _ = init_osgpr_from_prior(X0, Ydx0,     Z_GLOBAL, k_dx,     noise=NOISE_INIT, iters=650, lr=0.02)
m_dxdot,  _ = init_osgpr_from_prior(X0, Ydxdot0,  Z_GLOBAL, k_dxdot,  noise=NOISE_INIT, iters=650, lr=0.02)
m_dth,    _ = init_osgpr_from_prior(X0, Ydth0,    Z_GLOBAL, k_dth,    noise=NOISE_INIT, iters=650, lr=0.02)
m_dthdot, _ = init_osgpr_from_prior(X0, Ydthdot0, Z_GLOBAL, k_dthdot, noise=NOISE_INIT, iters=650, lr=0.02)

for run in range(N_RUNS):
    seed = SEED_BASE + run
    rng = np.random.default_rng(seed)
    env = make_env(render_mode=None, seed=seed, max_episode_steps=MAX_STEPS_PER_EP, start_down=True, edge_respawn=True)
    do_live = bool(LIVE_RENDER and ((not LIVE_ONLY_FIRST_RUN) or (run == 0)))

    # Hard reset models/inducing state for this run (fair independent runs)
    rng_init_run = np.random.default_rng(seed)
    M_GLOBAL = int(min(M_GLOBAL_INIT, X0.shape[0]))
    Z_GLOBAL = np.asarray(X0[rng_init_run.choice(X0.shape[0], size=M_GLOBAL, replace=False)], dtype=np.float64)
    Kzz_init = k_dxdot.K(Z_GLOBAL).numpy()
    anchor_idx = greedy_dopt_anchors_from_K(Kzz_init, m_anchors=M_ANCHORS, lam=1e-6)
    m_dx,     _ = init_osgpr_from_prior(X0, Ydx0,     Z_GLOBAL, k_dx,     noise=NOISE_INIT, iters=650, lr=0.02)
    m_dxdot,  _ = init_osgpr_from_prior(X0, Ydxdot0,  Z_GLOBAL, k_dxdot,  noise=NOISE_INIT, iters=650, lr=0.02)
    m_dth,    _ = init_osgpr_from_prior(X0, Ydth0,    Z_GLOBAL, k_dth,    noise=NOISE_INIT, iters=650, lr=0.02)
    m_dthdot, _ = init_osgpr_from_prior(X0, Ydthdot0, Z_GLOBAL, k_dthdot, noise=NOISE_INIT, iters=650, lr=0.02)

    run_frames = []
    if CAPTURE_REPLAY:
        run_frames.extend([_title_frame(f"{METHOD} | RUN {run} | start", W=REPLAY_SIZE[0], H=REPLAY_SIZE[1])] * TITLE_HOLD_FRAMES)

    # replay buffer for refit/training
    Xmem = [np.asarray(X0[i], dtype=np.float64) for i in range(X0.shape[0])]
    Ydx_mem     = [float(Ydx0[i, 0]) for i in range(Ydx0.shape[0])]
    Ydxdot_mem  = [float(Ydxdot0[i, 0]) for i in range(Ydxdot0.shape[0])]
    Ydth_mem    = [float(Ydth0[i, 0]) for i in range(Ydth0.shape[0])]
    Ydthdot_mem = [float(Ydthdot0[i, 0]) for i in range(Ydthdot0.shape[0])]
    mem_ids = list(range(len(Xmem)))
    next_id = int(len(Xmem))

    # online update buffer
    Xbuf, ydx_buf, ydxdot_buf, ydth_buf, ydthdot_buf = [], [], [], [], []

    u_mean = np.zeros((H,), dtype=np.float64)

    # traces
    pred_time_step, train_time_step, rebuild_time_step, wall_time_step, vis_time_step = [], [], [], [], []
    u_exec_step, unc_exec_step = [], []
    local_overlap_step, local_rebuild_flag_step, update_flag_step = [], [], []
    corr_poolN_step, corr_inside0_step, corr_inside1_step, corr_used_fallback_step = [], [], [], []
    update_t_global, update_rmse_mean, update_nlpd_mean, update_cover2_mean, update_std_mean = [], [], [], [], []

    # episode metrics
    ep_returns, ep_lengths, ep_success, ep_first_success_t_global = [], [], [], []
    ep_control_l2, ep_u_abs_mean, ep_unc_mean, ep_unc_max = [], [], [], []
    ep_hold_max_list = []

    total_reward_run = 0.0
    run_success = False
    first_success_t = None

    t_global = 0

    # init local packs
    models = (m_dx, m_dxdot, m_dth, m_dthdot)
    Z_sub0 = np.concatenate(
        [Z_GLOBAL[anchor_idx],
         Z_GLOBAL[rng.choice(Z_GLOBAL.shape[0], size=max(M_LOCAL - M_ANCHORS, 1), replace=True)]],
        axis=0
    )[:M_LOCAL]
    pack_tensors = packs_from_Zsub(models, Z_sub0)
    last_sel_ids = np.full((M_LOCAL - M_ANCHORS,), -999, dtype=np.int64)
    last_rebuild_t = 0

    for ep in range(N_EPISODES_PER_RUN):
        obs, _ = env.reset(seed=_shared_ep_reset_seed(run, ep))
        x, xdot, th, thdot = obs_to_state(obs)

        if do_live and (not LIVE_ONLY_FIRST_EP or ep == 0):
            LIVE_VIEWER.reset()
        if CAPTURE_REPLAY:
            run_frames.extend([_title_frame(f"{METHOD} | RUN {run} | EP {ep}", W=REPLAY_SIZE[0], H=REPLAY_SIZE[1])] * TITLE_HOLD_FRAMES)

        # episode accum
        ep_reward = 0.0
        ep_steps = 0
        ep_u2_sum = 0.0
        ep_u_abs_sum = 0.0
        ep_unc_sum = 0.0
        ep_unc_max_v = 0.0

        # HOLD success tracking
        hold_count = 0
        hold_max = 0
        ep_hold_complete_t = None

        ep_prog = TextProgress(MAX_STEPS_PER_EP, prefix=f"{METHOD} run{run} ep{ep} step ", every=10)

        for step in range(MAX_STEPS_PER_EP):
            # corridor candidate pool from recent replay
            if len(Xmem) > 0:
                lo = max(0, len(Xmem) - POOL_MAX)
                Xpool_np = np.asarray(Xmem[lo:], dtype=np.float64)
                pool_ids_np = np.asarray(mem_ids[lo:], dtype=np.int64)
            else:
                Xpool_np = np.zeros((0, 6), dtype=np.float64)
                pool_ids_np = np.zeros((0,), dtype=np.int64)

            # plan
            tp0 = time.perf_counter()
            s0 = np.array([x, xdot, th, thdot], dtype=np.float64)
            u0_tf, u_mean_new_tf, _, unc_w_tf, r2_pool_tf, _, corr_dbg_tf = plan_tf(
                s0, u_mean, Xpool_np, float(CORR_R0), float(CORR_R1),
                *pack_tensors
            )
            u0 = float(u0_tf.numpy())
            u_mean_new = u_mean_new_tf.numpy()

            # receding horizon
            u_mean = np.roll(u_mean_new, -1)
            u_mean[-1] = 0.0

            r2_pool = r2_pool_tf.numpy()
            corr_dbg = corr_dbg_tf.numpy()
            pred_dt = float(time.perf_counter() - tp0)

            # execute
            obs1, r, terminated, truncated, info = env.step(np.array([u0], dtype=np.float32))
            x1, xdot1, th1, thdot1 = obs_to_state(obs1)

            # live
            if do_live and (not LIVE_ONLY_FIRST_EP or ep == 0):
                LIVE_VIEWER.maybe_update(
                    step_i=t_global, x=x1, theta=th1,
                    text=f"{METHOD} | r{run} e{ep} t={t_global} | r={float(r):+.3f} u={u0:+.3f} hold={hold_count}/{SUCCESS_HOLD_STEPS}"
                )

            # replay frames
            if CAPTURE_REPLAY and ((t_global % REPLAY_EVERY_STEPS) == 0):
                run_frames.append(_render_with_text(
                    x1, th1,
                    text=f"{METHOD} r{run} e{ep} t{t_global} | r={float(r):+.2f} u={u0:+.2f} hold={hold_count}",
                    W=REPLAY_SIZE[0], H=REPLAY_SIZE[1]
                ))

            # ---- HOLD success logic ----
            succ_inst = is_success_state(
                x1, xdot1, th1, thdot1,
                cos_th_min=SUCCESS_COS_TH_MIN,
                x_abs_max=SUCCESS_X_ABS_MAX,
                xdot_abs_max=SUCCESS_XDOT_ABS_MAX,
                thdot_abs_max=SUCCESS_THDOT_ABS_MAX
            )

            if bool(info.get("respawned", False)) or bool(info.get("post_respawn_cooldown", 0)):
                hold_count = 0
            else:
                hold_count = (hold_count + 1) if succ_inst else 0

            hold_max = max(hold_max, hold_count)
            succ_hold = (hold_count >= SUCCESS_HOLD_STEPS)

            if succ_hold and (ep_hold_complete_t is None):
                ep_hold_complete_t = int(t_global)
            if succ_hold and (not run_success):
                run_success = True
                first_success_t = int(t_global)

            # transition
            feat = state_to_features(x, xdot, th, thdot, u0).astype(np.float64)
            d_x = (x1 - x)
            d_xdot = (xdot1 - xdot)
            d_th = wrap_pi(th1 - th)
            d_thdot = (thdot1 - thdot)

            if (not bool(info.get("respawned", False))) and (not bool(info.get("post_respawn_cooldown", 0))):
                Xbuf.append(feat)
                ydx_buf.append([d_x])
                ydxdot_buf.append([d_xdot])
                ydth_buf.append([d_th])
                ydthdot_buf.append([d_thdot])

                Xmem.append(feat)
                Ydx_mem.append(float(d_x))
                Ydxdot_mem.append(float(d_xdot))
                Ydth_mem.append(float(d_th))
                Ydthdot_mem.append(float(d_thdot))
                mem_ids.append(next_id)
                next_id += 1

            # episode accumulators
            ep_reward += float(r); ep_steps += 1
            ep_u2_sum += u0*u0; ep_u_abs_sum += abs(u0)
            ep_unc_sum += float(unc_w_tf.numpy()); ep_unc_max_v = max(ep_unc_max_v, float(unc_w_tf.numpy()))
            total_reward_run += float(r)

            # online update every UPDATE_EVERY
            trn_dt = 0.0
            reb_dt = 0.0
            did_update = False

            if ((t_global + 1) % int(UPDATE_EVERY) == 0) and (len(Xbuf) > 0):
                did_update = True
                Xnew = np.asarray(Xbuf, dtype=np.float64)
                ydx_new     = np.asarray(ydx_buf, dtype=np.float64)
                ydxdot_new  = np.asarray(ydxdot_buf, dtype=np.float64)
                ydth_new    = np.asarray(ydth_buf, dtype=np.float64)
                ydthdot_new = np.asarray(ydthdot_buf, dtype=np.float64)

                tu0 = time.perf_counter()
                m_dx,     _ = osgpr_stream_update_fixedZ(m_dx,     Xnew, ydx_new,     Z_GLOBAL)
                m_dxdot,  _ = osgpr_stream_update_fixedZ(m_dxdot,  Xnew, ydxdot_new,  Z_GLOBAL)
                m_dth,    _ = osgpr_stream_update_fixedZ(m_dth,    Xnew, ydth_new,    Z_GLOBAL)
                m_dthdot, _ = osgpr_stream_update_fixedZ(m_dthdot, Xnew, ydthdot_new, Z_GLOBAL)
                trn_dt = float(time.perf_counter() - tu0)
                models = (m_dx, m_dxdot, m_dth, m_dthdot)

                try:
                    met = eval_multihead_batch(
                        [m_dx, m_dxdot, m_dth, m_dthdot],
                        Xnew, [ydx_new, ydxdot_new, ydth_new, ydthdot_new],
                        add_likelihood_var=True
                    )
                    update_t_global.append(int(t_global))
                    update_rmse_mean.append(float(met["rmse_mean"]))
                    update_nlpd_mean.append(float(met["nlpd_mean"]))
                    update_cover2_mean.append(float(met["cover2_mean"]))
                    update_std_mean.append(float(met["std_mean_mean"]))
                except Exception:
                    update_t_global.append(int(t_global))
                    update_rmse_mean.append(np.nan)
                    update_nlpd_mean.append(np.nan)
                    update_cover2_mean.append(np.nan)
                    update_std_mean.append(np.nan)

                Xbuf.clear(); ydx_buf.clear(); ydxdot_buf.clear(); ydth_buf.clear(); ydthdot_buf.clear()

            # local rebuild policy (force after update)
            if len(pool_ids_np) > 0 and len(last_sel_ids) > 0 and np.all(last_sel_ids >= 0):
                overlap = float(np.mean(np.isin(last_sel_ids, pool_ids_np)))
            else:
                overlap = 0.0

            must_rebuild = (
                did_update or
                ((t_global - last_rebuild_t) >= int(LOCAL_REBUILD_EVERY)) or
                (overlap < float(LOCAL_OVERLAP_TRIG))
            )

            inside0 = int(corr_dbg[1]) if corr_dbg.size else 0
            inside1 = int(corr_dbg[2]) if corr_dbg.size else 0
            poolN = int(corr_dbg[0]) if corr_dbg.size else int(len(Xpool_np))
            used_fallback = False

            if must_rebuild:
                trb0 = time.perf_counter()
                Z_sub, sel_ids, dbg = select_local_Z_from_corridor(
                    Xpool_np, pool_ids_np, r2_pool,
                    Z_GLOBAL, anchor_idx,
                    m_local=M_LOCAL, m_anchors=M_ANCHORS,
                    r0=CORR_R0, r1=CORR_R1,
                    rng=rng
                )
                pack_tensors = packs_from_Zsub(models, Z_sub)
                last_sel_ids = np.asarray(sel_ids, dtype=np.int64)
                last_rebuild_t = int(t_global)
                reb_dt = float(time.perf_counter() - trb0)

                inside0 = int(dbg["inside0"])
                inside1 = int(dbg["inside1"])
                used_fallback = bool(dbg["used_fallback"])

            # logs
            wall_dt = pred_dt + trn_dt + reb_dt
            pred_time_step.append(pred_dt)
            train_time_step.append(trn_dt)
            rebuild_time_step.append(reb_dt)
            wall_time_step.append(wall_dt)
            vis_time_step.append(0.0)

            u_exec_step.append(float(u0))
            unc_exec_step.append(float(unc_w_tf.numpy()))
            local_overlap_step.append(float(overlap))
            local_rebuild_flag_step.append(1.0 if must_rebuild else 0.0)
            update_flag_step.append(1.0 if did_update else 0.0)

            corr_poolN_step.append(float(poolN))
            corr_inside0_step.append(float(inside0))
            corr_inside1_step.append(float(inside1))
            corr_used_fallback_step.append(1.0 if used_fallback else 0.0)

            ep_prog.update(step, extra=f"| r={float(r):+.2f} u={u0:+.2f} hold={hold_count}/{SUCCESS_HOLD_STEPS} upd={'Y' if did_update else 'n'}")

            # advance
            x, xdot, th, thdot = x1, xdot1, th1, thdot1
            t_global += 1

            if STOP_ON_HOLD_SUCCESS and succ_hold:
                break

        ep_prog.close()

        # episode stats
        ep_returns.append(float(ep_reward))
        ep_lengths.append(int(ep_steps))
        ep_success.append(bool(ep_hold_complete_t is not None))
        ep_first_success_t_global.append(int(ep_hold_complete_t) if ep_hold_complete_t is not None else None)
        ep_control_l2.append(float(ep_u2_sum))
        ep_u_abs_mean.append(float(ep_u_abs_sum / max(ep_steps, 1)))
        ep_unc_mean.append(float(ep_unc_sum / max(ep_steps, 1)))
        ep_unc_max.append(float(ep_unc_max_v))
        ep_hold_max_list.append(int(hold_max))

        # episode-end refit Z + reinit globals
        if len(Xmem) > REFIT_REPLAY_MAX:
            idxr = rng.choice(len(Xmem), size=REFIT_REPLAY_MAX, replace=False)
        else:
            idxr = np.arange(len(Xmem))

        Xr = np.asarray([Xmem[i] for i in idxr], dtype=np.float64)
        ydx_r     = np.asarray([[Ydx_mem[i]] for i in idxr], dtype=np.float64)
        ydxdot_r  = np.asarray([[Ydxdot_mem[i]] for i in idxr], dtype=np.float64)
        ydth_r    = np.asarray([[Ydth_mem[i]] for i in idxr], dtype=np.float64)
        ydthdot_r = np.asarray([[Ydthdot_mem[i]] for i in idxr], dtype=np.float64)

        Z_GLOBAL = refit_Z_global_multihead(
            Xr, M_new=int(M_GLOBAL),
            kernels=[k_dx, k_dxdot, k_dth, k_dthdot],
            lam=1e-6, rng=rng
        )
        Kzz_new = k_dxdot.K(Z_GLOBAL).numpy()
        anchor_idx = greedy_dopt_anchors_from_K(Kzz_new, m_anchors=M_ANCHORS, lam=1e-6)

        m_dx,     _ = init_osgpr_from_prior(Xr, ydx_r,     Z_GLOBAL, k_dx,     noise=NOISE_INIT, iters=250, lr=0.02)
        m_dxdot,  _ = init_osgpr_from_prior(Xr, ydxdot_r,  Z_GLOBAL, k_dxdot,  noise=NOISE_INIT, iters=250, lr=0.02)
        m_dth,    _ = init_osgpr_from_prior(Xr, ydth_r,    Z_GLOBAL, k_dth,    noise=NOISE_INIT, iters=250, lr=0.02)
        m_dthdot, _ = init_osgpr_from_prior(Xr, ydthdot_r, Z_GLOBAL, k_dthdot, noise=NOISE_INIT, iters=250, lr=0.02)
        models = (m_dx, m_dxdot, m_dth, m_dthdot)

        # rebuild locals after refit
        Z_sub_ep = np.concatenate(
            [Z_GLOBAL[anchor_idx],
             Z_GLOBAL[rng.choice(Z_GLOBAL.shape[0], size=max(M_LOCAL - M_ANCHORS, 1), replace=True)]],
            axis=0
        )[:M_LOCAL]
        pack_tensors = packs_from_Zsub(models, Z_sub_ep)
        last_sel_ids = np.full((M_LOCAL - M_ANCHORS,), -999, dtype=np.int64)
        last_rebuild_t = int(t_global)

    env.close()

    # registry
    run_stats = dict(
        total_reward=float(total_reward_run),
        success=bool(run_success),
        first_success_t_global=int(first_success_t) if first_success_t is not None else None,

        ep_returns=ep_returns,
        ep_lengths=ep_lengths,
        ep_success=ep_success,
        ep_first_success_t_global=ep_first_success_t_global,
        ep_control_l2=ep_control_l2,
        ep_u_abs_mean=ep_u_abs_mean,
        ep_unc_mean=ep_unc_mean,
        ep_unc_max=ep_unc_max,
        ep_hold_max=ep_hold_max_list,
    )

    run_traces = dict(
        wall_time_step=wall_time_step,
        pred_time_step=pred_time_step,
        train_time_step=train_time_step,
        rebuild_time_step=rebuild_time_step,
        vis_time_step=vis_time_step,

        u_exec_step=u_exec_step,
        unc_exec_step=unc_exec_step,

        local_overlap_step=local_overlap_step,
        local_rebuild_flag_step=local_rebuild_flag_step,
        update_flag_step=update_flag_step,

        corr_poolN_step=corr_poolN_step,
        corr_inside0_step=corr_inside0_step,
        corr_inside1_step=corr_inside1_step,
        corr_used_fallback_step=corr_used_fallback_step,

        update_t_global=update_t_global,
        update_rmse_mean=update_rmse_mean,
        update_nlpd_mean=update_nlpd_mean,
        update_cover2_mean=update_cover2_mean,
        update_std_mean=update_std_mean,
    )

    registry_add_run(METHOD, run_stats, run_traces)

    print(f"[{METHOD}] run {run+1}/{N_RUNS}: reward={total_reward_run:.2f} success={run_success} first_success_t={first_success_t}")

    if CAPTURE_REPLAY:
        print(f"\n▶️ Replay: {METHOD} RUN {run} (all episodes) | frames={len(run_frames)} | fps={REPLAY_FPS}")
        _show_replay(run_frames, fps=REPLAY_FPS)

print(f"\n✅ {METHOD} finished. Registry runs = {len(EVAL_REGISTRY[METHOD]['run_stats'])}")

# ===========================
# Cell 7 — SVGP_GLOBAL (MPPI + SVGP online training)
# Success = hold upright for SUCCESS_HOLD_STEPS consecutive steps
# Progress: TextProgress; Live + replay
# ===========================

import numpy as np
import tensorflow as tf
import gpflow
import time
import sys

from IPython.display import HTML, display
import matplotlib.pyplot as plt
from matplotlib import animation
from PIL import Image, ImageDraw

# Preconditions
for name in ["X0", "Ydx0", "Ydxdot0", "Ydth0", "Ydthdot0"]:
    assert name in globals(), f"Run Cell 2 first: missing {name}"
for name in ["build_pack_from_model", "stage_cost_tf", "tf_feature_map", "tf_wrap_angle", "is_success_state"]:
    assert name in globals(), f"Run Cell 4 first: missing {name}"
for name in ["registry_add_run", "eval_multihead_batch", "EVAL_REGISTRY"]:
    assert name in globals(), f"Run Cell 5 first: missing {name}"
assert "render_cartpole_frame_from_state" in globals(), "Run Cell 2 first (PIL renderer)"

tf.keras.backend.set_floatx("float64")
gpflow.config.set_default_float(np.float64)
gpflow.config.set_default_jitter(1e-6)
DTF = tf.float64

SUCCESS_HOLD_STEPS = int(globals().get("SUCCESS_HOLD_STEPS", 200))
STOP_ON_HOLD_SUCCESS = bool(globals().get("STOP_ON_HOLD_SUCCESS", True))

class TextProgress:
    def __init__(self, total, prefix="", every=10):
        self.total = int(total); self.prefix = str(prefix); self.every = int(max(1, every))
        self.t0 = time.perf_counter(); self.last_i = -1
    def update(self, i, extra=""):
        i = int(i)
        if (i == self.total - 1) or ((i % self.every) == 0 and i != self.last_i):
            dt = time.perf_counter() - self.t0
            rate = (i + 1) / max(dt, 1e-9)
            sys.stdout.write(f"\r{self.prefix}{i+1}/{self.total} | {rate:6.1f} it/s | {dt:6.1f}s {extra}")
            sys.stdout.flush()
            self.last_i = i
    def close(self):
        sys.stdout.write("\n"); sys.stdout.flush()

LIVE_RENDER = globals().get("LIVE_RENDER", True)
LIVE_ONLY_FIRST_RUN = globals().get("LIVE_ONLY_FIRST_RUN", True)
LIVE_ONLY_FIRST_EP  = globals().get("LIVE_ONLY_FIRST_EP", False)
LIVE_EVERY_STEPS = globals().get("LIVE_EVERY_STEPS", 3)
LIVE_MIN_DT = globals().get("LIVE_MIN_DT", 0.05)
LIVE_SIZE = globals().get("LIVE_SIZE", (720, 450))

CAPTURE_REPLAY = False
REPLAY_EVERY_STEPS = 2
REPLAY_FPS = 20
REPLAY_SIZE = (720, 450)
TITLE_HOLD_FRAMES = 12

if "LIVE_VIEWER" not in globals():
    class _LiveViewerFallback:
        def __init__(self, enabled=True, every_steps=3, min_dt=0.05, size=(720, 450)):
            self.enabled = bool(enabled); self.every_steps = int(every_steps); self.min_dt = float(min_dt); self.size = tuple(size)
            self._img_handle = None; self._txt_handle = None; self._last_step = -10**9; self._last_t = 0.0
        def reset(self):
            self._last_step = -10**9; self._last_t = 0.0
        def maybe_update(self, step_i, x, theta, text=""):
            if not self.enabled: return
            now = time.perf_counter()
            if (step_i - self._last_step) < self.every_steps: return
            if (now - self._last_t) < self.min_dt: return
            frame = render_cartpole_frame_from_state(x, theta, x_threshold=2.4, W=self.size[0], H=self.size[1])
            img = Image.fromarray(frame)
            if self._img_handle is None:
                self._img_handle = display(img, display_id=True)
                self._txt_handle = display(text, display_id=True)
            else:
                self._img_handle.update(img); self._txt_handle.update(text)
            self._last_step = int(step_i); self._last_t = now
    LIVE_VIEWER = _LiveViewerFallback(enabled=LIVE_RENDER, every_steps=LIVE_EVERY_STEPS, min_dt=LIVE_MIN_DT, size=LIVE_SIZE)

def _render_with_text(x, theta, text, W=720, H=450):
    frame = render_cartpole_frame_from_state(x, theta, x_threshold=2.4, W=W, H=H)
    img = Image.fromarray(frame)
    dr = ImageDraw.Draw(img)
    dr.rectangle([8, 8, W - 8, 44], fill=(255, 255, 255))
    dr.text((14, 14), text, fill=(0, 0, 0))
    return np.asarray(img, dtype=np.uint8)

def _title_frame(text, W=720, H=450):
    img = Image.new("RGB", (W, H), (245, 245, 245))
    dr = ImageDraw.Draw(img); dr.text((18, 18), text, fill=(0, 0, 0))
    return np.asarray(img, dtype=np.uint8)

def _show_replay(frames, fps=20):
    if len(frames) == 0:
        print("⚠️ No replay frames.")
        return
    fig, ax = plt.subplots(figsize=(8, 5))
    im = ax.imshow(frames[0]); ax.axis("off")
    def update(i):
        im.set_data(frames[i]); return [im]
    ani = animation.FuncAnimation(fig, update, frames=len(frames), interval=1000.0 / fps, blit=True)
    plt.close(fig); display(HTML(ani.to_jshtml()))

# Kernels
def make_kernel_6d():
    return gpflow.kernels.Matern52(lengthscales=np.ones(6, dtype=np.float64), variance=1.0)

k_dx     = make_kernel_6d()
k_dxdot  = make_kernel_6d()
k_dth    = make_kernel_6d()
k_dthdot = make_kernel_6d()

# SVGP builders
def make_svgp(Z, kernel, noise=NOISE_INIT, train_hypers=False):
    Z = np.asarray(Z, dtype=np.float64)
    m = gpflow.models.SVGP(
        kernel=kernel,
        likelihood=gpflow.likelihoods.Gaussian(variance=float(noise)),
        inducing_variable=gpflow.inducing_variables.InducingPoints(Z),
        num_latent_gps=1,
        mean_function=None,
        whiten=True,
        q_diag=False,
    )
    gpflow.set_trainable(m.inducing_variable, False)
    if not train_hypers:
        try:
            m.kernel.variance.trainable = False
            m.kernel.lengthscales.trainable = False
        except Exception:
            pass
    return m

def svgp_train_steps(model, X, Y, steps=80, lr=0.02, clip_norm=10.0):
    opt = tf.keras.optimizers.Adam(lr)
    Xt = tf.convert_to_tensor(np.asarray(X, dtype=np.float64))
    Yt = tf.convert_to_tensor(np.asarray(Y, dtype=np.float64))
    @tf.function
    def train_step():
        with tf.GradientTape() as tape:
            loss = -model.elbo((Xt, Yt))
        grads = tape.gradient(loss, model.trainable_variables)
        if clip_norm is not None:
            grads = [tf.clip_by_norm(g, clip_norm) if g is not None else None for g in grads]
        opt.apply_gradients([(g, v) for g, v in zip(grads, model.trainable_variables) if g is not None])
        return loss
    t0 = time.perf_counter()
    last = None
    for _ in range(int(steps)):
        last = train_step()
    return float(time.perf_counter() - t0)

# MPPI plan_tf (simple, no corridor/tube) using packs
@tf.function
def _predict_pack_mu_var_tf_tensors(Z, L, mw, CT, kernel, Xfeat):
    Kzx = kernel(Z, Xfeat)
    A = tf.linalg.triangular_solve(L, Kzx, lower=True)
    mu = tf.matmul(A, mw, transpose_a=True); mu = tf.reshape(mu, (-1,))
    kxx = kernel(Xfeat, full_cov=False)
    q = tf.reduce_sum(tf.square(A), axis=0)
    E = tf.matmul(CT, A)
    var_add = tf.reduce_sum(tf.square(E), axis=0)
    var = tf.maximum(kxx - q + var_add, tf.constant(1e-12, DTF))
    return mu, var

@tf.function
def _gp_step_batch_tf_packs(s, u,
                           Zdx,Ldx,mwdx,CTdx, likdx,
                           Zdxd,Ldxd,mwdxd,CTdxd, likdxd,
                           Zdth,Ldth,mwdth,CTdth, likdth,
                           Zdthd,Ldthd,mwdthd,CTdthd, likdthd,
                           unc_lambda):
    Xfeat = tf_feature_map(s, u)
    mu_dx,  _       = _predict_pack_mu_var_tf_tensors(Zdx,   Ldx,   mwdx,   CTdx,   k_dx,     Xfeat)
    mu_dxd, var_dxd = _predict_pack_mu_var_tf_tensors(Zdxd,  Ldxd,  mwdxd,  CTdxd,  k_dxdot,  Xfeat)
    mu_dt,  _       = _predict_pack_mu_var_tf_tensors(Zdth,  Ldth,  mwdth,  CTdth,  k_dth,    Xfeat)
    mu_dtd, _       = _predict_pack_mu_var_tf_tensors(Zdthd, Ldthd, mwdthd, CTdthd, k_dthdot, Xfeat)
    x     = s[:, 0] + mu_dx
    xdot  = s[:, 1] + mu_dxd
    th    = tf_wrap_angle(s[:, 2] + mu_dt)
    thdot = s[:, 3] + mu_dtd
    s_next = tf.stack([x, xdot, th, thdot], axis=-1)
    unc = tf.sqrt(tf.maximum(var_dxd + likdxd, tf.constant(1e-12, DTF)))
    return s_next, unc_lambda * unc, unc

def make_plan_tf_simple(H=H, K=K, sigma=MPPI_SIGMA, lam=MPPI_LAMBDA,
                        u_min=U_MIN, u_max=U_MAX, unc_lambda=UNC_LAMBDA):
    sigma = float(sigma); lam = float(lam)
    u_min = float(u_min); u_max = float(u_max)
    unc_lambda = float(unc_lambda)

    @tf.function(reduce_retracing=True)
    def plan_tf(s0, u_mean,
                Zdx,Ldx,mwdx,CTdx, likdx,
                Zdxd,Ldxd,mwdxd,CTdxd, likdxd,
                Zdth,Ldth,mwdth,CTdth, likdth,
                Zdthd,Ldthd,mwdthd,CTdthd, likdthd):
        s0 = tf.cast(tf.reshape(s0, (4,)), DTF)
        u_mean = tf.cast(tf.reshape(u_mean, (H,)), DTF)

        eps = tf.random.normal((K, H), mean=0.0, stddev=sigma, dtype=DTF)
        U = tf.clip_by_value(tf.expand_dims(u_mean, 0) + eps, u_min, u_max)

        s = tf.repeat(tf.expand_dims(s0, 0), repeats=K, axis=0)
        cost = tf.zeros((K,), dtype=DTF)
        unc_sum = tf.zeros((K,), dtype=DTF)

        for t in tf.range(H):
            u_t = U[:, t]
            s, unc_bonus, unc = _gp_step_batch_tf_packs(
                s, u_t,
                Zdx,Ldx,mwdx,CTdx, likdx,
                Zdxd,Ldxd,mwdxd,CTdxd, likdxd,
                Zdth,Ldth,mwdth,CTdth, likdth,
                Zdthd,Ldthd,mwdthd,CTdthd, likdthd,
                tf.constant(unc_lambda, DTF),
            )
            cost = cost + stage_cost_tf(s, u_t) + unc_bonus
            unc_sum = unc_sum + unc

        beta = tf.reduce_min(cost)
        w = tf.exp(-(cost - beta) / tf.constant(lam, DTF))
        w_sum = tf.reduce_sum(w) + tf.constant(1e-12, DTF)

        u_mean_new = tf.reduce_sum(tf.expand_dims(w, 1) * U, axis=0) / w_sum
        u0 = u_mean_new[0]
        unc_w = tf.reduce_sum(w * (unc_sum / tf.cast(H, DTF))) / w_sum
        return u0, u_mean_new, unc_w
    return plan_tf

plan_tf = make_plan_tf_simple()

def packs_from_global(models, Z_global):
    m_dx, m_dxdot, m_dth, m_dthdot = models
    p_dx     = build_pack_from_model(m_dx,     Z_global, jitter=1e-6, use_cached=False)
    p_dxdot  = build_pack_from_model(m_dxdot,  Z_global, jitter=1e-6, use_cached=False)
    p_dth    = build_pack_from_model(m_dth,    Z_global, jitter=1e-6, use_cached=False)
    p_dthdot = build_pack_from_model(m_dthdot, Z_global, jitter=1e-6, use_cached=False)
    return (
        p_dx.Z, p_dx.L, p_dx.m_w, p_dx.C_T, p_dx.lik_var,
        p_dxdot.Z, p_dxdot.L, p_dxdot.m_w, p_dxdot.C_T, p_dxdot.lik_var,
        p_dth.Z, p_dth.L, p_dth.m_w, p_dth.C_T, p_dth.lik_var,
        p_dthdot.Z, p_dthdot.L, p_dthdot.m_w, p_dthdot.C_T, p_dthdot.lik_var,
    )

# Run SVGP_GLOBAL
METHOD = "SVGP_GLOBAL"
print(f"\n===========================\nRunning {METHOD} (hold {SUCCESS_HOLD_STEPS} steps)\n===========================\n")

rng0 = np.random.default_rng(SEED_BASE)
M_GLOBAL = int(min(M_GLOBAL_INIT, X0.shape[0]))
Z_GLOBAL = np.asarray(X0[rng0.choice(X0.shape[0], size=M_GLOBAL, replace=False)], dtype=np.float64)

m_dx     = make_svgp(Z_GLOBAL, k_dx,     noise=NOISE_INIT, train_hypers=False)
m_dxdot  = make_svgp(Z_GLOBAL, k_dxdot,  noise=NOISE_INIT, train_hypers=False)
m_dth    = make_svgp(Z_GLOBAL, k_dth,    noise=NOISE_INIT, train_hypers=False)
m_dthdot = make_svgp(Z_GLOBAL, k_dthdot, noise=NOISE_INIT, train_hypers=False)

print("Initial SVGP training...")
_ = svgp_train_steps(m_dx,     X0, Ydx0,     steps=250, lr=0.02)
_ = svgp_train_steps(m_dxdot,  X0, Ydxdot0,  steps=250, lr=0.02)
_ = svgp_train_steps(m_dth,    X0, Ydth0,    steps=250, lr=0.02)
_ = svgp_train_steps(m_dthdot, X0, Ydthdot0, steps=250, lr=0.02)

models = (m_dx, m_dxdot, m_dth, m_dthdot)
pack_tensors = packs_from_global(models, Z_GLOBAL)

REPLAY_TRAIN_MAX = 2500
SVGP_STEPS_PER_UPDATE = 80

for run in range(N_RUNS):
    seed = SEED_BASE + run
    rng = np.random.default_rng(seed)
    env = make_env(render_mode=None, seed=seed, max_episode_steps=MAX_STEPS_PER_EP, start_down=True, edge_respawn=True)
    do_live = bool(LIVE_RENDER and ((not LIVE_ONLY_FIRST_RUN) or (run == 0)))

    # Hard reset models/inducing state for this run (fair independent runs)
    rng_init_run = np.random.default_rng(seed)
    M_GLOBAL = int(min(M_GLOBAL_INIT, X0.shape[0]))
    Z_GLOBAL = np.asarray(X0[rng_init_run.choice(X0.shape[0], size=M_GLOBAL, replace=False)], dtype=np.float64)
    m_dx     = make_svgp(Z_GLOBAL, k_dx,     noise=NOISE_INIT, train_hypers=False)
    m_dxdot  = make_svgp(Z_GLOBAL, k_dxdot,  noise=NOISE_INIT, train_hypers=False)
    m_dth    = make_svgp(Z_GLOBAL, k_dth,    noise=NOISE_INIT, train_hypers=False)
    m_dthdot = make_svgp(Z_GLOBAL, k_dthdot, noise=NOISE_INIT, train_hypers=False)
    _ = svgp_train_steps(m_dx,     X0, Ydx0,     steps=250, lr=0.02)
    _ = svgp_train_steps(m_dxdot,  X0, Ydxdot0,  steps=250, lr=0.02)
    _ = svgp_train_steps(m_dth,    X0, Ydth0,    steps=250, lr=0.02)
    _ = svgp_train_steps(m_dthdot, X0, Ydthdot0, steps=250, lr=0.02)
    models = (m_dx, m_dxdot, m_dth, m_dthdot)
    pack_tensors = packs_from_global(models, Z_GLOBAL)

    run_frames = []
    if CAPTURE_REPLAY:
        run_frames.extend([_title_frame(f"{METHOD} | RUN {run} | start", W=REPLAY_SIZE[0], H=REPLAY_SIZE[1])] * TITLE_HOLD_FRAMES)

    # replay buffer
    Xmem = [np.asarray(X0[i], dtype=np.float64) for i in range(X0.shape[0])]
    Ydx_mem     = [float(Ydx0[i, 0]) for i in range(Ydx0.shape[0])]
    Ydxdot_mem  = [float(Ydxdot0[i, 0]) for i in range(Ydxdot0.shape[0])]
    Ydth_mem    = [float(Ydth0[i, 0]) for i in range(Ydth0.shape[0])]
    Ydthdot_mem = [float(Ydthdot0[i, 0]) for i in range(Ydthdot0.shape[0])]

    u_mean = np.zeros((H,), dtype=np.float64)

    pred_time_step, train_time_step, rebuild_time_step, wall_time_step, vis_time_step = [], [], [], [], []
    u_exec_step, unc_exec_step = [], []
    update_flag_step = []
    update_t_global, update_rmse_mean, update_nlpd_mean, update_cover2_mean, update_std_mean = [], [], [], [], []

    ep_returns, ep_lengths, ep_success, ep_first_success_t_global = [], [], [], []
    ep_control_l2, ep_u_abs_mean, ep_unc_mean, ep_unc_max = [], [], [], []
    ep_hold_max_list = []

    total_reward_run = 0.0
    run_success = False
    first_success_t = None
    t_global = 0

    for ep in range(N_EPISODES_PER_RUN):
        obs, _ = env.reset(seed=_shared_ep_reset_seed(run, ep))
        x, xdot, th, thdot = obs_to_state(obs)

        if do_live and (not LIVE_ONLY_FIRST_EP or ep == 0):
            LIVE_VIEWER.reset()
        if CAPTURE_REPLAY:
            run_frames.extend([_title_frame(f"{METHOD} | RUN {run} | EP {ep}", W=REPLAY_SIZE[0], H=REPLAY_SIZE[1])] * TITLE_HOLD_FRAMES)

        ep_reward = 0.0; ep_steps = 0
        ep_u2_sum = 0.0; ep_u_abs_sum = 0.0
        ep_unc_sum = 0.0; ep_unc_max_v = 0.0

        hold_count = 0; hold_max = 0
        ep_hold_complete_t = None

        ep_prog = TextProgress(MAX_STEPS_PER_EP, prefix=f"{METHOD} run{run} ep{ep} step ", every=10)

        for step in range(MAX_STEPS_PER_EP):
            tp0 = time.perf_counter()
            s0 = np.array([x, xdot, th, thdot], dtype=np.float64)

            u0_tf, u_mean_new_tf, unc_w_tf = plan_tf(s0, u_mean, *pack_tensors)
            u0 = float(u0_tf.numpy())
            u_mean_new = u_mean_new_tf.numpy()

            u_mean = np.roll(u_mean_new, -1)
            u_mean[-1] = 0.0

            pred_dt = float(time.perf_counter() - tp0)

            obs1, r, terminated, truncated, info = env.step(np.array([u0], dtype=np.float32))
            x1, xdot1, th1, thdot1 = obs_to_state(obs1)

            if do_live and (not LIVE_ONLY_FIRST_EP or ep == 0):
                LIVE_VIEWER.maybe_update(
                    step_i=t_global, x=x1, theta=th1,
                    text=f"{METHOD} | r{run} e{ep} t={t_global} | r={float(r):+.3f} u={u0:+.3f} hold={hold_count}/{SUCCESS_HOLD_STEPS}"
                )

            if CAPTURE_REPLAY and ((t_global % REPLAY_EVERY_STEPS) == 0):
                run_frames.append(_render_with_text(
                    x1, th1,
                    text=f"{METHOD} r{run} e{ep} t{t_global} | r={float(r):+.2f} u={u0:+.2f} hold={hold_count}",
                    W=REPLAY_SIZE[0], H=REPLAY_SIZE[1]
                ))

            succ_inst = is_success_state(
                x1, xdot1, th1, thdot1,
                cos_th_min=SUCCESS_COS_TH_MIN,
                x_abs_max=SUCCESS_X_ABS_MAX,
                xdot_abs_max=SUCCESS_XDOT_ABS_MAX,
                thdot_abs_max=SUCCESS_THDOT_ABS_MAX
            )

            if bool(info.get("respawned", False)) or bool(info.get("post_respawn_cooldown", 0)):
                hold_count = 0
            else:
                hold_count = (hold_count + 1) if succ_inst else 0

            hold_max = max(hold_max, hold_count)
            succ_hold = (hold_count >= SUCCESS_HOLD_STEPS)

            if succ_hold and (ep_hold_complete_t is None):
                ep_hold_complete_t = int(t_global)
            if succ_hold and (not run_success):
                run_success = True
                first_success_t = int(t_global)

            # append to replay buffer
            feat = state_to_features(x, xdot, th, thdot, u0).astype(np.float64)
            d_x = (x1 - x); d_xdot = (xdot1 - xdot)
            d_th = wrap_pi(th1 - th); d_thdot = (thdot1 - thdot)

            if (not bool(info.get("respawned", False))) and (not bool(info.get("post_respawn_cooldown", 0))):
                Xmem.append(feat)
                Ydx_mem.append(float(d_x))
                Ydxdot_mem.append(float(d_xdot))
                Ydth_mem.append(float(d_th))
                Ydthdot_mem.append(float(d_thdot))

            ep_reward += float(r); ep_steps += 1
            ep_u2_sum += u0*u0; ep_u_abs_sum += abs(u0)
            ep_unc_sum += float(unc_w_tf.numpy()); ep_unc_max_v = max(ep_unc_max_v, float(unc_w_tf.numpy()))
            total_reward_run += float(r)

            # online update
            trn_dt = 0.0; reb_dt = 0.0
            did_update = False
            if ((t_global + 1) % int(UPDATE_EVERY) == 0) and (len(Xmem) > 200):
                did_update = True
                nmem = len(Xmem)
                bsz = int(min(REPLAY_TRAIN_MAX, nmem))
                idx = rng.choice(nmem, size=bsz, replace=False)

                Xb = np.asarray([Xmem[i] for i in idx], dtype=np.float64)
                ydx_b     = np.asarray([[Ydx_mem[i]] for i in idx], dtype=np.float64)
                ydxdot_b  = np.asarray([[Ydxdot_mem[i]] for i in idx], dtype=np.float64)
                ydth_b    = np.asarray([[Ydth_mem[i]] for i in idx], dtype=np.float64)
                ydthdot_b = np.asarray([[Ydthdot_mem[i]] for i in idx], dtype=np.float64)

                tu0 = time.perf_counter()
                _ = svgp_train_steps(m_dx,     Xb, ydx_b,     steps=SVGP_STEPS_PER_UPDATE, lr=0.02)
                _ = svgp_train_steps(m_dxdot,  Xb, ydxdot_b,  steps=SVGP_STEPS_PER_UPDATE, lr=0.02)
                _ = svgp_train_steps(m_dth,    Xb, ydth_b,    steps=SVGP_STEPS_PER_UPDATE, lr=0.02)
                _ = svgp_train_steps(m_dthdot, Xb, ydthdot_b, steps=SVGP_STEPS_PER_UPDATE, lr=0.02)
                trn_dt = float(time.perf_counter() - tu0)

                try:
                    met = eval_multihead_batch(
                        [m_dx, m_dxdot, m_dth, m_dthdot],
                        Xb, [ydx_b, ydxdot_b, ydth_b, ydthdot_b],
                        add_likelihood_var=True
                    )
                    update_t_global.append(int(t_global))
                    update_rmse_mean.append(float(met["rmse_mean"]))
                    update_nlpd_mean.append(float(met["nlpd_mean"]))
                    update_cover2_mean.append(float(met["cover2_mean"]))
                    update_std_mean.append(float(met["std_mean_mean"]))
                except Exception:
                    update_t_global.append(int(t_global))
                    update_rmse_mean.append(np.nan)
                    update_nlpd_mean.append(np.nan)
                    update_cover2_mean.append(np.nan)
                    update_std_mean.append(np.nan)

                trb0 = time.perf_counter()
                models = (m_dx, m_dxdot, m_dth, m_dthdot)
                pack_tensors = packs_from_global(models, Z_GLOBAL)
                reb_dt = float(time.perf_counter() - trb0)

            wall_dt = pred_dt + trn_dt + reb_dt
            pred_time_step.append(pred_dt)
            train_time_step.append(trn_dt)
            rebuild_time_step.append(reb_dt)
            wall_time_step.append(wall_dt)
            vis_time_step.append(0.0)
            u_exec_step.append(float(u0))
            unc_exec_step.append(float(unc_w_tf.numpy()))
            update_flag_step.append(1.0 if did_update else 0.0)

            ep_prog.update(step, extra=f"| r={float(r):+.2f} u={u0:+.2f} hold={hold_count}/{SUCCESS_HOLD_STEPS} upd={'Y' if did_update else 'n'}")

            x, xdot, th, thdot = x1, xdot1, th1, thdot1
            t_global += 1

            if STOP_ON_HOLD_SUCCESS and succ_hold:
                break

        ep_prog.close()

        ep_returns.append(float(ep_reward))
        ep_lengths.append(int(ep_steps))
        ep_success.append(bool(ep_hold_complete_t is not None))
        ep_first_success_t_global.append(int(ep_hold_complete_t) if ep_hold_complete_t is not None else None)
        ep_control_l2.append(float(ep_u2_sum))
        ep_u_abs_mean.append(float(ep_u_abs_sum / max(ep_steps, 1)))
        ep_unc_mean.append(float(ep_unc_sum / max(ep_steps, 1)))
        ep_unc_max.append(float(ep_unc_max_v))
        ep_hold_max_list.append(int(hold_max))

    env.close()

    run_stats = dict(
        total_reward=float(total_reward_run),
        success=bool(run_success),
        first_success_t_global=int(first_success_t) if first_success_t is not None else None,

        ep_returns=ep_returns,
        ep_lengths=ep_lengths,
        ep_success=ep_success,
        ep_first_success_t_global=ep_first_success_t_global,
        ep_control_l2=ep_control_l2,
        ep_u_abs_mean=ep_u_abs_mean,
        ep_unc_mean=ep_unc_mean,
        ep_unc_max=ep_unc_max,
        ep_hold_max=ep_hold_max_list,
    )
    run_traces = dict(
        wall_time_step=wall_time_step,
        pred_time_step=pred_time_step,
        train_time_step=train_time_step,
        rebuild_time_step=rebuild_time_step,
        vis_time_step=vis_time_step,
        u_exec_step=u_exec_step,
        unc_exec_step=unc_exec_step,
        update_flag_step=update_flag_step,
        update_t_global=update_t_global,
        update_rmse_mean=update_rmse_mean,
        update_nlpd_mean=update_nlpd_mean,
        update_cover2_mean=update_cover2_mean,
        update_std_mean=update_std_mean,
    )
    registry_add_run(METHOD, run_stats, run_traces)

    print(f"[{METHOD}] run {run+1}/{N_RUNS}: reward={total_reward_run:.2f} success={run_success} first_success_t={first_success_t}")

    if CAPTURE_REPLAY:
        print(f"\n▶️ Replay: {METHOD} RUN {run} (all episodes) | frames={len(run_frames)} | fps={REPLAY_FPS}")
        _show_replay(run_frames, fps=REPLAY_FPS)

print(f"\n✅ {METHOD} finished. Registry runs = {len(EVAL_REGISTRY[METHOD]['run_stats'])}")

# ===========================
# Cell 8 — OSGPR_GLOBAL (MPPI + Online OSGPR-VFE, global-only)
# Success = hold upright for SUCCESS_HOLD_STEPS consecutive steps
# Progress: TextProgress; Live + replay
# ===========================

import numpy as np
import tensorflow as tf
import gpflow
import time
import sys

from IPython.display import HTML, display
import matplotlib.pyplot as plt
from matplotlib import animation
from PIL import Image, ImageDraw

# Preconditions
for name in ["X0", "Ydx0", "Ydxdot0", "Ydth0", "Ydthdot0"]:
    assert name in globals(), f"Run Cell 2 first: missing {name}"
for name in ["OSGPR_VFE", "train_osgpr", "prior_summary", "extract_summary_from_model", "greedy_dopt_anchors_from_K"]:
    assert name in globals(), f"Run Cell 3 first: missing {name}"
for name in ["build_pack_from_model", "stage_cost_tf", "tf_feature_map", "tf_wrap_angle", "is_success_state"]:
    assert name in globals(), f"Run Cell 4 first: missing {name}"
for name in ["registry_add_run", "eval_multihead_batch", "EVAL_REGISTRY"]:
    assert name in globals(), f"Run Cell 5 first: missing {name}"
assert "render_cartpole_frame_from_state" in globals(), "Run Cell 2 first (PIL renderer)"

tf.keras.backend.set_floatx("float64")
gpflow.config.set_default_float(np.float64)
gpflow.config.set_default_jitter(1e-6)
DTF = tf.float64

SUCCESS_HOLD_STEPS = int(globals().get("SUCCESS_HOLD_STEPS", 200))
STOP_ON_HOLD_SUCCESS = bool(globals().get("STOP_ON_HOLD_SUCCESS", True))

class TextProgress:
    def __init__(self, total, prefix="", every=10):
        self.total = int(total); self.prefix = str(prefix); self.every = int(max(1, every))
        self.t0 = time.perf_counter(); self.last_i = -1
    def update(self, i, extra=""):
        i = int(i)
        if (i == self.total - 1) or ((i % self.every) == 0 and i != self.last_i):
            dt = time.perf_counter() - self.t0
            rate = (i + 1) / max(dt, 1e-9)
            sys.stdout.write(f"\r{self.prefix}{i+1}/{self.total} | {rate:6.1f} it/s | {dt:6.1f}s {extra}")
            sys.stdout.flush()
            self.last_i = i
    def close(self):
        sys.stdout.write("\n"); sys.stdout.flush()

LIVE_RENDER = globals().get("LIVE_RENDER", True)
LIVE_ONLY_FIRST_RUN = globals().get("LIVE_ONLY_FIRST_RUN", True)
LIVE_ONLY_FIRST_EP  = globals().get("LIVE_ONLY_FIRST_EP", False)
LIVE_EVERY_STEPS = globals().get("LIVE_EVERY_STEPS", 3)
LIVE_MIN_DT = globals().get("LIVE_MIN_DT", 0.05)
LIVE_SIZE = globals().get("LIVE_SIZE", (720, 450))

CAPTURE_REPLAY = False
REPLAY_EVERY_STEPS = 2
REPLAY_FPS = 20
REPLAY_SIZE = (720, 450)
TITLE_HOLD_FRAMES = 12

if "LIVE_VIEWER" not in globals():
    class _LiveViewerFallback:
        def __init__(self, enabled=True, every_steps=3, min_dt=0.05, size=(720, 450)):
            self.enabled = bool(enabled); self.every_steps = int(every_steps); self.min_dt = float(min_dt); self.size = tuple(size)
            self._img_handle = None; self._txt_handle = None; self._last_step = -10**9; self._last_t = 0.0
        def reset(self):
            self._last_step = -10**9; self._last_t = 0.0
        def maybe_update(self, step_i, x, theta, text=""):
            if not self.enabled: return
            now = time.perf_counter()
            if (step_i - self._last_step) < self.every_steps: return
            if (now - self._last_t) < self.min_dt: return
            frame = render_cartpole_frame_from_state(x, theta, x_threshold=2.4, W=self.size[0], H=self.size[1])
            img = Image.fromarray(frame)
            if self._img_handle is None:
                self._img_handle = display(img, display_id=True)
                self._txt_handle = display(text, display_id=True)
            else:
                self._img_handle.update(img); self._txt_handle.update(text)
            self._last_step = int(step_i); self._last_t = now
    LIVE_VIEWER = _LiveViewerFallback(enabled=LIVE_RENDER, every_steps=LIVE_EVERY_STEPS, min_dt=LIVE_MIN_DT, size=LIVE_SIZE)

def _render_with_text(x, theta, text, W=720, H=450):
    frame = render_cartpole_frame_from_state(x, theta, x_threshold=2.4, W=W, H=H)
    img = Image.fromarray(frame)
    dr = ImageDraw.Draw(img)
    dr.rectangle([8, 8, W - 8, 44], fill=(255, 255, 255))
    dr.text((14, 14), text, fill=(0, 0, 0))
    return np.asarray(img, dtype=np.uint8)

def _title_frame(text, W=720, H=450):
    img = Image.new("RGB", (W, H), (245, 245, 245))
    dr = ImageDraw.Draw(img); dr.text((18, 18), text, fill=(0, 0, 0))
    return np.asarray(img, dtype=np.uint8)

def _show_replay(frames, fps=20):
    if len(frames) == 0:
        print("⚠️ No replay frames.")
        return
    fig, ax = plt.subplots(figsize=(8, 5))
    im = ax.imshow(frames[0]); ax.axis("off")
    def update(i):
        im.set_data(frames[i]); return [im]
    ani = animation.FuncAnimation(fig, update, frames=len(frames), interval=1000.0 / fps, blit=True)
    plt.close(fig); display(HTML(ani.to_jshtml()))

# Kernels
def make_kernel_6d():
    return gpflow.kernels.Matern52(lengthscales=np.ones(6, dtype=np.float64), variance=1.0)

k_dx     = make_kernel_6d()
k_dxdot  = make_kernel_6d()
k_dth    = make_kernel_6d()
k_dthdot = make_kernel_6d()

# OSGPR init + update
def init_osgpr_from_prior(X, Y, Z, kernel, noise=NOISE_INIT, iters=650, lr=0.02):
    mu0, Su0, Kaa0, Z0 = prior_summary(kernel, Z)
    m = OSGPR_VFE(
        data=(np.asarray(X, dtype=np.float64), np.asarray(Y, dtype=np.float64)),
        kernel=kernel,
        mu_old=mu0, Su_old=Su0, Kaa_old=Kaa0, Z_old=Z0,
        Z=np.asarray(Z, dtype=np.float64),
    )
    m.likelihood.variance.assign(float(noise))
    t_sec, _ = train_osgpr(m, iters=iters, lr=lr, clip_norm=10.0)
    m.build_predict_cache()
    return m, float(t_sec)

def osgpr_stream_update_fixedZ(model_old, X_new, Y_new, Z_fixed,
                               iters=OSGPR_ITERS_PER_UPDATE, lr=OSGPR_LR, noise=NOISE_INIT):
    mu_old, Su_old, Kaa_old, Z_old = extract_summary_from_model(model_old)
    m = OSGPR_VFE(
        data=(np.asarray(X_new, dtype=np.float64), np.asarray(Y_new, dtype=np.float64)),
        kernel=model_old.kernel,
        mu_old=mu_old, Su_old=Su_old, Kaa_old=Kaa_old, Z_old=Z_old,
        Z=np.asarray(Z_fixed, dtype=np.float64),
    )
    m.likelihood.variance.assign(float(noise))
    t_sec, _ = train_osgpr(m, iters=iters, lr=lr, clip_norm=10.0)
    m.build_predict_cache()
    return m, float(t_sec)

# MPPI plan_tf (simple global packs)
@tf.function
def _predict_pack_mu_var_tf_tensors(Z, L, mw, CT, kernel, Xfeat):
    Kzx = kernel(Z, Xfeat)
    A = tf.linalg.triangular_solve(L, Kzx, lower=True)
    mu = tf.matmul(A, mw, transpose_a=True); mu = tf.reshape(mu, (-1,))
    kxx = kernel(Xfeat, full_cov=False)
    q = tf.reduce_sum(tf.square(A), axis=0)
    E = tf.matmul(CT, A)
    var_add = tf.reduce_sum(tf.square(E), axis=0)
    var = tf.maximum(kxx - q + var_add, tf.constant(1e-12, DTF))
    return mu, var

@tf.function
def _gp_step_batch_tf_packs(s, u,
                           Zdx,Ldx,mwdx,CTdx, likdx,
                           Zdxd,Ldxd,mwdxd,CTdxd, likdxd,
                           Zdth,Ldth,mwdth,CTdth, likdth,
                           Zdthd,Ldthd,mwdthd,CTdthd, likdthd,
                           unc_lambda):
    Xfeat = tf_feature_map(s, u)
    mu_dx,  _       = _predict_pack_mu_var_tf_tensors(Zdx,   Ldx,   mwdx,   CTdx,   k_dx,     Xfeat)
    mu_dxd, var_dxd = _predict_pack_mu_var_tf_tensors(Zdxd,  Ldxd,  mwdxd,  CTdxd,  k_dxdot,  Xfeat)
    mu_dt,  _       = _predict_pack_mu_var_tf_tensors(Zdth,  Ldth,  mwdth,  CTdth,  k_dth,    Xfeat)
    mu_dtd, _       = _predict_pack_mu_var_tf_tensors(Zdthd, Ldthd, mwdthd, CTdthd, k_dthdot, Xfeat)
    x     = s[:, 0] + mu_dx
    xdot  = s[:, 1] + mu_dxd
    th    = tf_wrap_angle(s[:, 2] + mu_dt)
    thdot = s[:, 3] + mu_dtd
    s_next = tf.stack([x, xdot, th, thdot], axis=-1)
    unc = tf.sqrt(tf.maximum(var_dxd + likdxd, tf.constant(1e-12, DTF)))
    return s_next, unc_lambda * unc, unc

def make_plan_tf_simple(H=H, K=K, sigma=MPPI_SIGMA, lam=MPPI_LAMBDA,
                        u_min=U_MIN, u_max=U_MAX, unc_lambda=UNC_LAMBDA):
    sigma = float(sigma); lam = float(lam)
    u_min = float(u_min); u_max = float(u_max)
    unc_lambda = float(unc_lambda)

    @tf.function(reduce_retracing=True)
    def plan_tf(s0, u_mean,
                Zdx,Ldx,mwdx,CTdx, likdx,
                Zdxd,Ldxd,mwdxd,CTdxd, likdxd,
                Zdth,Ldth,mwdth,CTdth, likdth,
                Zdthd,Ldthd,mwdthd,CTdthd, likdthd):
        s0 = tf.cast(tf.reshape(s0, (4,)), DTF)
        u_mean = tf.cast(tf.reshape(u_mean, (H,)), DTF)

        eps = tf.random.normal((K, H), mean=0.0, stddev=sigma, dtype=DTF)
        U = tf.clip_by_value(tf.expand_dims(u_mean, 0) + eps, u_min, u_max)

        s = tf.repeat(tf.expand_dims(s0, 0), repeats=K, axis=0)
        cost = tf.zeros((K,), dtype=DTF)
        unc_sum = tf.zeros((K,), dtype=DTF)

        for t in tf.range(H):
            u_t = U[:, t]
            s, unc_bonus, unc = _gp_step_batch_tf_packs(
                s, u_t,
                Zdx,Ldx,mwdx,CTdx, likdx,
                Zdxd,Ldxd,mwdxd,CTdxd, likdxd,
                Zdth,Ldth,mwdth,CTdth, likdth,
                Zdthd,Ldthd,mwdthd,CTdthd, likdthd,
                tf.constant(unc_lambda, DTF),
            )
            cost = cost + stage_cost_tf(s, u_t) + unc_bonus
            unc_sum = unc_sum + unc

        beta = tf.reduce_min(cost)
        w = tf.exp(-(cost - beta) / tf.constant(lam, DTF))
        w_sum = tf.reduce_sum(w) + tf.constant(1e-12, DTF)

        u_mean_new = tf.reduce_sum(tf.expand_dims(w, 1) * U, axis=0) / w_sum
        u0 = u_mean_new[0]
        unc_w = tf.reduce_sum(w * (unc_sum / tf.cast(H, DTF))) / w_sum
        return u0, u_mean_new, unc_w
    return plan_tf

plan_tf = make_plan_tf_simple()

def packs_from_global(models, Z_global):
    m_dx, m_dxdot, m_dth, m_dthdot = models
    p_dx     = build_pack_from_model(m_dx,     Z_global, jitter=1e-6, use_cached=True)
    p_dxdot  = build_pack_from_model(m_dxdot,  Z_global, jitter=1e-6, use_cached=True)
    p_dth    = build_pack_from_model(m_dth,    Z_global, jitter=1e-6, use_cached=True)
    p_dthdot = build_pack_from_model(m_dthdot, Z_global, jitter=1e-6, use_cached=True)
    return (
        p_dx.Z, p_dx.L, p_dx.m_w, p_dx.C_T, p_dx.lik_var,
        p_dxdot.Z, p_dxdot.L, p_dxdot.m_w, p_dxdot.C_T, p_dxdot.lik_var,
        p_dth.Z, p_dth.L, p_dth.m_w, p_dth.C_T, p_dth.lik_var,
        p_dthdot.Z, p_dthdot.L, p_dthdot.m_w, p_dthdot.C_T, p_dthdot.lik_var,
    )

def refit_Z_global_multihead(Xcand, M_new, kernels, lam=1e-6, rng=None):
    if rng is None:
        rng = np.random.default_rng(0)
    Xcand = np.asarray(Xcand, dtype=np.float64)
    N = Xcand.shape[0]
    if N <= M_new:
        return Xcand.copy()
    Ncap = min(N, 1800)
    if N > Ncap:
        idx = rng.choice(N, size=Ncap, replace=False)
        Xuse = Xcand[idx]
    else:
        Xuse = Xcand
    Ksum = None
    for k in kernels:
        Kk = k.K(Xuse).numpy()
        Ksum = Kk if Ksum is None else (Ksum + Kk)
    Kavg = Ksum / float(len(kernels))
    Kavg = 0.5 * (Kavg + Kavg.T) + float(lam) * np.eye(Kavg.shape[0], dtype=np.float64)
    sel = greedy_dopt_anchors_from_K(Kavg, m_anchors=int(M_new), lam=lam)
    return Xuse[sel]

# Run OSGPR_GLOBAL
METHOD = "OSGPR_GLOBAL"
print(f"\n===========================\nRunning {METHOD} (hold {SUCCESS_HOLD_STEPS} steps)\n===========================\n")

rng0 = np.random.default_rng(SEED_BASE)
M_GLOBAL = int(min(M_GLOBAL_INIT, X0.shape[0]))
Z_GLOBAL = np.asarray(X0[rng0.choice(X0.shape[0], size=M_GLOBAL, replace=False)], dtype=np.float64)

print("Training initial OSGPR models...")
m_dx,     _ = init_osgpr_from_prior(X0, Ydx0,     Z_GLOBAL, k_dx,     noise=NOISE_INIT, iters=650, lr=0.02)
m_dxdot,  _ = init_osgpr_from_prior(X0, Ydxdot0,  Z_GLOBAL, k_dxdot,  noise=NOISE_INIT, iters=650, lr=0.02)
m_dth,    _ = init_osgpr_from_prior(X0, Ydth0,    Z_GLOBAL, k_dth,    noise=NOISE_INIT, iters=650, lr=0.02)
m_dthdot, _ = init_osgpr_from_prior(X0, Ydthdot0, Z_GLOBAL, k_dthdot, noise=NOISE_INIT, iters=650, lr=0.02)

models = (m_dx, m_dxdot, m_dth, m_dthdot)
pack_tensors = packs_from_global(models, Z_GLOBAL)

REFIT_REPLAY_MAX = 4000

for run in range(N_RUNS):
    seed = SEED_BASE + run
    rng = np.random.default_rng(seed)
    env = make_env(render_mode=None, seed=seed, max_episode_steps=MAX_STEPS_PER_EP, start_down=True, edge_respawn=True)
    do_live = bool(LIVE_RENDER and ((not LIVE_ONLY_FIRST_RUN) or (run == 0)))

    # Hard reset models/inducing state for this run (fair independent runs)
    rng_init_run = np.random.default_rng(seed)
    M_GLOBAL = int(min(M_GLOBAL_INIT, X0.shape[0]))
    Z_GLOBAL = np.asarray(X0[rng_init_run.choice(X0.shape[0], size=M_GLOBAL, replace=False)], dtype=np.float64)
    m_dx,     _ = init_osgpr_from_prior(X0, Ydx0,     Z_GLOBAL, k_dx,     noise=NOISE_INIT, iters=650, lr=0.02)
    m_dxdot,  _ = init_osgpr_from_prior(X0, Ydxdot0,  Z_GLOBAL, k_dxdot,  noise=NOISE_INIT, iters=650, lr=0.02)
    m_dth,    _ = init_osgpr_from_prior(X0, Ydth0,    Z_GLOBAL, k_dth,    noise=NOISE_INIT, iters=650, lr=0.02)
    m_dthdot, _ = init_osgpr_from_prior(X0, Ydthdot0, Z_GLOBAL, k_dthdot, noise=NOISE_INIT, iters=650, lr=0.02)
    models = (m_dx, m_dxdot, m_dth, m_dthdot)
    pack_tensors = packs_from_global(models, Z_GLOBAL)

    run_frames = []
    if CAPTURE_REPLAY:
        run_frames.extend([_title_frame(f"{METHOD} | RUN {run} | start", W=REPLAY_SIZE[0], H=REPLAY_SIZE[1])] * TITLE_HOLD_FRAMES)

    Xmem = [np.asarray(X0[i], dtype=np.float64) for i in range(X0.shape[0])]
    Ydx_mem     = [float(Ydx0[i, 0]) for i in range(Ydx0.shape[0])]
    Ydxdot_mem  = [float(Ydxdot0[i, 0]) for i in range(Ydxdot0.shape[0])]
    Ydth_mem    = [float(Ydth0[i, 0]) for i in range(Ydth0.shape[0])]
    Ydthdot_mem = [float(Ydthdot0[i, 0]) for i in range(Ydthdot0.shape[0])]

    Xbuf, ydx_buf, ydxdot_buf, ydth_buf, ydthdot_buf = [], [], [], [], []
    u_mean = np.zeros((H,), dtype=np.float64)

    pred_time_step, train_time_step, rebuild_time_step, wall_time_step, vis_time_step = [], [], [], [], []
    u_exec_step, unc_exec_step = [], []
    update_flag_step = []
    update_t_global, update_rmse_mean, update_nlpd_mean, update_cover2_mean, update_std_mean = [], [], [], [], []

    ep_returns, ep_lengths, ep_success, ep_first_success_t_global = [], [], [], []
    ep_control_l2, ep_u_abs_mean, ep_unc_mean, ep_unc_max = [], [], [], []
    ep_hold_max_list = []

    total_reward_run = 0.0
    run_success = False
    first_success_t = None
    t_global = 0

    for ep in range(N_EPISODES_PER_RUN):
        obs, _ = env.reset(seed=_shared_ep_reset_seed(run, ep))
        x, xdot, th, thdot = obs_to_state(obs)

        if do_live and (not LIVE_ONLY_FIRST_EP or ep == 0):
            LIVE_VIEWER.reset()
        if CAPTURE_REPLAY:
            run_frames.extend([_title_frame(f"{METHOD} | RUN {run} | EP {ep}", W=REPLAY_SIZE[0], H=REPLAY_SIZE[1])] * TITLE_HOLD_FRAMES)

        ep_reward = 0.0; ep_steps = 0
        ep_u2_sum = 0.0; ep_u_abs_sum = 0.0
        ep_unc_sum = 0.0; ep_unc_max_v = 0.0

        hold_count = 0; hold_max = 0
        ep_hold_complete_t = None

        ep_prog = TextProgress(MAX_STEPS_PER_EP, prefix=f"{METHOD} run{run} ep{ep} step ", every=10)

        for step in range(MAX_STEPS_PER_EP):
            tp0 = time.perf_counter()
            s0 = np.array([x, xdot, th, thdot], dtype=np.float64)

            u0_tf, u_mean_new_tf, unc_w_tf = plan_tf(s0, u_mean, *pack_tensors)
            u0 = float(u0_tf.numpy())
            u_mean_new = u_mean_new_tf.numpy()

            u_mean = np.roll(u_mean_new, -1)
            u_mean[-1] = 0.0

            pred_dt = float(time.perf_counter() - tp0)

            obs1, r, terminated, truncated, info = env.step(np.array([u0], dtype=np.float32))
            x1, xdot1, th1, thdot1 = obs_to_state(obs1)

            if do_live and (not LIVE_ONLY_FIRST_EP or ep == 0):
                LIVE_VIEWER.maybe_update(
                    step_i=t_global, x=x1, theta=th1,
                    text=f"{METHOD} | r{run} e{ep} t={t_global} | r={float(r):+.3f} u={u0:+.3f} hold={hold_count}/{SUCCESS_HOLD_STEPS}"
                )

            if CAPTURE_REPLAY and ((t_global % REPLAY_EVERY_STEPS) == 0):
                run_frames.append(_render_with_text(
                    x1, th1,
                    text=f"{METHOD} r{run} e{ep} t{t_global} | r={float(r):+.2f} u={u0:+.2f} hold={hold_count}",
                    W=REPLAY_SIZE[0], H=REPLAY_SIZE[1]
                ))

            succ_inst = is_success_state(
                x1, xdot1, th1, thdot1,
                cos_th_min=SUCCESS_COS_TH_MIN,
                x_abs_max=SUCCESS_X_ABS_MAX,
                xdot_abs_max=SUCCESS_XDOT_ABS_MAX,
                thdot_abs_max=SUCCESS_THDOT_ABS_MAX
            )
            if bool(info.get("respawned", False)) or bool(info.get("post_respawn_cooldown", 0)):
                hold_count = 0
            else:
                hold_count = (hold_count + 1) if succ_inst else 0
            hold_max = max(hold_max, hold_count)
            succ_hold = (hold_count >= SUCCESS_HOLD_STEPS)

            if succ_hold and (ep_hold_complete_t is None):
                ep_hold_complete_t = int(t_global)
            if succ_hold and (not run_success):
                run_success = True
                first_success_t = int(t_global)

            feat = state_to_features(x, xdot, th, thdot, u0).astype(np.float64)
            d_x = (x1 - x); d_xdot = (xdot1 - xdot)
            d_th = wrap_pi(th1 - th); d_thdot = (thdot1 - thdot)

            if (not bool(info.get("respawned", False))) and (not bool(info.get("post_respawn_cooldown", 0))):
                Xbuf.append(feat)
                ydx_buf.append([d_x])
                ydxdot_buf.append([d_xdot])
                ydth_buf.append([d_th])
                ydthdot_buf.append([d_thdot])

                Xmem.append(feat)
                Ydx_mem.append(float(d_x))
                Ydxdot_mem.append(float(d_xdot))
                Ydth_mem.append(float(d_th))
                Ydthdot_mem.append(float(d_thdot))

            ep_reward += float(r); ep_steps += 1
            ep_u2_sum += u0*u0; ep_u_abs_sum += abs(u0)
            ep_unc_sum += float(unc_w_tf.numpy()); ep_unc_max_v = max(ep_unc_max_v, float(unc_w_tf.numpy()))
            total_reward_run += float(r)

            trn_dt = 0.0; reb_dt = 0.0
            did_update = False

            if ((t_global + 1) % int(UPDATE_EVERY) == 0) and (len(Xbuf) > 0):
                did_update = True
                Xnew = np.asarray(Xbuf, dtype=np.float64)
                ydx_new     = np.asarray(ydx_buf, dtype=np.float64)
                ydxdot_new  = np.asarray(ydxdot_buf, dtype=np.float64)
                ydth_new    = np.asarray(ydth_buf, dtype=np.float64)
                ydthdot_new = np.asarray(ydthdot_buf, dtype=np.float64)

                tu0 = time.perf_counter()
                m_dx,     _ = osgpr_stream_update_fixedZ(m_dx,     Xnew, ydx_new,     Z_GLOBAL)
                m_dxdot,  _ = osgpr_stream_update_fixedZ(m_dxdot,  Xnew, ydxdot_new,  Z_GLOBAL)
                m_dth,    _ = osgpr_stream_update_fixedZ(m_dth,    Xnew, ydth_new,    Z_GLOBAL)
                m_dthdot, _ = osgpr_stream_update_fixedZ(m_dthdot, Xnew, ydthdot_new, Z_GLOBAL)
                trn_dt = float(time.perf_counter() - tu0)

                models = (m_dx, m_dxdot, m_dth, m_dthdot)

                try:
                    met = eval_multihead_batch(
                        [m_dx, m_dxdot, m_dth, m_dthdot],
                        Xnew, [ydx_new, ydxdot_new, ydth_new, ydthdot_new],
                        add_likelihood_var=True
                    )
                    update_t_global.append(int(t_global))
                    update_rmse_mean.append(float(met["rmse_mean"]))
                    update_nlpd_mean.append(float(met["nlpd_mean"]))
                    update_cover2_mean.append(float(met["cover2_mean"]))
                    update_std_mean.append(float(met["std_mean_mean"]))
                except Exception:
                    update_t_global.append(int(t_global))
                    update_rmse_mean.append(np.nan)
                    update_nlpd_mean.append(np.nan)
                    update_cover2_mean.append(np.nan)
                    update_std_mean.append(np.nan)

                Xbuf.clear(); ydx_buf.clear(); ydxdot_buf.clear(); ydth_buf.clear(); ydthdot_buf.clear()

                trb0 = time.perf_counter()
                pack_tensors = packs_from_global(models, Z_GLOBAL)
                reb_dt = float(time.perf_counter() - trb0)

            wall_dt = pred_dt + trn_dt + reb_dt
            pred_time_step.append(pred_dt)
            train_time_step.append(trn_dt)
            rebuild_time_step.append(reb_dt)
            wall_time_step.append(wall_dt)
            vis_time_step.append(0.0)

            u_exec_step.append(float(u0))
            unc_exec_step.append(float(unc_w_tf.numpy()))
            update_flag_step.append(1.0 if did_update else 0.0)

            ep_prog.update(step, extra=f"| r={float(r):+.2f} u={u0:+.2f} hold={hold_count}/{SUCCESS_HOLD_STEPS} upd={'Y' if did_update else 'n'}")

            x, xdot, th, thdot = x1, xdot1, th1, thdot1
            t_global += 1

            if STOP_ON_HOLD_SUCCESS and succ_hold:
                break

        ep_prog.close()

        ep_returns.append(float(ep_reward))
        ep_lengths.append(int(ep_steps))
        ep_success.append(bool(ep_hold_complete_t is not None))
        ep_first_success_t_global.append(int(ep_hold_complete_t) if ep_hold_complete_t is not None else None)
        ep_control_l2.append(float(ep_u2_sum))
        ep_u_abs_mean.append(float(ep_u_abs_sum / max(ep_steps, 1)))
        ep_unc_mean.append(float(ep_unc_sum / max(ep_steps, 1)))
        ep_unc_max.append(float(ep_unc_max_v))
        ep_hold_max_list.append(int(hold_max))

        # episode-end refit Z and re-init models
        if len(Xmem) > REFIT_REPLAY_MAX:
            idxr = rng.choice(len(Xmem), size=REFIT_REPLAY_MAX, replace=False)
        else:
            idxr = np.arange(len(Xmem))

        Xr = np.asarray([Xmem[i] for i in idxr], dtype=np.float64)
        ydx_r     = np.asarray([[Ydx_mem[i]] for i in idxr], dtype=np.float64)
        ydxdot_r  = np.asarray([[Ydxdot_mem[i]] for i in idxr], dtype=np.float64)
        ydth_r    = np.asarray([[Ydth_mem[i]] for i in idxr], dtype=np.float64)
        ydthdot_r = np.asarray([[Ydthdot_mem[i]] for i in idxr], dtype=np.float64)

        Z_GLOBAL = refit_Z_global_multihead(
            Xr, M_new=int(M_GLOBAL),
            kernels=[k_dx, k_dxdot, k_dth, k_dthdot],
            lam=1e-6, rng=rng
        )

        m_dx,     _ = init_osgpr_from_prior(Xr, ydx_r,     Z_GLOBAL, k_dx,     noise=NOISE_INIT, iters=250, lr=0.02)
        m_dxdot,  _ = init_osgpr_from_prior(Xr, ydxdot_r,  Z_GLOBAL, k_dxdot,  noise=NOISE_INIT, iters=250, lr=0.02)
        m_dth,    _ = init_osgpr_from_prior(Xr, ydth_r,    Z_GLOBAL, k_dth,    noise=NOISE_INIT, iters=250, lr=0.02)
        m_dthdot, _ = init_osgpr_from_prior(Xr, ydthdot_r, Z_GLOBAL, k_dthdot, noise=NOISE_INIT, iters=250, lr=0.02)

        models = (m_dx, m_dxdot, m_dth, m_dthdot)
        pack_tensors = packs_from_global(models, Z_GLOBAL)

    env.close()

    run_stats = dict(
        total_reward=float(total_reward_run),
        success=bool(run_success),
        first_success_t_global=int(first_success_t) if first_success_t is not None else None,

        ep_returns=ep_returns,
        ep_lengths=ep_lengths,
        ep_success=ep_success,
        ep_first_success_t_global=ep_first_success_t_global,
        ep_control_l2=ep_control_l2,
        ep_u_abs_mean=ep_u_abs_mean,
        ep_unc_mean=ep_unc_mean,
        ep_unc_max=ep_unc_max,
        ep_hold_max=ep_hold_max_list,
    )
    run_traces = dict(
        wall_time_step=wall_time_step,
        pred_time_step=pred_time_step,
        train_time_step=train_time_step,
        rebuild_time_step=rebuild_time_step,
        vis_time_step=vis_time_step,
        u_exec_step=u_exec_step,
        unc_exec_step=unc_exec_step,
        update_flag_step=update_flag_step,
        update_t_global=update_t_global,
        update_rmse_mean=update_rmse_mean,
        update_nlpd_mean=update_nlpd_mean,
        update_cover2_mean=update_cover2_mean,
        update_std_mean=update_std_mean,
    )
    registry_add_run(METHOD, run_stats, run_traces)

    print(f"[{METHOD}] run {run+1}/{N_RUNS}: reward={total_reward_run:.2f} success={run_success} first_success_t={first_success_t}")

    if CAPTURE_REPLAY:
        print(f"\n▶️ Replay: {METHOD} RUN {run} (all episodes) | frames={len(run_frames)} | fps={REPLAY_FPS}")
        _show_replay(run_frames, fps=REPLAY_FPS)

print(f"\n✅ {METHOD} finished. Registry runs = {len(EVAL_REGISTRY[METHOD]['run_stats'])}")

# ===========================
# Cell 8.5 — EXACTGP_GLOBAL (MPPI + Exact GPR baseline, subset replay retrain)
# Runs after OSGPR so it can reuse shared helpers (planner, rendering, registry).
# Can be disabled by setting ENABLE_EXACTGP_GLOBAL = False before running experiments.py
# ===========================
if bool(globals().get("ENABLE_EXACTGP_GLOBAL", True)):
    _exec_file(here / "method_exactgp.py", g)

if not args.skip_eval:
    _exec_file(here / "evaluation.py", g)


