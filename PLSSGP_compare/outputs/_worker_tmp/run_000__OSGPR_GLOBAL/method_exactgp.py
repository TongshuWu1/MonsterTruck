# ===========================
# Cell 8.5 — EXACTGP_GLOBAL (MPPI + Exact GPR baseline)
#   - Exact GP (gpflow.models.GPR) per output head
#   - Online retrain from the full replay buffer every UPDATE_EVERY steps
#   - Uses an exact posterior pack (mean/variance exact on the current full training set)
#     so it plugs into the same fast MPPI planner interface used by other methods.
# Notes:
#   - Uses the full growing replay buffer for retraining (no cap).
#   - Planning uses the same full exact-training set as the support set for the pack.
# ===========================

import numpy as np
import tensorflow as tf
import gpflow
import time

# Preconditions (shared globals are created earlier in experiments.py)
for _name in ["X0", "Ydx0", "Ydxdot0", "Ydth0", "Ydthdot0"]:
    assert _name in globals(), f"Missing {_name} (run earlier cells first)"
for _name in ["registry_add_run", "eval_multihead_batch", "EVAL_REGISTRY", "DiagSparsePosteriorPack", "make_plan_tf_simple"]:
    assert _name in globals(), f"Missing {_name} (run earlier cells first)"
for _name in ["state_to_features", "wrap_pi", "obs_to_state", "make_env", "is_success_state"]:
    assert _name in globals(), f"Missing {_name} (env/controller helpers not found)"

# Numeric config
_tf_float = np.float64
tf.keras.backend.set_floatx("float64")
gpflow.config.set_default_float(np.float64)
gpflow.config.set_default_jitter(1e-6)
DTF = tf.float64

# Exact GP baseline: use the full growing replay buffer (no cap).
# You can still control retrain cadence / optimizer iterations via config.
EXACT_MIN_MEM_FOR_UPDATE = int(globals().get("EXACT_MIN_MEM_FOR_UPDATE", 200))
EXACT_ITERS_INIT = int(globals().get("EXACT_ITERS_INIT", 220))
EXACT_ITERS_UPDATE = int(globals().get("EXACT_ITERS_UPDATE", 60))
EXACT_LR = float(globals().get("EXACT_LR", 0.02))
EXACT_GRAD_CLIP = float(globals().get("EXACT_GRAD_CLIP", 10.0))

SUCCESS_HOLD_STEPS = int(globals().get("SUCCESS_HOLD_STEPS", 200))
STOP_ON_HOLD_SUCCESS = bool(globals().get("STOP_ON_HOLD_SUCCESS", True))
PROGRESS_EVERY_STEPS = int(globals().get("PROGRESS_EVERY_STEPS", 10))

# Reuse live/replay settings defined by prior method blocks when available
LIVE_RENDER = bool(globals().get("LIVE_RENDER", True))
LIVE_ONLY_FIRST_RUN = bool(globals().get("LIVE_ONLY_FIRST_RUN", True))
LIVE_ONLY_FIRST_EP = bool(globals().get("LIVE_ONLY_FIRST_EP", False))
CAPTURE_REPLAY = bool(globals().get("CAPTURE_REPLAY", False))
REPLAY_EVERY_STEPS = int(globals().get("REPLAY_EVERY_STEPS", 2))
REPLAY_FPS = int(globals().get("REPLAY_FPS", 20))
REPLAY_SIZE = tuple(globals().get("REPLAY_SIZE", (720, 450)))
TITLE_HOLD_FRAMES = int(globals().get("TITLE_HOLD_FRAMES", 12))

# Small fallback progress printer if earlier blocks were reorganized
if "TextProgress" not in globals():
    import sys
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
            import sys
            sys.stdout.write("\n"); sys.stdout.flush()

# If no live viewer exists, create a no-op fallback
if "LIVE_VIEWER" not in globals():
    class _NoopLiveViewer:
        def reset(self):
            return None
        def maybe_update(self, *args, **kwargs):
            return None
    LIVE_VIEWER = _NoopLiveViewer()

# Replay helper fallbacks (normally defined in prior blocks)
if "_title_frame" not in globals():
    def _title_frame(text, W=720, H=450):
        return np.zeros((H, W, 3), dtype=np.uint8)
if "_render_with_text" not in globals():
    def _render_with_text(x, theta, text, W=720, H=450):
        fn = globals().get("render_cartpole_frame_from_state", None)
        if callable(fn):
            try:
                return fn(x, theta, x_threshold=2.4, W=W, H=H)
            except TypeError:
                return fn(x, theta, W=W, H=H)
        return np.zeros((H, W, 3), dtype=np.uint8)
if "_show_replay" not in globals():
    def _show_replay(frames, fps=20):
        print(f"[EXACTGP_GLOBAL] replay disabled/fallback (frames={len(frames)}, fps={fps})")

# Fresh kernels for exact baseline (independent from previously trained methods)
def _make_kernel_6d():
    return gpflow.kernels.Matern52(lengthscales=np.ones(6, dtype=np.float64), variance=1.0)

k_dx = _make_kernel_6d()
k_dxdot = _make_kernel_6d()
k_dth = _make_kernel_6d()
k_dthdot = _make_kernel_6d()


def _clone_kernel_params(dst, src):
    try:
        dst.variance.assign(src.variance.numpy())
    except Exception:
        pass
    try:
        dst.lengthscales.assign(src.lengthscales.numpy())
    except Exception:
        pass


def _make_exact_gpr(X, Y, kernel, noise=1e-4):
    X = np.asarray(X, dtype=np.float64)
    Y = np.asarray(Y, dtype=np.float64).reshape(-1, 1)
    m = gpflow.models.GPR(data=(X, Y), kernel=kernel, mean_function=None)
    try:
        m.likelihood.variance.assign(float(noise))
    except Exception:
        pass
    return m


def _train_exact_gpr(model, steps=100, lr=0.02, clip_norm=10.0):
    opt = tf.keras.optimizers.Adam(float(lr))
    @tf.function
    def _step():
        with tf.GradientTape() as tape:
            loss = model.training_loss()
        grads = tape.gradient(loss, model.trainable_variables)
        if clip_norm is not None:
            grads = [tf.clip_by_norm(g, clip_norm) if g is not None else None for g in grads]
        opt.apply_gradients([(g, v) for g, v in zip(grads, model.trainable_variables) if g is not None])
        return loss
    t0 = time.perf_counter()
    for _ in range(int(max(0, steps))):
        _ = _step()
    return float(time.perf_counter() - t0)


def _fit_exact_head(X, Y, kernel_seed=None, prev_model=None, noise=NOISE_INIT, steps=100, lr=0.02):
    ker = _make_kernel_6d()
    if kernel_seed is not None:
        _clone_kernel_params(ker, kernel_seed)
    if prev_model is not None:
        try:
            _clone_kernel_params(ker, prev_model.kernel)
        except Exception:
            pass
    m = _make_exact_gpr(X, Y, ker, noise=noise)
    if prev_model is not None:
        try:
            m.likelihood.variance.assign(float(prev_model.likelihood.variance.numpy()))
        except Exception:
            pass
    dt = _train_exact_gpr(m, steps=steps, lr=lr, clip_norm=EXACT_GRAD_CLIP)
    return m, dt


def _build_exact_pack_from_gpr(model, X_train, Y_train, jitter=1e-6):
    """
    Build a DiagSparsePosteriorPack that is EXACT for a GPR model on support X_train:
      mean = K_xX (K_XX + sigma^2 I)^-1 y
      var  = kxx - K_xX (K_XX + sigma^2 I)^-1 K_Xx
    This matches the pack predictor when we use:
      Z = X_train, L = chol(K + sigma^2 I), m_w = L^-1 y, C_T = 0.
    """
    X_train = np.asarray(X_train, dtype=np.float64)
    Y_train = np.asarray(Y_train, dtype=np.float64).reshape(-1, 1)
    Xt = tf.convert_to_tensor(X_train, dtype=DTF)
    Yt = tf.convert_to_tensor(Y_train, dtype=DTF)
    M = int(X_train.shape[0])
    if M <= 0:
        raise ValueError("Exact pack needs non-empty training support")

    K = model.kernel(Xt)
    K = 0.5 * (K + tf.transpose(K))
    try:
        lik = tf.cast(model.likelihood.variance, DTF)
        lik_f = float(model.likelihood.variance.numpy())
    except Exception:
        lik = tf.constant(0.0, dtype=DTF)
        lik_f = 0.0
    K = K + (lik + tf.constant(float(jitter), DTF)) * tf.eye(M, dtype=DTF)
    L = tf.linalg.cholesky(K)
    m_w = tf.linalg.triangular_solve(L, Yt, lower=True)
    C_T = tf.zeros((M, M), dtype=DTF)
    return DiagSparsePosteriorPack(
        Z=Xt,
        L=L,
        m_w=m_w,
        C_T=C_T,
        lik_var=tf.constant(lik_f, dtype=DTF),
    )


def _packs_from_exact(models, Xfit, Ys_fit):
    m1, m2, m3, m4 = models
    y1, y2, y3, y4 = Ys_fit
    p1 = _build_exact_pack_from_gpr(m1, Xfit, y1, jitter=1e-6)
    p2 = _build_exact_pack_from_gpr(m2, Xfit, y2, jitter=1e-6)
    p3 = _build_exact_pack_from_gpr(m3, Xfit, y3, jitter=1e-6)
    p4 = _build_exact_pack_from_gpr(m4, Xfit, y4, jitter=1e-6)
    return (
        p1.Z, p1.L, p1.m_w, p1.C_T, p1.lik_var,
        p2.Z, p2.L, p2.m_w, p2.C_T, p2.lik_var,
        p3.Z, p3.L, p3.m_w, p3.C_T, p3.lik_var,
        p4.Z, p4.L, p4.m_w, p4.C_T, p4.lik_var,
    )


def _replay_to_arrays(Xmem, Ydx_mem, Ydxdot_mem, Ydth_mem, Ydthdot_mem):
    nmem = len(Xmem)
    if nmem <= 0:
        raise ValueError("empty replay")
    Xb = np.asarray(Xmem, dtype=np.float64)
    ydx_b = np.asarray(Ydx_mem, dtype=np.float64).reshape(-1, 1)
    ydxdot_b = np.asarray(Ydxdot_mem, dtype=np.float64).reshape(-1, 1)
    ydth_b = np.asarray(Ydth_mem, dtype=np.float64).reshape(-1, 1)
    ydthdot_b = np.asarray(Ydthdot_mem, dtype=np.float64).reshape(-1, 1)
    return Xb, ydx_b, ydxdot_b, ydth_b, ydthdot_b


# Build planner (reuses the same simple pack-based MPPI used by global baselines)
plan_tf = make_plan_tf_simple()

METHOD = "EXACTGP_GLOBAL"
print(f"\n===========================\nRunning {METHOD} (hold {SUCCESS_HOLD_STEPS} steps)\n===========================\n")
print(f"[{METHOD}] full-replay exact retraining (no cap) | init={int(X0.shape[0])} update_uses=all_memory | update_every={UPDATE_EVERY}")

# Initial exact models (full offline random dataset)
rng0 = np.random.default_rng(SEED_BASE)
X_fit_cur = np.asarray(X0, dtype=np.float64)
ydx_fit_cur = np.asarray(Ydx0, dtype=np.float64)
ydxdot_fit_cur = np.asarray(Ydxdot0, dtype=np.float64)
ydth_fit_cur = np.asarray(Ydth0, dtype=np.float64)
ydthdot_fit_cur = np.asarray(Ydthdot0, dtype=np.float64)

print(f"[{METHOD}] Training initial Exact GPR models on {X_fit_cur.shape[0]} samples...")
m_dx, _dt1 = _fit_exact_head(X_fit_cur, ydx_fit_cur, kernel_seed=None, prev_model=None, noise=NOISE_INIT, steps=EXACT_ITERS_INIT, lr=EXACT_LR)
m_dxdot, _dt2 = _fit_exact_head(X_fit_cur, ydxdot_fit_cur, kernel_seed=None, prev_model=None, noise=NOISE_INIT, steps=EXACT_ITERS_INIT, lr=EXACT_LR)
m_dth, _dt3 = _fit_exact_head(X_fit_cur, ydth_fit_cur, kernel_seed=None, prev_model=None, noise=NOISE_INIT, steps=EXACT_ITERS_INIT, lr=EXACT_LR)
m_dthdot, _dt4 = _fit_exact_head(X_fit_cur, ydthdot_fit_cur, kernel_seed=None, prev_model=None, noise=NOISE_INIT, steps=EXACT_ITERS_INIT, lr=EXACT_LR)
models = (m_dx, m_dxdot, m_dth, m_dthdot)
pack_tensors = _packs_from_exact(models, X_fit_cur, (ydx_fit_cur, ydxdot_fit_cur, ydth_fit_cur, ydthdot_fit_cur))

for run in range(N_RUNS):
    seed = SEED_BASE + run
    rng = np.random.default_rng(seed)
    env = make_env(render_mode=None, seed=seed, max_episode_steps=MAX_STEPS_PER_EP, start_down=True, edge_respawn=True)
    do_live = bool(LIVE_RENDER and ((not LIVE_ONLY_FIRST_RUN) or (run == 0)))

    run_frames = []
    if CAPTURE_REPLAY:
        run_frames.extend([_title_frame(f"{METHOD} | RUN {run} | start", W=REPLAY_SIZE[0], H=REPLAY_SIZE[1])] * TITLE_HOLD_FRAMES)

    # replay buffer starts from the same offline dataset for fairness
    Xmem = [np.asarray(X0[i], dtype=np.float64) for i in range(X0.shape[0])]
    Ydx_mem     = [float(Ydx0[i, 0]) for i in range(Ydx0.shape[0])]
    Ydxdot_mem  = [float(Ydxdot0[i, 0]) for i in range(Ydxdot0.shape[0])]
    Ydth_mem    = [float(Ydth0[i, 0]) for i in range(Ydth0.shape[0])]
    Ydthdot_mem = [float(Ydthdot0[i, 0]) for i in range(Ydthdot0.shape[0])]

    # fresh exact models each run from the same full offline dataset (fair reset)
    X_fit_cur = np.asarray(X0, dtype=np.float64)
    ydx_fit_cur = np.asarray(Ydx0, dtype=np.float64)
    ydxdot_fit_cur = np.asarray(Ydxdot0, dtype=np.float64)
    ydth_fit_cur = np.asarray(Ydth0, dtype=np.float64)
    ydthdot_fit_cur = np.asarray(Ydthdot0, dtype=np.float64)

    m_dx, _ = _fit_exact_head(X_fit_cur, ydx_fit_cur, prev_model=None, noise=NOISE_INIT, steps=EXACT_ITERS_INIT, lr=EXACT_LR)
    m_dxdot, _ = _fit_exact_head(X_fit_cur, ydxdot_fit_cur, prev_model=None, noise=NOISE_INIT, steps=EXACT_ITERS_INIT, lr=EXACT_LR)
    m_dth, _ = _fit_exact_head(X_fit_cur, ydth_fit_cur, prev_model=None, noise=NOISE_INIT, steps=EXACT_ITERS_INIT, lr=EXACT_LR)
    m_dthdot, _ = _fit_exact_head(X_fit_cur, ydthdot_fit_cur, prev_model=None, noise=NOISE_INIT, steps=EXACT_ITERS_INIT, lr=EXACT_LR)
    models = (m_dx, m_dxdot, m_dth, m_dthdot)
    pack_tensors = _packs_from_exact(models, X_fit_cur, (ydx_fit_cur, ydxdot_fit_cur, ydth_fit_cur, ydthdot_fit_cur))

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
            try:
                LIVE_VIEWER.reset()
            except Exception:
                pass
        if CAPTURE_REPLAY:
            run_frames.extend([_title_frame(f"{METHOD} | RUN {run} | EP {ep}", W=REPLAY_SIZE[0], H=REPLAY_SIZE[1])] * TITLE_HOLD_FRAMES)

        ep_reward = 0.0; ep_steps = 0
        ep_u2_sum = 0.0; ep_u_abs_sum = 0.0
        ep_unc_sum = 0.0; ep_unc_max_v = 0.0

        hold_count = 0; hold_max = 0
        ep_hold_complete_t = None

        ep_prog = TextProgress(MAX_STEPS_PER_EP, prefix=f"{METHOD} run{run} ep{ep} step ", every=PROGRESS_EVERY_STEPS)

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
                try:
                    LIVE_VIEWER.maybe_update(
                        step_i=t_global, x=x1, theta=th1,
                        text=f"{METHOD} | r{run} e{ep} t={t_global} | r={float(r):+.3f} u={u0:+.3f} hold={hold_count}/{SUCCESS_HOLD_STEPS}"
                    )
                except Exception:
                    pass

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

            # append executed transition to replay
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

            trn_dt = 0.0; reb_dt = 0.0
            did_update = False
            if ((t_global + 1) % int(UPDATE_EVERY) == 0) and (len(Xmem) > EXACT_MIN_MEM_FOR_UPDATE):
                did_update = True
                Xb, ydx_b, ydxdot_b, ydth_b, ydthdot_b = _replay_to_arrays(
                    Xmem, Ydx_mem, Ydxdot_mem, Ydth_mem, Ydthdot_mem
                )

                tu0 = time.perf_counter()
                m_dx, dt1 = _fit_exact_head(Xb, ydx_b, prev_model=m_dx, noise=NOISE_INIT, steps=EXACT_ITERS_UPDATE, lr=EXACT_LR)
                m_dxdot, dt2 = _fit_exact_head(Xb, ydxdot_b, prev_model=m_dxdot, noise=NOISE_INIT, steps=EXACT_ITERS_UPDATE, lr=EXACT_LR)
                m_dth, dt3 = _fit_exact_head(Xb, ydth_b, prev_model=m_dth, noise=NOISE_INIT, steps=EXACT_ITERS_UPDATE, lr=EXACT_LR)
                m_dthdot, dt4 = _fit_exact_head(Xb, ydthdot_b, prev_model=m_dthdot, noise=NOISE_INIT, steps=EXACT_ITERS_UPDATE, lr=EXACT_LR)
                trn_dt = float(time.perf_counter() - tu0)

                X_fit_cur = Xb
                ydx_fit_cur, ydxdot_fit_cur, ydth_fit_cur, ydthdot_fit_cur = ydx_b, ydxdot_b, ydth_b, ydthdot_b
                models = (m_dx, m_dxdot, m_dth, m_dthdot)

                try:
                    met = eval_multihead_batch(
                        [m_dx, m_dxdot, m_dth, m_dthdot],
                        Xb, [ydx_b, ydxdot_b, ydth_b, ydthdot_b],
                        add_likelihood_var=True,
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
                pack_tensors = _packs_from_exact(models, X_fit_cur, (ydx_fit_cur, ydxdot_fit_cur, ydth_fit_cur, ydthdot_fit_cur))
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
