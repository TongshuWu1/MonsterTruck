# ===========================
# Cell 4 — MPPI (TF/GPU) planner + GP dynamics rollout + corridor utilities
#
# What this cell provides (definitions only; run loops are in Cells 6/7/8):
#   - tf_feature_map(): state/action -> 6D features (float64)
#   - DiagSparsePosteriorPack: fast local/global predictor pack (mean + diag var)
#   - build_pack_from_model(): build pack from a GP model at chosen inducing Z_sub
#   - predict_pack_mu_var_tf(): TF prediction using pack (fast; no full_cov)
#   - mppi_build_planner_tf(): compiles a tf.function MPPI planner (returns u0,u_mean_new,tubeX,unc_w,r2,inv_var,corr_dbg)
#   - corridor_min_dist_tf(): min whitened distance from Xpool to nominal tube features
#   - kcenter_farthest_first(): numpy k-center selection utility (used later in PALSGP Cell 6)
#   - is_success_state(): success predicate (used in run loops)
#
# Notes:
#   - All TF/GP compute uses float64.
#   - Dynamics are modeled as DELTAS: s_{t+1} = s_t + Δs
#   - Theta is wrapped each step using atan2(sin,cos).
# ===========================

import numpy as np
import tensorflow as tf
import gpflow
import time

DTF = tf.float64

# ------------------------------------------------------------
# TF feature map: state (x,xdot,th,thdot) + action u -> (6,)
# ------------------------------------------------------------
@tf.function
def tf_feature_map(s, u, x_scale=2.4, v_scale=3.0, w_scale=8.0):
    """
    s: (...,4) float64
    u: (...,)  float64
    returns: (...,6) float64
    """
    x     = s[..., 0]
    xdot  = s[..., 1]
    th    = s[..., 2]
    thdot = s[..., 3]

    f0 = tf.tanh(x / x_scale)
    f1 = tf.tanh(xdot / v_scale)
    f2 = tf.sin(th)
    f3 = tf.cos(th)
    f4 = tf.tanh(thdot / w_scale)
    f5 = u

    return tf.stack([f0, f1, f2, f3, f4, f5], axis=-1)

@tf.function
def tf_wrap_angle(th):
    return tf.atan2(tf.sin(th), tf.cos(th))

# ------------------------------------------------------------
# Success predicate (for CartPole swing-up)
# ------------------------------------------------------------
def is_success_state(x, xdot, th, thdot,
                     cos_th_min=0.98, x_abs_max=0.35, xdot_abs_max=1.0, thdot_abs_max=2.5):
    """
    A simple swing-up success check. Tune thresholds if needed.
    If a task hook is provided by the env module, use it for consistency.
    """
    task_fn = globals().get("task_is_success_state", None)
    if callable(task_fn):
        return bool(task_fn(x, xdot, th, thdot))
    return (np.cos(th) >= cos_th_min) and (abs(x) <= x_abs_max) and (abs(xdot) <= xdot_abs_max) and (abs(thdot) <= thdot_abs_max)

# ------------------------------------------------------------
# Diagonal posterior inducing pack (fast predictor)
# ------------------------------------------------------------
class DiagSparsePosteriorPack:
    """
    Stores everything needed for fast predictive mean/diag-var:
      Kzz = chol(L)
      m_w = L^{-1} mu_u
      C_T = (L^{-1} diag(sqrt(Su_diag)))^T   (so we can compute var_add cheaply)

    Prediction for f(x):
      A = L^{-1} Kzx
      mean = A^T m_w
      var  = kxx - sum(A^2) + sum((C_T @ A)^2)   (diag posterior approx)
    """
    __slots__ = ("Z", "L", "m_w", "C_T", "lik_var")

    def __init__(self, Z, L, m_w, C_T, lik_var):
        self.Z = Z
        self.L = L
        self.m_w = m_w
        self.C_T = C_T
        self.lik_var = lik_var

def build_pack_from_model(model, Z_sub, jitter=1e-6, use_cached=True):
    """
    Build DiagSparsePosteriorPack from a GP model at inducing points Z_sub.
    We approximate Su as diag(var(f(Z_sub))).
    """
    Z_sub = np.asarray(Z_sub, dtype=np.float64)
    Zt = tf.convert_to_tensor(Z_sub, dtype=DTF)

    # posterior at Z_sub (diag only)
    if use_cached and hasattr(model, "predict_f_cached"):
        mu_tf, var_tf = model.predict_f_cached(Zt, full_cov=False)
    else:
        mu_tf, var_tf = model.predict_f(Zt, full_cov=False)

    mu = tf.reshape(mu_tf, (-1, 1))                      # (M,1)
    Su_diag = tf.reshape(var_tf, (-1,))                  # (M,)
    Su_diag = tf.maximum(Su_diag, tf.constant(1e-12, DTF))

    # Kzz
    Kzz = model.kernel(Zt)                               # (M,M)
    Kzz = 0.5 * (Kzz + tf.transpose(Kzz))
    Kzz = Kzz + tf.constant(float(jitter), DTF) * tf.eye(tf.shape(Zt)[0], dtype=DTF)
    L = tf.linalg.cholesky(Kzz)

    # m_w = L^{-1} mu
    m_w = tf.linalg.triangular_solve(L, mu, lower=True)

    # C = L^{-1} diag(sqrt(Su_diag))
    # compute Linv once (M,M) and scale columns
    I = tf.eye(tf.shape(Zt)[0], dtype=DTF)
    Linv = tf.linalg.triangular_solve(L, I, lower=True)  # L^{-1}
    sqrt_s = tf.sqrt(Su_diag)
    C = Linv * tf.reshape(sqrt_s, (1, -1))               # scale columns
    C_T = tf.transpose(C)

    try:
        lik_var = float(model.likelihood.variance.numpy())
    except Exception:
        lik_var = 0.0

    return DiagSparsePosteriorPack(
        Z=Zt, L=L, m_w=m_w, C_T=C_T, lik_var=tf.constant(lik_var, dtype=DTF)
    )

# ------------------------------------------------------------
# TF prediction using a pack (mean + diag-var)
# ------------------------------------------------------------
@tf.function
def predict_pack_mu_var_tf(pack_Z, pack_L, pack_mw, pack_CT, lik_var, kernel, Xfeat):
    """
    Xfeat: (B,6)
    returns:
      mu:  (B,)
      var: (B,)  (latent f variance; does NOT include likelihood unless you add it)
    """
    # Kzx: (M,B)
    Kzx = kernel(pack_Z, Xfeat)          # (M,B)
    A = tf.linalg.triangular_solve(pack_L, Kzx, lower=True)   # (M,B)

    mu = tf.matmul(A, pack_mw, transpose_a=True)             # (B,1)
    mu = tf.reshape(mu, (-1,))

    kxx = kernel(Xfeat, full_cov=False)                      # (B,)
    q = tf.reduce_sum(tf.square(A), axis=0)                  # (B,)

    # var_add = sum((C_T @ A)^2)
    E = tf.matmul(pack_CT, A)                                # (M,B)
    var_add = tf.reduce_sum(tf.square(E), axis=0)            # (B,)

    var = kxx - q + var_add
    var = tf.maximum(var, tf.constant(1e-12, DTF))
    return mu, var

# ------------------------------------------------------------
# Corridor distance: min whitened squared distance from pool to tube
# ------------------------------------------------------------
@tf.function
def corridor_min_dist_tf(Xpool, tubeX, inv_var, softmin_temp=20.0):
    """
    Xpool : (N,6)
    tubeX : (H,6)
    inv_var: (6,) whitening weights (e.g., 1/(var+eps))
    Returns:
      r2_min: (N,) min_h sum_d inv_var[d]*(Xpool - tubeX[h])^2
    """
    # (N,H,6)
    diff = tf.expand_dims(Xpool, 1) - tf.expand_dims(tubeX, 0)
    wdiff2 = tf.square(diff) * tf.reshape(inv_var, (1, 1, -1))
    d2 = tf.reduce_sum(wdiff2, axis=-1)  # (N,H)

    # hard min
    r2_min = tf.reduce_min(d2, axis=1)

    return r2_min

# ------------------------------------------------------------
# MPPI cost function (tunable weights)
# ------------------------------------------------------------
@tf.function
def stage_cost_tf(s, u,
                  w_theta=8.0, w_x=0.8, w_xdot=0.08, w_thdot=0.08, w_u=0.02):
    """
    s: (K,4) or (...,4)
    u: (K,)  or (...,)
    cost ~ (1-cos(theta)) + penalties

    If the env module exports task_stage_cost_tf, use it so MPPI planning cost
    stays consistent with the environment/task definition.
    """
    task_cost_fn = globals().get("task_stage_cost_tf", None)
    if callable(task_cost_fn):
        return tf.cast(task_cost_fn(s, u), DTF)

    x     = s[..., 0]
    xdot  = s[..., 1]
    th    = s[..., 2]
    thdot = s[..., 3]

    c = (w_theta * (1.0 - tf.cos(th))
         + w_x * tf.square(x)
         + w_xdot * tf.square(xdot)
         + w_thdot * tf.square(thdot)
         + w_u * tf.square(u))
    return c

# ------------------------------------------------------------
# One GP dynamics step for a batch (K rollouts) using 4 packs
# ------------------------------------------------------------
@tf.function
def gp_step_batch_tf(s, u,
                     pack_dx, pack_dxdot, pack_dth, pack_dthdot,
                     k_dx, k_dxdot, k_dth, k_dthdot,
                     unc_lambda=0.0):
    """
    s: (K,4), u: (K,)
    Returns:
      s_next: (K,4)
      unc_bonus: (K,)  (optional uncertainty penalty term)
      unc_scalar: (K,) scalar uncertainty proxy (for logging)
    """
    Xfeat = tf_feature_map(s, u)  # (K,6)

    mu_dx,  var_dx  = predict_pack_mu_var_tf(pack_dx.Z,     pack_dx.L,     pack_dx.m_w,     pack_dx.C_T,     pack_dx.lik_var,     k_dx,     Xfeat)
    mu_dxd, var_dxd = predict_pack_mu_var_tf(pack_dxdot.Z,  pack_dxdot.L,  pack_dxdot.m_w,  pack_dxdot.C_T,  pack_dxdot.lik_var,  k_dxdot,  Xfeat)
    mu_dt,  var_dt  = predict_pack_mu_var_tf(pack_dth.Z,    pack_dth.L,    pack_dth.m_w,    pack_dth.C_T,    pack_dth.lik_var,    k_dth,    Xfeat)
    mu_dtd, var_dtd = predict_pack_mu_var_tf(pack_dthdot.Z, pack_dthdot.L, pack_dthdot.m_w, pack_dthdot.C_T, pack_dthdot.lik_var, k_dthdot, Xfeat)

    # state update with DELTAS
    x     = s[:, 0] + mu_dx
    xdot  = s[:, 1] + mu_dxd
    th    = tf_wrap_angle(s[:, 2] + mu_dt)
    thdot = s[:, 3] + mu_dtd

    s_next = tf.stack([x, xdot, th, thdot], axis=-1)

    # scalar uncertainty proxy (use dxdot head; include likelihood variance)
    unc = tf.sqrt(tf.maximum(var_dxd + pack_dxdot.lik_var, tf.constant(1e-12, DTF)))

    unc_bonus = unc_lambda * unc
    return s_next, unc_bonus, unc

# ------------------------------------------------------------
# Build a compiled MPPI planner (tf.function)
# ------------------------------------------------------------
def mppi_build_planner_tf(
    k_dx, k_dxdot, k_dth, k_dthdot,
    H=25, K=4096,
    sigma=0.35,
    lam=1.0,
    u_min=-1.0, u_max=1.0,
    unc_lambda=0.0,
    tube_invvar_eps=1e-4,
):
    """
    Returns a tf.function:
      plan_tf(s0, u_mean, pack_dx, pack_dxdot, pack_dth, pack_dthdot, Xpool, corr_r0, corr_r1)
        -> u0, u_mean_new, tubeX, unc_w, r2_pool, inv_var, corr_dbg

    Inputs:
      s0      : (4,) float64
      u_mean  : (H,) float64
      packs   : DiagSparsePosteriorPack (python objects; captured by closure)
      Xpool   : (N,6) float64 or empty (0,6)
      corr_r0 : float (threshold for "core" corridor)
      corr_r1 : float (threshold for "expanded" corridor)

    Notes:
      - Corridor outputs are optional (if Xpool is empty, returns empty r2_pool).
      - tubeX is computed from nominal rollout (using u_mean, mean dynamics).
    """

    sigma = float(sigma)
    lam = float(lam)
    u_min = float(u_min)
    u_max = float(u_max)

    @tf.function
    def plan_tf(s0, u_mean,
                pack_dx, pack_dxdot, pack_dth, pack_dthdot,
                Xpool, corr_r0, corr_r1):
        s0 = tf.cast(tf.reshape(s0, (4,)), DTF)
        u_mean = tf.cast(tf.reshape(u_mean, (H,)), DTF)

        # Sample controls: U = clip(u_mean + eps)
        eps = tf.random.normal(shape=(K, H), mean=0.0, stddev=sigma, dtype=DTF)
        U = tf.clip_by_value(tf.expand_dims(u_mean, 0) + eps, u_min, u_max)  # (K,H)

        # Rollout trajectories & costs
        s = tf.repeat(tf.expand_dims(s0, 0), repeats=K, axis=0)  # (K,4)
        cost = tf.zeros((K,), dtype=DTF)

        # for logging: average uncertainty over horizon
        unc_sum = tf.zeros((K,), dtype=DTF)

        for t in tf.range(H):
            u_t = U[:, t]
            s, unc_bonus, unc = gp_step_batch_tf(
                s, u_t,
                pack_dx, pack_dxdot, pack_dth, pack_dthdot,
                k_dx, k_dxdot, k_dth, k_dthdot,
                unc_lambda=tf.constant(unc_lambda, DTF),
            )
            c_t = stage_cost_tf(s, u_t)
            cost = cost + c_t + unc_bonus
            unc_sum = unc_sum + unc

        # MPPI weights
        beta = tf.reduce_min(cost)
        w = tf.exp(-(cost - beta) / tf.constant(lam, DTF))
        w_sum = tf.reduce_sum(w) + tf.constant(1e-12, DTF)

        # update mean controls
        U_weighted = tf.reduce_sum(tf.expand_dims(w, 1) * U, axis=0) / w_sum  # (H,)
        u_mean_new = U_weighted

        # first control to execute
        u0 = u_mean_new[0]

        # uncertainty summary (for logging)
        unc_w = tf.reduce_sum(w * (unc_sum / tf.cast(H, DTF))) / w_sum

        # ---- nominal tube rollout for corridor ----
        sT = tf.reshape(s0, (1, 4))
        tubeX_list = []
        for t in tf.range(H):
            u_t = tf.reshape(u_mean_new[t], (1,))
            Xft = tf_feature_map(sT, u_t)          # (1,6)
            tubeX_list.append(tf.reshape(Xft, (6,)))
            sT, _, _ = gp_step_batch_tf(
                sT, u_t,
                pack_dx, pack_dxdot, pack_dth, pack_dthdot,
                k_dx, k_dxdot, k_dth, k_dthdot,
                unc_lambda=tf.constant(0.0, DTF),   # nominal tube shouldn't include penalty
            )
        tubeX = tf.stack(tubeX_list, axis=0)       # (H,6)

        # whitening weights based on tube feature variance
        tube_var = tf.math.reduce_variance(tubeX, axis=0)
        inv_var = 1.0 / (tube_var + tf.constant(tube_invvar_eps, DTF))      # (6,)

        # corridor distances for a candidate pool (if provided)
        Npool = tf.shape(Xpool)[0]
        r2_pool = tf.cond(
            Npool > 0,
            lambda: corridor_min_dist_tf(Xpool, tubeX, inv_var),
            lambda: tf.zeros((0,), dtype=DTF),
        )

        # corridor debug / counts
        inside0 = tf.reduce_sum(tf.cast(r2_pool <= corr_r0, tf.int32))
        inside1 = tf.reduce_sum(tf.cast(r2_pool <= corr_r1, tf.int32))

        corr_dbg = tf.stack([
            tf.cast(Npool, tf.float64),
            tf.cast(inside0, tf.float64),
            tf.cast(inside1, tf.float64),
        ], axis=0)  # [Npool, inside0, inside1]

        return u0, u_mean_new, tubeX, unc_w, r2_pool, inv_var, corr_dbg

    return plan_tf

# ------------------------------------------------------------
# Numpy k-center / farthest-first (used later in PALSGP)
# ------------------------------------------------------------
def kcenter_farthest_first(X, m, start_idx=0):
    """
    Classic farthest-first traversal (k-center) on X (N,D).
    Returns selected indices length <= m.
    """
    X = np.asarray(X, dtype=np.float64)
    N = X.shape[0]
    if N == 0 or m <= 0:
        return np.array([], dtype=np.int64)
    m = min(int(m), N)

    sel = np.empty((m,), dtype=np.int64)
    sel[0] = int(np.clip(start_idx, 0, N - 1))

    # initialize min distances to first center
    d2 = np.sum((X - X[sel[0]])**2, axis=1)

    for k in range(1, m):
        i = int(np.argmax(d2))
        sel[k] = i
        d2_new = np.sum((X - X[i])**2, axis=1)
        d2 = np.minimum(d2, d2_new)

    return sel

print("✅ Cell 4 ready: MPPI + packs + corridor utilities")
