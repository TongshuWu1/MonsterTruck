# ===========================
# Cell 3 — OSGPR-VFE core (Streaming Sparse GP) + training + summaries + anchors
#
# Matches OUR pipeline:
#   - GLOBAL sparse GP lives on an inducing set Z (usually Z_GLOBAL, size M, capped)
#   - Each online update:
#       (a) extract old posterior summary at Z_old
#       (b) build a NEW OSGPR_VFE model on (X_new, Y_new) + old summary
#       (c) train a bit, then cache for fast prediction
#   - Anchors are selected FROM THE CURRENT inducing set Z (so you can reselect after updates)
#
# Provides:
#   - batch_state_to_features(): (B,4)+(B,) -> (B,6)
#   - OSGPR_VFE (single-output)
#   - train_osgpr()
#   - prior_summary(), extract_summary_from_model()
#   - greedy_dopt_anchors_from_K()
#   - rebuild_osgpr_from_old_summary(): returns (model_new, train_time, neg_obj)  ✅ fixes your unpack bug
# ===========================

import time
import copy
import numpy as np
import tensorflow as tf
import gpflow

from gpflow.inducing_variables import InducingPoints
from gpflow.models import GPModel, InternalDataTrainingLossMixin
from gpflow import covariances

# ---- numerics ----
gpflow.config.set_default_float(np.float64)
gpflow.config.set_default_jitter(1e-6)
tf.keras.backend.set_floatx("float64")

print("TF built with CUDA:", tf.test.is_built_with_cuda())
try:
    print("GPUs visible:", tf.config.list_physical_devices("GPU"))
except Exception as e:
    print("GPU query failed:", e)

DTYPE = gpflow.default_float()

# ---------------------------
# helpers
# ---------------------------
def sym_jitter(A, jitter=1e-6):
    """Make symmetric + add jitter (numpy)."""
    A = np.asarray(A, dtype=np.float64)
    A = 0.5 * (A + A.T)
    A = A + float(jitter) * np.eye(A.shape[0], dtype=np.float64)
    return A

def finite_mask(*arrs):
    """Row-wise finite mask across arrays."""
    m = None
    for a in arrs:
        a = np.asarray(a)
        mm = np.isfinite(a).all(axis=1) if a.ndim == 2 else np.isfinite(a)
        m = mm if m is None else (m & mm)
    return m

def clone_kernel(kernel):
    """
    Clone a GPflow kernel (to avoid variable-sharing across models).
    gpflow.utilities.deepcopy exists in many versions; fallback to copy.deepcopy.
    """
    try:
        from gpflow.utilities import deepcopy as gf_deepcopy
        return gf_deepcopy(kernel)
    except Exception:
        return copy.deepcopy(kernel)

# ------------------------------------------------------------
# Batch feature map (FAST) — used by MPPI later
# ------------------------------------------------------------
def batch_state_to_features(S, U, x_scale=2.4, v_scale=3.0, w_scale=8.0):
    """
    Vectorized mapping from physical CartPole state to 6D GP features.

    S: (B,4)  [x, xdot, theta, thetadot]
    U: (B,)   action in [-1,1]
    Returns:
      Xfeat: (B,6) [tanh(x/xs), tanh(xdot/vs), sin(theta), cos(theta), tanh(thetadot/ws), u]
    """
    S = np.asarray(S, dtype=np.float64)
    U = np.asarray(U, dtype=np.float64).reshape(-1)
    assert S.ndim == 2 and S.shape[1] == 4, "S must be (B,4)"
    assert U.shape[0] == S.shape[0], "U must match batch size"

    x     = S[:, 0]
    xdot  = S[:, 1]
    th    = S[:, 2]
    thdot = S[:, 3]

    Xf = np.empty((S.shape[0], 6), dtype=np.float64)
    Xf[:, 0] = np.tanh(x / x_scale)
    Xf[:, 1] = np.tanh(xdot / v_scale)
    Xf[:, 2] = np.sin(th)
    Xf[:, 3] = np.cos(th)
    Xf[:, 4] = np.tanh(thdot / w_scale)
    Xf[:, 5] = U
    return Xf

# ============================================================
# OSGPR-VFE model — regression-only, single-output
# ============================================================
class OSGPR_VFE(GPModel, InternalDataTrainingLossMixin):
    """
    Online Sparse Variational GP Regression (VFE), SINGLE-OUTPUT.

    Provide:
      - current batch data (X, Y)
      - old summary q_old(u)=N(mu_old, Su_old) at Z_old
      - Kaa_old = K(Z_old,Z_old) from old step
      - new inducing Z (usually Z_GLOBAL; you MAY refresh Z over time, but size should be capped)

    Includes:
      - predict_f (correct but slower)
      - build_predict_cache + predict_f_cached (FAST diag predictions)
    """
    def __init__(self, data, kernel, mu_old, Su_old, Kaa_old, Z_old, Z, mean_function=None):
        X, Y = gpflow.models.util.data_input_to_tensor(data)
        self.X, self.Y = X, Y

        likelihood = gpflow.likelihoods.Gaussian()
        num_latent_gps = GPModel.calc_num_latent_gps_from_data(data, kernel, likelihood)
        super().__init__(kernel, likelihood, mean_function, num_latent_gps)

        Z = np.asarray(Z, dtype=np.float64)
        assert Z.ndim == 2, "Z must be (M, D)"
        self.inducing_variable = InducingPoints(Z)
        gpflow.set_trainable(self.inducing_variable, False)
        mu_old  = np.asarray(mu_old, dtype=np.float64).reshape(-1, 1)
        Su_old  = sym_jitter(Su_old, 1e-6)
        Kaa_old = sym_jitter(Kaa_old, 1e-6)
        Z_old   = np.asarray(Z_old, dtype=np.float64)

        self.mu_old  = tf.Variable(mu_old,  trainable=False, dtype=DTYPE)
        self.Su_old  = tf.Variable(Su_old,  trainable=False, dtype=DTYPE)
        self.Kaa_old = tf.Variable(Kaa_old, trainable=False, dtype=DTYPE)
        self.Z_old   = tf.Variable(Z_old,   trainable=False, dtype=DTYPE)

        if self.mean_function is None:
            self.mean_function = gpflow.mean_functions.Zero()

        # cache for fast predict
        self._cache_ready = False
        self._cache_Lb = None
        self._cache_LD = None
        self._cache_rhs = None

    def _common_terms(self):
        """
        Build common matrices used by both ELBO and prediction.

        Z   : new inducing (Mb)
        Za  : old inducing (Ma) == self.Z_old
        X   : current batch inputs

        Kbf = K(Z, X)    [Mb, N]
        Kbb = K(Z, Z)    [Mb, Mb]
        Kba = K(Z, Za)   [Mb, Ma]
        """
        jitter = gpflow.utilities.to_default_float(1e-6)
        sigma2 = self.likelihood.variance

        Saa = self.Su_old  # [Ma,Ma]
        ma  = self.mu_old  # [Ma,1]

        Kbf = covariances.Kuf(self.inducing_variable, self.kernel, self.X)           # [Mb, N]
        Kbb = covariances.Kuu(self.inducing_variable, self.kernel, jitter=jitter)   # [Mb, Mb]
        Kba = covariances.Kuf(self.inducing_variable, self.kernel, self.Z_old)      # [Mb, Ma]

        Kaa_cur = gpflow.utilities.add_noise_cov(self.kernel(self.Z_old), jitter)   # [Ma,Ma]
        Kaa     = gpflow.utilities.add_noise_cov(self.Kaa_old, jitter)              # [Ma,Ma]

        err = self.Y - self.mean_function(self.X)  # [N,1]

        # c = Kbf*(Y/sigma2) + Kba*(Saa^{-1} ma)
        Sainv_ma = tf.linalg.solve(Saa, ma)                                # [Ma,1]
        c = tf.matmul(Kbf, self.Y / sigma2) + tf.matmul(Kba, Sainv_ma)     # [Mb,1]

        # Cholesky(Kbb)
        Lb = tf.linalg.cholesky(Kbb)                                       # [Mb,Mb]
        Lbinv_c   = tf.linalg.triangular_solve(Lb, c,   lower=True)        # [Mb,1]
        Lbinv_Kba = tf.linalg.triangular_solve(Lb, Kba, lower=True)        # [Mb,Ma]
        Lbinv_Kbf = tf.linalg.triangular_solve(Lb, Kbf, lower=True) / tf.sqrt(sigma2)  # [Mb,N]

        d1 = tf.matmul(Lbinv_Kbf, Lbinv_Kbf, transpose_b=True)             # [Mb,Mb]

        # T = (Lb^{-1}Kba)^T  => [Ma,Mb]
        T = tf.linalg.matrix_transpose(Lbinv_Kba)

        # d2
        LSa = tf.linalg.cholesky(Saa)
        LSainv_T = tf.linalg.triangular_solve(LSa, T, lower=True)
        d2 = tf.matmul(LSainv_T, LSainv_T, transpose_a=True)               # [Mb,Mb]

        # d3
        La = tf.linalg.cholesky(Kaa)
        Lainv_T = tf.linalg.triangular_solve(La, T, lower=True)
        d3 = tf.matmul(Lainv_T, Lainv_T, transpose_a=True)                 # [Mb,Mb]

        Mb = self.inducing_variable.num_inducing
        D = tf.eye(Mb, dtype=DTYPE) + d1 + d2 - d3
        D = gpflow.utilities.add_noise_cov(D, jitter)
        LD = tf.linalg.cholesky(D)

        rhs = tf.linalg.triangular_solve(LD, Lbinv_c, lower=True)          # [Mb,1]

        Qff_diag = tf.reduce_sum(tf.square(Lbinv_Kbf), axis=0)             # [N]

        tf.debugging.assert_all_finite(Lb,  "Lb has NaN/Inf")
        tf.debugging.assert_all_finite(LD,  "LD has NaN/Inf")
        tf.debugging.assert_all_finite(rhs, "rhs has NaN/Inf")

        return (Kbf, Kba, Kaa, Kaa_cur, La, Kbb, Lb, D, LD, Lbinv_Kba, rhs, err, Qff_diag)

    def maximum_log_likelihood_objective(self):
        sigma2 = self.likelihood.variance
        N = tf.cast(tf.shape(self.X)[0], DTYPE)

        Saa = self.Su_old
        ma  = self.mu_old
        Kfdiag = self.kernel(self.X, full_cov=False)

        (Kbf, Kba, Kaa, Kaa_cur, La, Kbb, Lb, D, LD,
         Lbinv_Kba, rhs, err, Qff_diag) = self._common_terms()

        LSa = tf.linalg.cholesky(Saa)
        Lainv_ma = tf.linalg.triangular_solve(LSa, ma, lower=True)

        bound = -0.5 * N * np.log(2.0 * np.pi)
        bound += -0.5 * tf.reduce_sum(tf.square(err)) / sigma2
        bound += -0.5 * tf.reduce_sum(tf.square(Lainv_ma))
        bound +=  0.5 * tf.reduce_sum(tf.square(rhs))

        bound += -0.5 * N * tf.math.log(sigma2)
        bound += -tf.reduce_sum(tf.math.log(tf.linalg.diag_part(LD)))

        bound += -0.5 * tf.reduce_sum(Kfdiag) / sigma2
        bound +=  0.5 * tf.reduce_sum(Qff_diag)

        bound += tf.reduce_sum(tf.math.log(tf.linalg.diag_part(La)))
        bound += -tf.reduce_sum(tf.math.log(tf.linalg.diag_part(LSa)))

        # correction term involving Kaa_cur - Qaa
        Kaadiff = Kaa_cur - tf.matmul(Lbinv_Kba, Lbinv_Kba, transpose_a=True)
        Sainv_Kaadiff = tf.linalg.solve(Saa, Kaadiff)
        Kainv_Kaadiff = tf.linalg.solve(Kaa, Kaadiff)

        bound += -0.5 * tf.reduce_sum(
            tf.linalg.diag_part(Sainv_Kaadiff) - tf.linalg.diag_part(Kainv_Kaadiff)
        )
        return bound

    def predict_f(self, Xnew, full_cov=False):
        jitter = gpflow.utilities.to_default_float(1e-6)

        Kbs = covariances.Kuf(self.inducing_variable, self.kernel, Xnew)  # [Mb, Nnew]
        (_, _, _, _, _, _, Lb, _, LD, _, rhs, _, _) = self._common_terms()

        Lbinv_Kbs = tf.linalg.triangular_solve(Lb, Kbs, lower=True)
        LDinv_Lbinv_Kbs = tf.linalg.triangular_solve(LD, Lbinv_Kbs, lower=True)
        mean = tf.matmul(LDinv_Lbinv_Kbs, rhs, transpose_a=True)  # [Nnew,1]

        if full_cov:
            Kss = self.kernel(Xnew) + jitter * tf.eye(tf.shape(Xnew)[0], dtype=DTYPE)
            var = (
                Kss
                - tf.matmul(Lbinv_Kbs, Lbinv_Kbs, transpose_a=True)
                + tf.matmul(LDinv_Lbinv_Kbs, LDinv_Lbinv_Kbs, transpose_a=True)
            )
            return mean + self.mean_function(Xnew), var
        else:
            var = (
                self.kernel(Xnew, full_cov=False)
                - tf.reduce_sum(tf.square(Lbinv_Kbs), axis=0)
                + tf.reduce_sum(tf.square(LDinv_Lbinv_Kbs), axis=0)
            )
            var = tf.maximum(var, tf.cast(1e-12, var.dtype))
            return mean + self.mean_function(Xnew), var

    def build_predict_cache(self):
        """Build cached matrices for fast predict_f_cached(). Call after training / after each update."""
        (_, _, _, _, _, _, Lb, _, LD, _, rhs, _, _) = self._common_terms()
        self._cache_Lb = Lb
        self._cache_LD = LD
        self._cache_rhs = rhs
        self._cache_ready = True

    def predict_f_cached(self, Xnew, full_cov=False):
        """Fast diag prediction using cached Lb, LD, rhs."""
        if not self._cache_ready:
            return self.predict_f(Xnew, full_cov=full_cov)

        jitter = gpflow.utilities.to_default_float(1e-6)
        Lb  = self._cache_Lb
        LD  = self._cache_LD
        rhs = self._cache_rhs

        Kbs = covariances.Kuf(self.inducing_variable, self.kernel, Xnew)  # [Mb,Nnew]
        Lbinv_Kbs = tf.linalg.triangular_solve(Lb, Kbs, lower=True)
        LDinv_Lbinv_Kbs = tf.linalg.triangular_solve(LD, Lbinv_Kbs, lower=True)
        mean = tf.matmul(LDinv_Lbinv_Kbs, rhs, transpose_a=True)

        if full_cov:
            Kss = self.kernel(Xnew) + jitter * tf.eye(tf.shape(Xnew)[0], dtype=DTYPE)
            var = (
                Kss
                - tf.matmul(Lbinv_Kbs, Lbinv_Kbs, transpose_a=True)
                + tf.matmul(LDinv_Lbinv_Kbs, LDinv_Lbinv_Kbs, transpose_a=True)
            )
            return mean + self.mean_function(Xnew), var
        else:
            var = (
                self.kernel(Xnew, full_cov=False)
                - tf.reduce_sum(tf.square(Lbinv_Kbs), axis=0)
                + tf.reduce_sum(tf.square(LDinv_Lbinv_Kbs), axis=0)
            )
            var = tf.maximum(var, tf.cast(1e-12, var.dtype))
            return mean + self.mean_function(Xnew), var

# ----------------------------
# training helper
# ----------------------------
def train_osgpr(model, iters=250, lr=0.02, clip_norm=10.0):
    """Adam optimize the negative ELBO."""
    opt = tf.keras.optimizers.Adam(lr)

    @tf.function
    def step():
        with tf.GradientTape() as tape:
            loss = -model.maximum_log_likelihood_objective()
        grads = tape.gradient(loss, model.trainable_variables)
        if clip_norm is not None:
            grads = [tf.clip_by_norm(g, clip_norm) if g is not None else None for g in grads]
        opt.apply_gradients([(g, v) for g, v in zip(grads, model.trainable_variables) if g is not None])
        return loss

    t0 = time.time()
    last = None
    for _ in range(int(iters)):
        last = step()
    return float(time.time() - t0), float(last.numpy())

# ----------------------------
# summaries (to chain online)
# ----------------------------
def prior_summary(kernel, Z):
    """
    Prior summary at inducing Z for the first model:
      mu0 = 0
      Su0 = Kzz
      Kaa0 = Kzz
    """
    Z = np.asarray(Z, dtype=np.float64)
    Kzz = kernel.K(Z).numpy()
    Kzz = sym_jitter(Kzz, 1e-6)
    mu0 = np.zeros((Z.shape[0], 1), dtype=np.float64)
    return mu0, Kzz, Kzz, Z

def extract_summary_from_model(model):
    """
    Extract q(u)=N(mu,Su) at model's current inducing Z plus Kaa=K(Z,Z).
    """
    Z = model.inducing_variable.Z.numpy().astype(np.float64)

    mu_tf, Sig_tf = model.predict_f(Z, full_cov=True)
    mu = mu_tf.numpy().reshape(-1, 1)

    Su = Sig_tf.numpy()
    if Su.ndim == 3:
        Su = Su[0]
    Su = sym_jitter(Su, 1e-6)

    Kaa = model.kernel.K(Z).numpy()
    Kaa = sym_jitter(Kaa, 1e-6)
    return mu, Su, Kaa, Z

# ============================================================
# Anchors: greedy D-opt (log-det) on Kzz
# ============================================================
def greedy_dopt_anchors_from_K(Kzz, m_anchors=24, lam=1e-6):
    """
    Greedy log-det anchor selection on PSD Kzz using incremental Cholesky updates.
    Returns indices of size m_anchors.
    """
    K = np.asarray(Kzz, dtype=np.float64)
    M = K.shape[0]
    assert K.shape == (M, M)
    K = sym_jitter(K, lam)

    chosen = []
    diag = np.clip(np.diag(K).copy(), 1e-12, None)
    remaining = np.ones(M, dtype=bool)
    L = None

    for k in range(min(int(m_anchors), M)):
        if k == 0:
            i = int(np.argmax(diag))
            chosen.append(i)
            remaining[i] = False
            L = np.array([[np.sqrt(diag[i])]], dtype=np.float64)
            continue

        S = np.array(chosen, dtype=np.int64)
        Ks_all = K[np.ix_(S, np.arange(M))]     # (k,M)

        v = np.linalg.solve(L, Ks_all)          # (k,M)
        vn2 = np.sum(v * v, axis=0)             # (M,)
        s2 = diag - vn2
        s2 = np.where(remaining, s2, -np.inf)

        i = int(np.argmax(s2))
        if not np.isfinite(s2[i]) or s2[i] <= 1e-12:
            cand = np.where(remaining)[0]
            if len(cand) == 0:
                break
            i = int(cand[np.argmax(diag[cand])])
            s2_i = max(diag[i], 1e-12)
        else:
            s2_i = float(s2[i])

        chosen.append(i)
        remaining[i] = False

        kvec = K[np.ix_(S, [i])].reshape(-1, 1)  # (k,1)
        w = np.linalg.solve(L, kvec)             # (k,1)
        alpha = np.sqrt(max(s2_i, 1e-12))

        L_new = np.zeros((k + 1, k + 1), dtype=np.float64)
        L_new[:k, :k] = L
        L_new[k, :k] = w.reshape(-1)
        L_new[k, k] = alpha
        L = L_new

    return np.array(chosen, dtype=np.int64)

# ============================================================
# Online update builder (GLOBAL update step)
# ============================================================
def rebuild_osgpr_from_old_summary(
    model_old,
    X_new,
    Y_new,
    Z_new=None,
    iters=120,
    lr=0.02,
    noise=1e-4,
    freeze_kernel=False,
    clip_norm=10.0,
):
    """
    Build a NEW OSGPR_VFE model using:
      - old posterior summary extracted from model_old at its inducing Z_old
      - new executed batch (X_new, Y_new)
      - inducing set Z_new (defaults to model_old.Z; you may pass a refreshed Z here)

    Returns:
      model_new, train_time_sec, last_neg_obj
    """
    # old summary
    mu_old, Su_old, Kaa_old, Z_old = extract_summary_from_model(model_old)

    # inducing set for the new model
    if Z_new is None:
        Z_use = Z_old
    else:
        Z_use = np.asarray(Z_new, dtype=np.float64)

    # clone kernel to avoid variable-sharing surprises
    k_new = clone_kernel(model_old.kernel)

    m = OSGPR_VFE(
        data=(np.asarray(X_new, dtype=np.float64), np.asarray(Y_new, dtype=np.float64)),
        kernel=k_new,
        mu_old=mu_old, Su_old=Su_old, Kaa_old=Kaa_old, Z_old=Z_old,
        Z=Z_use,
    )
    m.likelihood.variance.assign(float(noise))

    if freeze_kernel:
        try:
            m.kernel.variance.trainable = False
            m.kernel.lengthscales.trainable = False
        except Exception:
            pass

    t_sec, neg = train_osgpr(m, iters=iters, lr=lr, clip_norm=clip_norm)
    m.build_predict_cache()
    return m, float(t_sec), float(neg)

print("✅ OSGPR core + helpers ready (Cell 3 aligned to pipeline)")
