# models/palsgp.py
from __future__ import annotations

import numpy as np
import tensorflow as tf
import gpflow

from .base import TimedModelMixin, ensure_xy
from ._osgpr_core import (
    sym_jitter,
    prior_summary,
    train_osgpr,
    OSGPR_VFE,
    greedy_dopt_anchors_from_K,
    grow_Z_global,
    osgpr_stream_update,
)

# Your notebook is currently inside cartpole_gp_bench/, same level as cartpole_env.py
from cartpole_env import wrap_pi, state_to_features


def se_ard_kernel_Kzx(Z, X, lengthscales, variance):
    Z = np.asarray(Z, dtype=np.float64)
    X = np.asarray(X, dtype=np.float64)
    ls = np.asarray(lengthscales, dtype=np.float64).reshape(1, -1)
    var = float(variance)

    Zs = Z / ls
    Xs = X / ls
    z2 = np.sum(Zs * Zs, axis=1, keepdims=True)
    x2 = np.sum(Xs * Xs, axis=1, keepdims=True).T
    zx = Zs @ Xs.T
    r2 = np.maximum(z2 + x2 - 2.0 * zx, 0.0)
    return var * np.exp(-0.5 * r2)


def build_subset_predictor_from_global(model_global, idx_sub, jitter=1e-6):
    idx_sub = np.asarray(idx_sub, dtype=np.int64)

    Zg = model_global.inducing_variable.Z.numpy()
    Z = Zg[idx_sub].copy().astype(np.float64)

    Ztf = tf.convert_to_tensor(Z, dtype=gpflow.default_float())
    mu_tf, Sig_tf = model_global.predict_f(Ztf, full_cov=True)
    m = mu_tf.numpy().reshape(-1, 1)

    S = Sig_tf.numpy()
    if S.ndim == 3:
        S = S[0]
    S = sym_jitter(S, 1e-6)

    ls = model_global.kernel.lengthscales.numpy()
    var = model_global.kernel.variance.numpy()

    Kzz = se_ard_kernel_Kzx(Z, Z, ls, var)
    Kzz = sym_jitter(Kzz, jitter=jitter)
    L = np.linalg.cholesky(Kzz)

    y = np.linalg.solve(L, m)
    alpha = np.linalg.solve(L.T, y)

    def predict(Xfeat):
        Xfeat = np.asarray(Xfeat, dtype=np.float64)
        if Xfeat.ndim == 1:
            Xfeat = Xfeat.reshape(1, -1)

        Kzx = se_ard_kernel_Kzx(Z, Xfeat, ls, var)

        W = np.linalg.solve(L, Kzx)
        W = np.linalg.solve(L.T, W)

        mu = (Kzx.T @ alpha).reshape(-1)

        kxx = float(var) * np.ones((Xfeat.shape[0],), dtype=np.float64)
        Qdiag = np.sum(Kzx * W, axis=0)
        SW = S @ W
        Sdiag = np.sum(W * SW, axis=0)

        v = kxx - Qdiag + Sdiag
        v = np.maximum(v, 1e-12)
        return mu, v

    return predict, Z


def rollout_tube_features(state, u_seq, pred_bundle):
    pred_dx, pred_dxdot, pred_dth, pred_dthdot = pred_bundle
    H = int(len(u_seq))
    tubeX = np.zeros((H, 6), dtype=np.float64)

    s = np.asarray(state, dtype=np.float64).reshape(4,)
    for t in range(H):
        u = float(u_seq[t])
        xfeat = state_to_features(s[0], s[1], s[2], s[3], u)
        tubeX[t] = xfeat

        xfeat2 = xfeat[None, :]
        dx = float(pred_dx(xfeat2)[0])
        dxdot = float(pred_dxdot(xfeat2)[0])
        dth = float(pred_dth(xfeat2)[0])
        dthdot = float(pred_dthdot(xfeat2)[0])

        s = np.array([s[0] + dx, s[1] + dxdot, wrap_pi(s[2] + dth), s[3] + dthdot], dtype=np.float64)

    return tubeX


def select_subset_from_tube(Z_global, tubeX, m_sub, anchor_idx, diversity_frac=0.35, n_close=None):
    Zg = np.asarray(Z_global, dtype=np.float64)
    tubeX = np.asarray(tubeX, dtype=np.float64)
    M = Zg.shape[0]

    anchor_idx = (
        np.unique(np.asarray(anchor_idx, dtype=np.int64))
        if anchor_idx is not None
        else np.array([], dtype=np.int64)
    )

    chosen = list(anchor_idx[: min(len(anchor_idx), m_sub)])
    chosen_set = set(int(i) for i in chosen)

    remain = int(m_sub - len(chosen))
    if remain <= 0:
        return np.array(chosen[:m_sub], dtype=np.int64)

    diff = Zg[:, None, :] - tubeX[None, :, :]
    d2 = np.sum(diff * diff, axis=2)
    mind2 = np.min(d2, axis=1)

    if n_close is None:
        n_close = int(max(4 * m_sub, 64))

    order = np.argsort(mind2)

    cand = []
    for idx in order:
        ii = int(idx)
        if ii in chosen_set:
            continue
        cand.append(ii)
        if len(cand) >= n_close:
            break

    if len(cand) == 0:
        for ii in range(M):
            if ii not in chosen_set:
                chosen.append(ii)
                chosen_set.add(ii)
                if len(chosen) >= m_sub:
                    break
        return np.array(chosen[:m_sub], dtype=np.int64)

    n_div = int(np.round(diversity_frac * remain))
    n_close_pick = remain - n_div

    for i in range(n_close_pick):
        if len(chosen) >= m_sub:
            break
        ii = cand[i % len(cand)]
        if ii not in chosen_set:
            chosen.append(ii)
            chosen_set.add(ii)

    Zcand = Zg[cand]
    for _ in range(n_div):
        if len(chosen) >= m_sub:
            break
        chosen_idx = np.array(list(chosen_set), dtype=np.int64)
        Zchosen = Zg[chosen_idx]

        d2cand = np.sum((Zcand[:, None, :] - Zchosen[None, :, :]) ** 2, axis=2)
        min_d2 = np.min(d2cand, axis=1)
        far_order = np.argsort(-min_d2)

        picked = None
        for j in far_order:
            ii = cand[int(j)]
            if ii not in chosen_set:
                picked = ii
                break
        if picked is None:
            break
        chosen.append(int(picked))
        chosen_set.add(int(picked))

    return np.array(np.unique(chosen[:m_sub]), dtype=np.int64)


class PALSGPDynamics(TimedModelMixin):
    def __init__(
        self,
        *,
        name="PALSGP",
        seed=0,
        M_global_init=128,
        M_add_per_update=48,
        M_global_max=512,
        M_sub=64,
        diversity_frac=0.35,
        anchor_m=18,
        anchor_lam=1e-6,
        iters_init=250,
        lr_init=0.02,
        noise_init=1e-4,
        iters_update=80,
        lr_update=0.02,
        noise_update=1e-4,
        freeze_kernel=True,
        clip_norm=10.0,
        subset_jitter=1e-6,
    ):
        super().__init__(name)
        self.rng = np.random.default_rng(int(seed))

        self.M_global_init = int(M_global_init)
        self.M_add_per_update = int(M_add_per_update)
        self.M_global_max = int(M_global_max)

        self.M_sub = int(M_sub)
        self.diversity_frac = float(diversity_frac)

        self.anchor_m = int(anchor_m)
        self.anchor_lam = float(anchor_lam)

        self.iters_init = int(iters_init)
        self.lr_init = float(lr_init)
        self.noise_init = float(noise_init)

        self.iters_update = int(iters_update)
        self.lr_update = float(lr_update)
        self.noise_update = float(noise_update)

        self.freeze_kernel = bool(freeze_kernel)
        self.clip_norm = float(clip_norm)
        self.subset_jitter = float(subset_jitter)

        self.Z_GLOBAL = None
        self.ANCHOR_IDX = None
        self.IDX_SUB = None

        self.m_dx = None
        self.m_dxdot = None
        self.m_dth = None
        self.m_dthdot = None

        self._pred_local_full = None
        self._unc_local = None

        self._n_data = 0
        self._set_extra(kind="palsgp")

    def _make_kernel_6d(self):
        return gpflow.kernels.SquaredExponential(lengthscales=np.ones(6, dtype=np.float64), variance=1.0)

    def _pred_mean_cached(self, model, Xfeat):
        Xfeat = np.asarray(Xfeat, dtype=np.float64)
        if Xfeat.ndim == 1:
            Xfeat = Xfeat.reshape(1, -1)
        Xtf = tf.convert_to_tensor(Xfeat, dtype=gpflow.default_float())
        if hasattr(model, "predict_f_cached"):
            mu_tf, _ = model.predict_f_cached(Xtf, full_cov=False)
        else:
            mu_tf, _ = model.predict_f(Xtf, full_cov=False)
        return mu_tf.numpy().reshape(-1)

    def _make_global_predictors_bundle(self):
        return (
            lambda X: self._pred_mean_cached(self.m_dx, X),
            lambda X: self._pred_mean_cached(self.m_dxdot, X),
            lambda X: self._pred_mean_cached(self.m_dth, X),
            lambda X: self._pred_mean_cached(self.m_dthdot, X),
        )

    def _compute_anchor_idx(self):
        Z = np.asarray(self.Z_GLOBAL, dtype=np.float64)
        Kzz = self.m_dxdot.kernel.K(Z).numpy()
        self.ANCHOR_IDX = greedy_dopt_anchors_from_K(Kzz, m_anchors=self.anchor_m, lam=self.anchor_lam)

    def _rebuild_local_from_idx(self, idx_sub):
        idx_sub = np.asarray(idx_sub, dtype=np.int64)
        self.IDX_SUB = idx_sub

        p_dx, _ = build_subset_predictor_from_global(self.m_dx, idx_sub, jitter=self.subset_jitter)
        p_dxdot, _ = build_subset_predictor_from_global(self.m_dxdot, idx_sub, jitter=self.subset_jitter)
        p_dth, _ = build_subset_predictor_from_global(self.m_dth, idx_sub, jitter=self.subset_jitter)
        p_dthdot, _ = build_subset_predictor_from_global(self.m_dthdot, idx_sub, jitter=self.subset_jitter)

        self._pred_local_full = (p_dx, p_dxdot, p_dth, p_dthdot)

        def unc(Xfeat):
            _, v = p_dxdot(Xfeat)
            return v

        self._unc_local = unc

    def get_uncertainty_fn(self):
        return self._unc_local

    def set_tube(self, tubeX: np.ndarray) -> None:
        if self.Z_GLOBAL is None or self.ANCHOR_IDX is None:
            return
        tubeX = np.asarray(tubeX, dtype=np.float64)
        if tubeX.ndim != 2 or tubeX.shape[1] != 6:
            return
        idx_sub = select_subset_from_tube(
            self.Z_GLOBAL,
            tubeX,
            m_sub=self.M_sub,
            anchor_idx=self.ANCHOR_IDX,
            diversity_frac=self.diversity_frac,
        )
        self._rebuild_local_from_idx(idx_sub)

    def rebuild_local(self, state, u_mean):
        if self.Z_GLOBAL is None:
            return
        pred_global = self._make_global_predictors_bundle()
        tubeX = rollout_tube_features(state, u_mean, pred_global)
        idx_sub = select_subset_from_tube(
            self.Z_GLOBAL,
            tubeX,
            m_sub=self.M_sub,
            anchor_idx=self.ANCHOR_IDX,
            diversity_frac=self.diversity_frac,
        )
        self._rebuild_local_from_idx(idx_sub)

    def fit_init(self, X: np.ndarray, Y: np.ndarray) -> None:
        X, Y = ensure_xy(X, Y)
        self._n_data = int(X.shape[0])

        def _fit():
            M0 = int(min(self.M_global_init, X.shape[0]))
            idxZ = self.rng.choice(X.shape[0], size=M0, replace=False)
            self.Z_GLOBAL = np.asarray(X[idxZ], dtype=np.float64).copy()

            k_dx = self._make_kernel_6d()
            k_dxdot = self._make_kernel_6d()
            k_dth = self._make_kernel_6d()
            k_dthdot = self._make_kernel_6d()

            # Correct init: create OSGPR_VFE from prior summary at Z_GLOBAL
            def build_initial(kernel, y_col):
                mu0, Su0, Kaa0, Z0 = prior_summary(kernel, self.Z_GLOBAL)
                m = OSGPR_VFE(
                    data=(np.asarray(X, dtype=np.float64), np.asarray(y_col, dtype=np.float64).reshape(-1, 1)),
                    kernel=kernel,
                    mu_old=mu0,
                    Su_old=Su0,
                    Kaa_old=Kaa0,
                    Z_old=Z0,
                    Z=self.Z_GLOBAL,
                )
                m.likelihood.variance.assign(float(self.noise_init))

                # init: allow kernel to train
                try:
                    m.kernel.variance.trainable = True
                    m.kernel.lengthscales.trainable = True
                except Exception:
                    pass

                train_osgpr(
                    m,
                    iters=int(self.iters_init),
                    lr=float(self.lr_init),
                    clip_norm=float(self.clip_norm),
                )
                m.build_predict_cache()
                return m

            self.m_dx = build_initial(k_dx, Y[:, 0])
            self.m_dxdot = build_initial(k_dxdot, Y[:, 1])
            self.m_dth = build_initial(k_dth, Y[:, 2])
            self.m_dthdot = build_initial(k_dthdot, Y[:, 3])

            self._compute_anchor_idx()
            self._pred_local_full = None
            self._unc_local = None
            self.IDX_SUB = None

        self._timeit("fit", _fit)
        self._set_n_data(self._n_data)
        self._set_n_inducing(int(self.Z_GLOBAL.shape[0]))

    def update(self, X_new: np.ndarray, Y_new: np.ndarray) -> None:
        X_new, Y_new = ensure_xy(X_new, Y_new)
        self._n_data += int(X_new.shape[0])

        def _upd():
            self.Z_GLOBAL = grow_Z_global(
                self.Z_GLOBAL,
                np.asarray(X_new, dtype=np.float64),
                add_m=self.M_add_per_update,
                max_m=self.M_global_max,
            )

            self.m_dx, _ = osgpr_stream_update(
                self.m_dx,
                np.asarray(X_new, dtype=np.float64),
                np.asarray(Y_new[:, 0:1], dtype=np.float64),
                self.Z_GLOBAL,
                iters=self.iters_update,
                lr=self.lr_update,
                noise=self.noise_update,
                freeze_kernel=self.freeze_kernel,
                clip_norm=self.clip_norm,
            )
            self.m_dxdot, _ = osgpr_stream_update(
                self.m_dxdot,
                np.asarray(X_new, dtype=np.float64),
                np.asarray(Y_new[:, 1:2], dtype=np.float64),
                self.Z_GLOBAL,
                iters=self.iters_update,
                lr=self.lr_update,
                noise=self.noise_update,
                freeze_kernel=self.freeze_kernel,
                clip_norm=self.clip_norm,
            )
            self.m_dth, _ = osgpr_stream_update(
                self.m_dth,
                np.asarray(X_new, dtype=np.float64),
                np.asarray(Y_new[:, 2:3], dtype=np.float64),
                self.Z_GLOBAL,
                iters=self.iters_update,
                lr=self.lr_update,
                noise=self.noise_update,
                freeze_kernel=self.freeze_kernel,
                clip_norm=self.clip_norm,
            )
            self.m_dthdot, _ = osgpr_stream_update(
                self.m_dthdot,
                np.asarray(X_new, dtype=np.float64),
                np.asarray(Y_new[:, 3:4], dtype=np.float64),
                self.Z_GLOBAL,
                iters=self.iters_update,
                lr=self.lr_update,
                noise=self.noise_update,
                freeze_kernel=self.freeze_kernel,
                clip_norm=self.clip_norm,
            )

            self._compute_anchor_idx()
            self._pred_local_full = None
            self._unc_local = None
            self.IDX_SUB = None

        self._timeit("update", _upd)
        self._set_n_data(self._n_data)
        self._set_n_inducing(int(self.Z_GLOBAL.shape[0]))

    def predict(self, Xq: np.ndarray):
        Xq = np.asarray(Xq, dtype=np.float64)
        if Xq.ndim == 1:
            Xq = Xq.reshape(1, -1)
        if Xq.shape[1] != 6:
            raise ValueError(f"Xq must be (B,6), got {Xq.shape}")

        def _pred():
            if self._pred_local_full is not None:
                p_dx, p_dxdot, p_dth, p_dthdot = self._pred_local_full
                mu0, v0 = p_dx(Xq)
                mu1, v1 = p_dxdot(Xq)
                mu2, v2 = p_dth(Xq)
                mu3, v3 = p_dthdot(Xq)
                mu = np.stack([mu0, mu1, mu2, mu3], axis=1).astype(np.float64)
                var = np.stack([v0, v1, v2, v3], axis=1).astype(np.float64)
                return mu, var

            Xtf = tf.convert_to_tensor(Xq, dtype=gpflow.default_float())
            mus = []
            vars_ = []
            for m in (self.m_dx, self.m_dxdot, self.m_dth, self.m_dthdot):
                if hasattr(m, "predict_f_cached"):
                    mu_tf, var_tf = m.predict_f_cached(Xtf, full_cov=False)
                else:
                    mu_tf, var_tf = m.predict_f(Xtf, full_cov=False)
                mus.append(mu_tf.numpy().reshape(-1))
                vars_.append(var_tf.numpy().reshape(-1))
            mu = np.stack(mus, axis=1).astype(np.float64)
            var = np.stack(vars_, axis=1).astype(np.float64)
            return mu, var

        return self._timeit("predict", _pred)
