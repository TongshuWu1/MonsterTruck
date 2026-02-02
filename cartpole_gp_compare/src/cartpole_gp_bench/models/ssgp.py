# models/ssgp.py
from __future__ import annotations

from dataclasses import dataclass
from typing import Optional, Tuple

import numpy as np
import tensorflow as tf
import gpflow

from .base import TimedModelMixin, ensure_xy
from ._osgpr_core import (
    OSGPR_VFE,
    prior_summary,
    train_osgpr,
    osgpr_stream_update,
)

DTYPE = gpflow.default_float()


@dataclass
class SSGPParams:
    M_init: int = 128  # fixed inducing size

    iters_init: int = 250
    iters_update: int = 150
    lr_init: float = 0.02
    lr_update: float = 0.02

    noise_init: float = 1e-4
    noise_update: float = 1e-4
    freeze_kernel: bool = True
    clip_norm: float = 10.0

    seed: int = 0


def _make_kernel() -> gpflow.kernels.Kernel:
    return gpflow.kernels.SquaredExponential(lengthscales=np.ones(6, dtype=np.float64))


def _as_tf64(x: np.ndarray) -> tf.Tensor:
    return tf.convert_to_tensor(np.asarray(x, dtype=np.float64), dtype=DTYPE)


class SSGPDynamics(TimedModelMixin):
    """
    Standard Streaming Sparse GP (SSGP) — FIXED inducing set size.

    - Z_global is fixed at size M_init (picked from initial data once)
    - 4 independent OSGPR_VFE models (one per delta dimension)
    - update uses osgpr_stream_update with Z_new = Z_global (unchanged)
    - predict uses predict_f_cached
    """

    def __init__(self, name: str = "SSGP", params: Optional[SSGPParams] = None):
        super().__init__(name=name)
        self.p = params if params is not None else SSGPParams()

        self.kernels = [_make_kernel() for _ in range(4)]
        self.models: list[Optional[OSGPR_VFE]] = [None] * 4

        self.Z_global: Optional[np.ndarray] = None
        self._n_data = 0
        self._rng = np.random.default_rng(self.p.seed)

        self._set_extra(kind="ssgp_fixedM", M_init=int(self.p.M_init))

    def fit_init(self, X: np.ndarray, Y: np.ndarray) -> None:
        X, Y = ensure_xy(X, Y)

        def _fit():
            self._n_data = int(X.shape[0])

            M0 = int(min(self.p.M_init, X.shape[0]))

            # pick inducing points once (random subset is typical; or X[:M0] if you prefer deterministic)
            idx = self._rng.choice(X.shape[0], size=M0, replace=False)
            self.Z_global = np.asarray(X[idx], dtype=np.float64).copy()

            for j in range(4):
                yj = np.asarray(Y[:, j:j + 1], dtype=np.float64)

                mu0, Su0, Kaa0, Z0 = prior_summary(self.kernels[j], self.Z_global)

                m = OSGPR_VFE(
                    data=(np.asarray(X, dtype=np.float64), yj),
                    kernel=self.kernels[j],
                    mu_old=mu0,
                    Su_old=Su0,
                    Kaa_old=Kaa0,
                    Z_old=Z0,
                    Z=self.Z_global,
                )
                m.likelihood.variance.assign(float(self.p.noise_init))

                if self.p.freeze_kernel:
                    try:
                        m.kernel.variance.trainable = False
                        m.kernel.lengthscales.trainable = False
                    except Exception:
                        pass

                train_osgpr(
                    m,
                    iters=int(self.p.iters_init),
                    lr=float(self.p.lr_init),
                    clip_norm=float(self.p.clip_norm),
                )
                m.build_predict_cache()
                self.models[j] = m

            self._set_n_data(self._n_data)
            self._set_n_inducing(int(self.Z_global.shape[0]))

        self._timeit("fit", _fit)

    def update(self, X_new: np.ndarray, Y_new: np.ndarray) -> None:
        X_new, Y_new = ensure_xy(X_new, Y_new)

        def _upd():
            if self.Z_global is None or any(m is None for m in self.models):
                raise RuntimeError("Call fit_init before update().")

            self._n_data += int(X_new.shape[0])

            # IMPORTANT: fixed inducing set (no grow_Z_global)
            Z_fixed = self.Z_global

            for j in range(4):
                model_old = self.models[j]
                assert model_old is not None

                yj = np.asarray(Y_new[:, j:j + 1], dtype=np.float64)

                model_new, _info = osgpr_stream_update(
                    model_old=model_old,
                    X_new=np.asarray(X_new, dtype=np.float64),
                    Y_new=yj,
                    Z_new=Z_fixed,  # unchanged
                    iters=int(self.p.iters_update),
                    lr=float(self.p.lr_update),
                    noise=float(self.p.noise_update),
                    freeze_kernel=bool(self.p.freeze_kernel),
                    clip_norm=float(self.p.clip_norm),
                )
                self.models[j] = model_new

            self._set_n_data(self._n_data)
            self._set_n_inducing(int(Z_fixed.shape[0]))

        self._timeit("update", _upd)

    def predict(self, Xq: np.ndarray) -> Tuple[np.ndarray, Optional[np.ndarray]]:
        Xq = np.asarray(Xq, dtype=np.float64)
        if Xq.ndim == 1:
            Xq = Xq.reshape(1, -1)
        if Xq.shape[1] != 6:
            raise ValueError(f"Xq must be (B,6), got {Xq.shape}")
        if any(m is None for m in self.models):
            raise RuntimeError("Call fit_init before predict().")

        def _to_2d_col(a: np.ndarray) -> np.ndarray:
            a = np.asarray(a)
            if a.ndim == 1:
                return a.reshape(-1, 1)
            if a.ndim == 2 and a.shape[1] == 1:
                return a
            if a.ndim == 2:
                return a[:, :1]
            raise ValueError(f"Unexpected array shape: {a.shape}")

        def _pred():
            Xtf = _as_tf64(Xq)
            mus = []
            vars_ = []
            for j in range(4):
                m = self.models[j]
                assert m is not None
                mu_tf, var_tf = m.predict_f_cached(Xtf, full_cov=False)

                mu = _to_2d_col(mu_tf.numpy())
                var = _to_2d_col(var_tf.numpy())

                mus.append(mu)
                vars_.append(var)

            mu_all = np.concatenate(mus, axis=1).astype(np.float64, copy=False)
            var_all = np.concatenate(vars_, axis=1).astype(np.float64, copy=False)
            std_all = np.sqrt(np.maximum(var_all, 0.0))
            return mu_all, std_all

        return self._timeit("predict", _pred)
