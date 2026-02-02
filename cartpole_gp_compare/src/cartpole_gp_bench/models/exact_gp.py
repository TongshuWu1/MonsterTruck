# models/exact_gp.py
from __future__ import annotations

from dataclasses import dataclass
from typing import Optional, Tuple, List

import numpy as np
import tensorflow as tf
import gpflow

from .base import TimedModelMixin, ensure_xy


DTYPE = gpflow.default_float()


@dataclass
class ExactGPParams:
    n_window: Optional[int] = None   # None -> keep ALL data forever
    fit_steps: int = 80
    update_steps: int = 60
    lr: float = 0.03

    # fixed likelihood noise
    noise: float = 1e-4

    # if True, freeze kernel AFTER init
    freeze_kernel_after_init: bool = False


def _make_kernel() -> gpflow.kernels.Kernel:
    return gpflow.kernels.SquaredExponential(
        lengthscales=np.ones(6, dtype=np.float64),
        variance=1.0,
    )


def _train_gpr_adam(model: gpflow.models.GPR, steps: int, lr: float):
    opt = tf.optimizers.Adam(float(lr))

    @tf.function
    def step_fn():
        with tf.GradientTape() as tape:
            loss = -model.log_marginal_likelihood()
        vars_ = model.trainable_variables
        grads = tape.gradient(loss, vars_)
        pairs = [(g, v) for g, v in zip(grads, vars_) if g is not None]
        if pairs:
            opt.apply_gradients(pairs)
        return loss

    for _ in range(int(steps)):
        step_fn()


class ExactGPDynamics(TimedModelMixin):
    """
    Exact GP baseline:
      - 4 independent GPR models (one per delta output dim)
      - update() APPENDS new transitions into stored dataset (unless n_window caps)
      - retrains on the whole stored dataset every update
      - predict() returns (mu, var)
    """

    def __init__(self, *, name: str = "ExactGP", params: Optional[ExactGPParams] = None):
        super().__init__(name=name)
        self.p = params if params is not None else ExactGPParams()

        self._X: Optional[np.ndarray] = None
        self._Y: Optional[np.ndarray] = None
        self._models: List[gpflow.models.GPR] = []

        self._set_extra(
            kind="exact",
            n_window=None if self.p.n_window is None else int(self.p.n_window),
            noise=float(self.p.noise),
        )

    def _apply_window(self):
        if self._X is None or self._Y is None:
            return
        if self.p.n_window is None:
            return
        w = int(self.p.n_window)
        if self._X.shape[0] > w:
            self._X = self._X[-w:]
            self._Y = self._Y[-w:]

    def _build_models(self):
        self._models = []
        for j in range(4):
            kern = _make_kernel()
            lik = gpflow.likelihoods.Gaussian(variance=float(self.p.noise))
            gpflow.set_trainable(lik, False)  # keep noise fixed

            m = gpflow.models.GPR(
                data=(self._X, self._Y[:, j:j+1]),
                kernel=kern,
                mean_function=None,
                likelihood=lik,
            )
            self._models.append(m)

        if bool(self.p.freeze_kernel_after_init):
            for m in self._models:
                try:
                    m.kernel.lengthscales.trainable = False
                    m.kernel.variance.trainable = False
                except Exception:
                    pass

    def fit_init(self, X: np.ndarray, Y: np.ndarray) -> None:
        X, Y = ensure_xy(X, Y)

        self._X = np.asarray(X, dtype=np.float64).copy()
        self._Y = np.asarray(Y, dtype=np.float64).copy()
        self._apply_window()

        def _fit():
            self._build_models()
            for m in self._models:
                _train_gpr_adam(m, steps=int(self.p.fit_steps), lr=float(self.p.lr))

        self._timeit("fit", _fit)
        self._set_n_data(int(self._X.shape[0]))
        self._set_n_inducing(None)

    def update(self, X_new: np.ndarray, Y_new: np.ndarray) -> None:
        X_new, Y_new = ensure_xy(X_new, Y_new)

        def _upd():
            if self._X is None or self._Y is None:
                raise RuntimeError("Call fit_init before update().")

            # ✅ append to ALL old data (unless n_window caps)
            self._X = np.concatenate([self._X, np.asarray(X_new, dtype=np.float64)], axis=0)
            self._Y = np.concatenate([self._Y, np.asarray(Y_new, dtype=np.float64)], axis=0)
            self._apply_window()

            self._build_models()
            for m in self._models:
                _train_gpr_adam(m, steps=int(self.p.update_steps), lr=float(self.p.lr))

        self._timeit("update", _upd)
        self._set_n_data(int(self._X.shape[0]))
        self._set_n_inducing(None)

    def predict(self, Xq: np.ndarray) -> Tuple[np.ndarray, Optional[np.ndarray]]:
        Xq = np.asarray(Xq, dtype=np.float64)
        if Xq.ndim == 1:
            Xq = Xq.reshape(1, -1)
        if Xq.shape[1] != 6:
            raise ValueError(f"Xq must be (B,6), got {Xq.shape}")
        if len(self._models) != 4:
            raise RuntimeError("Call fit_init before predict().")

        def _pred():
            Xtf = tf.convert_to_tensor(Xq, dtype=DTYPE)
            mus = []
            vars_ = []
            for m in self._models:
                mu_tf, var_tf = m.predict_f(Xtf)  # variance
                mus.append(mu_tf.numpy())
                vars_.append(var_tf.numpy())
            mu = np.concatenate(mus, axis=1).astype(np.float64, copy=False)
            var = np.concatenate(vars_, axis=1).astype(np.float64, copy=False)
            var = np.maximum(var, 0.0)
            return mu, var

        return self._timeit("predict", _pred)
