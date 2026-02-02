# models/svgp.py
from __future__ import annotations

from dataclasses import dataclass
from typing import Optional, Tuple, List

import numpy as np
import tensorflow as tf
import gpflow

from .base import TimedModelMixin, ensure_xy


DTYPE = gpflow.default_float()


@dataclass
class SVGPParams:
    M: int = 256
    max_data: Optional[int] = None   # None -> keep ALL data forever
    fit_steps: int = 80
    update_steps: int = 40
    lr: float = 0.03
    batch_size: int = 512
    seed: int = 0

    # fixed likelihood noise
    noise: float = 1e-4

    # if True, freeze kernel params AFTER initial fit
    freeze_kernel_after_init: bool = False


def _make_kernel() -> gpflow.kernels.Kernel:
    return gpflow.kernels.SquaredExponential(
        lengthscales=np.ones(6, dtype=np.float64),
        variance=1.0,
    )


def _init_Z(X: np.ndarray, M: int, rng: np.random.Generator) -> np.ndarray:
    X = np.asarray(X, dtype=np.float64)
    N = X.shape[0]
    if N <= M:
        return X.copy()
    idx = rng.choice(N, size=M, replace=False)
    return X[idx].copy()


def _as_tf(x: np.ndarray) -> tf.Tensor:
    return tf.convert_to_tensor(np.asarray(x, dtype=np.float64), dtype=DTYPE)


def _train_svgp_adam(
    model: gpflow.models.SVGP,
    X: np.ndarray,
    Y: np.ndarray,
    steps: int,
    lr: float,
    batch_size: int,
):
    """
    Stable SVGP training using Adam only (no NatGrad).
    Uses mini-batches and repeats dataset for `steps` updates.
    """
    X = np.asarray(X, dtype=np.float64)
    Y = np.asarray(Y, dtype=np.float64).reshape(-1, 1)

    opt = tf.optimizers.Adam(float(lr))

    ds = tf.data.Dataset.from_tensor_slices((_as_tf(X), _as_tf(Y)))
    ds = ds.shuffle(buffer_size=min(10_000, X.shape[0]), reshuffle_each_iteration=True)
    ds = ds.repeat()
    ds = ds.batch(int(batch_size), drop_remainder=False)
    it = iter(ds)

    @tf.function
    def step_fn(xb, yb):
        with tf.GradientTape() as tape:
            loss = model.training_loss((xb, yb))
        vars_ = model.trainable_variables
        grads = tape.gradient(loss, vars_)
        pairs = [(g, v) for g, v in zip(grads, vars_) if g is not None]
        if pairs:
            opt.apply_gradients(pairs)
        return loss

    for _ in range(int(steps)):
        xb, yb = next(it)
        step_fn(xb, yb)


class SVGPDynamics(TimedModelMixin):
    """
    SVGP baseline:
      - 4 independent SVGPs (one per delta output dim)
      - update() APPENDS new transitions into stored dataset (unless max_data caps)
      - retrains on the whole stored dataset every update
      - predict() returns (mu, var)
    """

    def __init__(self, *, name: str = "SVGP", params: Optional[SVGPParams] = None, seed: Optional[int] = None):
        super().__init__(name=name)
        self.p = params if params is not None else SVGPParams()
        if seed is not None:
            self.p.seed = int(seed)

        self._rng = np.random.default_rng(int(self.p.seed))

        self._X: Optional[np.ndarray] = None
        self._Y: Optional[np.ndarray] = None
        self._models: List[gpflow.models.SVGP] = []

        self._set_extra(
            kind="svgp",
            M=int(self.p.M),
            max_data=None if self.p.max_data is None else int(self.p.max_data),
            noise=float(self.p.noise),
        )

    def _apply_max_data(self):
        if self._X is None or self._Y is None:
            return
        if self.p.max_data is None:
            return
        md = int(self.p.max_data)
        if self._X.shape[0] > md:
            self._X = self._X[-md:]
            self._Y = self._Y[-md:]

    def _maybe_freeze_kernel(self):
        if not bool(self.p.freeze_kernel_after_init):
            return
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
        self._apply_max_data()

        def _fit():
            self._models = []
            for j in range(4):
                kern = _make_kernel()
                lik = gpflow.likelihoods.Gaussian(variance=float(self.p.noise))
                gpflow.set_trainable(lik, False)  # keep noise fixed

                Z = _init_Z(self._X, int(self.p.M), self._rng)

                m = gpflow.models.SVGP(
                    kernel=kern,
                    likelihood=lik,
                    inducing_variable=Z,
                    num_latent_gps=1,
                    mean_function=None,
                    whiten=True,
                    q_diag=False,
                )
                self._models.append(m)

            for j, m in enumerate(self._models):
                _train_svgp_adam(
                    m,
                    self._X,
                    self._Y[:, j:j+1],
                    steps=int(self.p.fit_steps),
                    lr=float(self.p.lr),
                    batch_size=int(self.p.batch_size),
                )

            self._maybe_freeze_kernel()

        self._timeit("fit", _fit)
        self._set_n_data(int(self._X.shape[0]))
        self._set_n_inducing(int(self.p.M))

    def update(self, X_new: np.ndarray, Y_new: np.ndarray) -> None:
        X_new, Y_new = ensure_xy(X_new, Y_new)

        def _upd():
            if self._X is None or self._Y is None or len(self._models) == 0:
                raise RuntimeError("Call fit_init before update().")

            # ✅ append to ALL old data (unless max_data caps)
            self._X = np.concatenate([self._X, np.asarray(X_new, dtype=np.float64)], axis=0)
            self._Y = np.concatenate([self._Y, np.asarray(Y_new, dtype=np.float64)], axis=0)
            self._apply_max_data()

            for j, m in enumerate(self._models):
                _train_svgp_adam(
                    m,
                    self._X,
                    self._Y[:, j:j+1],
                    steps=int(self.p.update_steps),
                    lr=float(self.p.lr),
                    batch_size=int(self.p.batch_size),
                )

        self._timeit("update", _upd)
        self._set_n_data(int(self._X.shape[0]))
        self._set_n_inducing(int(self.p.M))

    def predict(self, Xq: np.ndarray) -> Tuple[np.ndarray, Optional[np.ndarray]]:
        Xq = np.asarray(Xq, dtype=np.float64)
        if Xq.ndim == 1:
            Xq = Xq.reshape(1, -1)
        if Xq.shape[1] != 6:
            raise ValueError(f"Xq must be (B,6), got {Xq.shape}")
        if len(self._models) != 4:
            raise RuntimeError("Call fit_init before predict().")

        def _pred():
            Xtf = _as_tf(Xq)
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
