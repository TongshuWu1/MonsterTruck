# models/base.py

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any, Dict, Optional, Protocol, Tuple

import numpy as np


@dataclass
class ModelStats:
    name: str
    n_data: int = 0
    n_inducing: Optional[int] = None
    fit_calls: int = 0
    update_calls: int = 0
    predict_calls: int = 0
    fit_ms: float = 0.0
    update_ms: float = 0.0
    predict_ms: float = 0.0
    extra: Dict[str, Any] = field(default_factory=dict)

    def as_dict(self) -> Dict[str, Any]:
        d = {
            "name": self.name,
            "n_data": int(self.n_data),
            "n_inducing": None if self.n_inducing is None else int(self.n_inducing),
            "fit_calls": int(self.fit_calls),
            "update_calls": int(self.update_calls),
            "predict_calls": int(self.predict_calls),
            "fit_ms": float(self.fit_ms),
            "update_ms": float(self.update_ms),
            "predict_ms": float(self.predict_ms),
        }
        d.update(self.extra)
        return d


class DynamicsModel(Protocol):
    def fit_init(self, X: np.ndarray, Y: np.ndarray) -> None:
        ...

    def update(self, X_new: np.ndarray, Y_new: np.ndarray) -> None:
        ...

    def predict(self, Xq: np.ndarray) -> Tuple[np.ndarray, Optional[np.ndarray]]:
        ...

    def stats(self) -> Dict[str, Any]:
        ...


def ensure_2d(a: np.ndarray, d: int) -> np.ndarray:
    a = np.asarray(a)
    if a.ndim == 1:
        a = a.reshape(1, -1)
    if a.ndim != 2 or a.shape[1] != d:
        raise ValueError(f"Expected shape (B,{d}), got {a.shape}")
    return a


def ensure_xy(X: np.ndarray, Y: np.ndarray) -> Tuple[np.ndarray, np.ndarray]:
    X = ensure_2d(X, 6).astype(np.float64, copy=False)
    Y = ensure_2d(Y, 4).astype(np.float64, copy=False)
    if X.shape[0] != Y.shape[0]:
        raise ValueError(f"X and Y batch mismatch: {X.shape} vs {Y.shape}")
    return X, Y


def time_now() -> float:
    import time
    return time.perf_counter()


class TimedModelMixin:
    def __init__(self, name: str):
        self._stats = ModelStats(name=name)

    def _timeit(self, kind: str, fn, *args, **kwargs):
        t0 = time_now()
        out = fn(*args, **kwargs)
        dt = (time_now() - t0) * 1000.0
        if kind == "fit":
            self._stats.fit_ms += dt
            self._stats.fit_calls += 1
        elif kind == "update":
            self._stats.update_ms += dt
            self._stats.update_calls += 1
        elif kind == "predict":
            self._stats.predict_ms += dt
            self._stats.predict_calls += 1
        return out

    def stats(self) -> Dict[str, Any]:
        return self._stats.as_dict()

    def _set_n_data(self, n: int):
        self._stats.n_data = int(n)

    def _set_n_inducing(self, m: Optional[int]):
        self._stats.n_inducing = None if m is None else int(m)

    def _set_extra(self, **kwargs):
        self._stats.extra.update(kwargs)
