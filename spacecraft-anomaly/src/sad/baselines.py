"""Non-LSTM baselines. Each returns an anomaly score per test timestep (higher = more anomalous)
plus a threshold calibrated on the nominal train data only (no test labels are used)."""

from __future__ import annotations

import numpy as np
import pandas as pd
from sklearn.ensemble import IsolationForest


def _rolling_z(x: np.ndarray, window: int) -> np.ndarray:
    s = pd.Series(x)
    mu = s.rolling(window, min_periods=max(2, window // 2)).mean().shift(1)
    sd = s.rolling(window, min_periods=max(2, window // 2)).std().shift(1)
    z = (s - mu).abs() / sd.replace(0, np.nan)
    return z.fillna(0).to_numpy()


def zscore_scores(train: np.ndarray, test: np.ndarray, window: int = 250,
                  quantile: float = 0.999) -> tuple[np.ndarray, float]:
    """|x_t - rolling mean| / rolling std on the telemetry value (column 0)."""
    tr = _rolling_z(train[:, 0], window)
    te = _rolling_z(test[:, 0], window)
    return te, float(np.quantile(tr, quantile))


def _window_features(arr: np.ndarray, w: int) -> np.ndarray:
    x = np.pad(arr[:, 0], (w - 1, 0), mode="edge")
    idx = np.arange(w)[None, :] + np.arange(len(arr))[:, None]
    vals = x[idx]                                   # window ending at t
    cmds = arr[:, 1:] if arr.shape[1] > 1 else np.zeros((len(arr), 0))
    return np.hstack([vals, np.diff(vals, axis=1), cmds])


def iforest_scores(train: np.ndarray, test: np.ndarray, w: int = 30, quantile: float = 0.999,
                   seed: int = 0, max_train: int = 20000) -> tuple[np.ndarray, float]:
    Xtr = _window_features(train, w)
    Xte = _window_features(test, w)
    rng = np.random.default_rng(seed)
    if len(Xtr) > max_train:
        Xtr = Xtr[rng.choice(len(Xtr), max_train, replace=False)]
    model = IsolationForest(n_estimators=200, random_state=seed).fit(Xtr)
    tr = -model.score_samples(Xtr)
    te = -model.score_samples(Xte)
    return te, float(np.quantile(tr, quantile))
