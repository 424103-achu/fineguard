"""Error smoothing, nonparametric dynamic thresholding and pruning (Hundman et al. 2018, §3.2)."""

from __future__ import annotations

import numpy as np
import pandas as pd

from .config import Config


def smooth_errors(y_true: np.ndarray, y_hat: np.ndarray, cfg: Config) -> np.ndarray:
    """Absolute prediction error smoothed with an exponentially weighted moving average."""
    e = np.abs(np.asarray(y_true, dtype=float) - np.asarray(y_hat, dtype=float))
    span = max(1, int(cfg.error_window * cfg.smoothing_perc))
    return pd.Series(e).ewm(span=span).mean().to_numpy()


def group_sequences(idx: np.ndarray) -> list[tuple[int, int]]:
    """Contiguous runs of indices -> [(start, end), ...] (inclusive)."""
    if len(idx) == 0:
        return []
    idx = np.sort(np.unique(idx))
    breaks = np.where(np.diff(idx) > 1)[0]
    starts = np.r_[idx[0], idx[breaks + 1]]
    ends = np.r_[idx[breaks], idx[-1]]
    return [(int(s), int(e)) for s, e in zip(starts, ends)]


def _buffered(idx: np.ndarray, buffer: int, n: int) -> np.ndarray:
    if len(idx) == 0 or buffer == 0:
        return idx
    out = (idx[:, None] + np.arange(-buffer, buffer + 1)[None, :]).ravel()
    return np.unique(out[(out >= 0) & (out < n)])


def find_epsilon(e_s: np.ndarray, cfg: Config) -> float:
    """Pick eps = mean + z*std maximising
        (Δmean/mean + Δstd/std) / (|e_a| + |E_seq|^2)
    where Δ is the drop in mean/std after removing errors above eps, |e_a| the number of
    points above eps and |E_seq| the number of contiguous anomalous sequences."""
    mean, sd = float(np.mean(e_s)), float(np.std(e_s))
    best_eps, best_score = mean + cfg.z_range[1] * sd, -np.inf  # default: effectively nothing
    if sd == 0 or mean == 0:
        return best_eps
    for z in np.arange(*cfg.z_range):
        eps = mean + z * sd
        below = e_s[e_s < eps]
        i_anom = np.where(e_s >= eps)[0]
        if len(i_anom) == 0 or len(below) == 0:
            continue
        i_anom = _buffered(i_anom, cfg.error_buffer, len(e_s))
        seqs = group_sequences(i_anom)
        if len(i_anom) >= 0.5 * len(e_s) or len(seqs) > 5:
            continue
        d_mean = (mean - np.mean(below)) / mean
        d_sd = (sd - np.std(below)) / sd
        score = (d_mean + d_sd) / (len(i_anom) + len(seqs) ** 2)
        if score > best_score:
            best_score, best_eps = score, eps
    return best_eps


def prune(e_s: np.ndarray, seqs: list[tuple[int, int]], p: float) -> list[tuple[int, int]]:
    """Drop sequences whose max error is not at least `p` (fraction) above the next one.

    Sorted max errors of the flagged sequences, followed by the max error outside them, are
    compared pairwise; everything after the last drop larger than `p` is reclassified nominal.
    """
    if not seqs:
        return seqs
    mask = np.zeros(len(e_s), dtype=bool)
    for s, e in seqs:
        mask[s:e + 1] = True
    seq_max = np.array([e_s[s:e + 1].max() for s, e in seqs])
    normal_max = e_s[~mask].max() if (~mask).any() else 0.0
    order = np.argsort(-seq_max)
    sorted_max = np.r_[seq_max[order], normal_max]
    drops = (sorted_max[:-1] - sorted_max[1:]) / np.where(sorted_max[:-1] == 0, 1, sorted_max[:-1])
    keep_upto = np.where(drops > p)[0]
    if len(keep_upto) == 0:
        return []
    kept = set(order[: keep_upto.max() + 1].tolist())
    return [seq for i, seq in enumerate(seqs) if i in kept]


def detect(e_s: np.ndarray, cfg: Config, y_true: np.ndarray | None = None,
           prune_p: float | None = None) -> tuple[list[tuple[int, int]], np.ndarray]:
    """Run dynamic thresholding over consecutive windows of `cfg.error_window` errors.

    Returns flagged sequences (indices into e_s) and the per-timestep threshold used.
    If `y_true` is given, windows whose largest error is below 5% of the signal range are
    skipped, as in the reference implementation (tiny errors on near-flat signals).
    """
    n = len(e_s)
    win = cfg.error_window if cfg.error_window and cfg.error_window < n else n
    thresholds = np.zeros(n)
    flagged: list[np.ndarray] = []
    for start in range(0, n, win):
        end = min(n, start + win)
        if end - start < win and start > 0:       # fold a short tail into the previous window
            start = max(0, end - win)
        window = e_s[start:end]
        eps = find_epsilon(window, cfg)
        thresholds[start:end] = eps
        if y_true is not None:
            yr = np.ptp(y_true[start:end])
            if window.max() < 0.05 * yr:
                continue
        idx = np.where(window >= eps)[0] + start
        flagged.append(idx)
    idx = np.unique(np.concatenate(flagged)) if flagged else np.array([], dtype=int)
    seqs = group_sequences(_buffered(idx, cfg.error_buffer, n))
    seqs = prune(e_s, seqs, cfg.p if prune_p is None else prune_p)
    return seqs, thresholds


def static_threshold(scores: np.ndarray, threshold: float, buffer: int = 0) -> list[tuple[int, int]]:
    idx = np.where(scores >= threshold)[0]
    return group_sequences(_buffered(idx, buffer, len(scores)))
