"""Shared evaluation for every model, so the results table is a fair comparison.

Main metrics: 8-category micro/macro-F1 and unfair-sentence recall.
LexGLUE-style scores (extra "none" column) are only for comparing with the paper.
"""
import numpy as np
from sklearn.metrics import classification_report, f1_score, precision_score, recall_score

DEFAULT_GRID = np.round(np.arange(0.05, 0.96, 0.05), 2)


def add_none_column(y):
    """Append a 'no unfair term' column, as LexGLUE does for this task."""
    y = np.asarray(y)
    none = (y.sum(axis=1) == 0).astype(int).reshape(-1, 1)
    return np.hstack([y, none])


def evaluate(y_true, y_pred):
    y_true = np.asarray(y_true)
    y_pred = np.asarray(y_pred)
    yt9, yp9 = add_none_column(y_true), add_none_column(y_pred)
    any_true, any_pred = y_true.any(axis=1), y_pred.any(axis=1)
    return {
        # 8 unfair categories only: the honest measure of detection
        "micro_f1_8": float(f1_score(y_true, y_pred, average="micro", zero_division=0)),
        "macro_f1_8": float(f1_score(y_true, y_pred, average="macro", zero_division=0)),
        # LexGLUE-style (8 + 'none'): use when comparing with the paper
        "micro_f1_lexglue": float(f1_score(yt9, yp9, average="micro", zero_division=0)),
        "macro_f1_lexglue": float(f1_score(yt9, yp9, average="macro", zero_division=0)),
        # sentence level: is it potentially unfair at all?
        "unfair_precision": float(precision_score(any_true, any_pred, zero_division=0)),
        "unfair_recall": float(recall_score(any_true, any_pred, zero_division=0)),
    }


def per_label_report(y_true, y_pred, label_names, output_dict=False):
    return classification_report(y_true, y_pred, target_names=label_names,
                                 zero_division=0, output_dict=output_dict)


def per_label_f1(y_true, y_pred, label_names):
    scores = f1_score(y_true, y_pred, average=None, zero_division=0)
    return {name: float(s) for name, s in zip(label_names, scores)}


def tune_thresholds(y_true, probs, grid=None):
    """Pick one threshold per label that maximises that label's F1.
    Tune on the VALIDATION set only, then apply the thresholds to test."""
    grid = DEFAULT_GRID if grid is None else grid
    y_true, probs = np.asarray(y_true), np.asarray(probs)
    thresholds = np.full(probs.shape[1], 0.5)
    for j in range(probs.shape[1]):
        best_f1, best_t = -1.0, 0.5
        for t in grid:
            f = f1_score(y_true[:, j], (probs[:, j] >= t).astype(int), zero_division=0)
            if f > best_f1:
                best_f1, best_t = f, float(t)
        thresholds[j] = best_t
    return thresholds


def apply_thresholds(probs, thresholds):
    return (np.asarray(probs) >= np.asarray(thresholds)).astype(int)


def evaluate_probs(y_val, p_val, y_test, p_test, label_names=None):
    """Standard protocol: tune thresholds on validation, report on test.

    Returns (metrics, thresholds). `metrics` holds the tuned test scores, the
    fixed-0.5 test scores (prefixed `t05_`) and, if label names are given,
    per-label test F1 (prefixed `f1_`).
    """
    th = tune_thresholds(y_val, p_val)
    metrics = evaluate(y_test, apply_thresholds(p_test, th))
    fixed = evaluate(y_test, apply_thresholds(p_test, np.full(p_test.shape[1], 0.5)))
    metrics.update({f"t05_{k}": v for k, v in fixed.items()})
    if label_names is not None:
        f1s = per_label_f1(y_test, apply_thresholds(p_test, th), label_names)
        metrics.update({f"f1_{slug(k)}": v for k, v in f1s.items()})
    return metrics, th


def slug(name):
    return "".join(c if c.isalnum() else "_" for c in name.lower()).strip("_")
