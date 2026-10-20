"""Phase 7: error analysis helpers for the best model."""
import numpy as np
import pandas as pd


def error_table(texts, y_true, y_pred, label_names, k=5):
    """Up to k false negatives and k false positives per category."""
    rows = []
    for j, name in enumerate(label_names):
        fn = np.where((y_true[:, j] == 1) & (y_pred[:, j] == 0))[0][:k]
        fp = np.where((y_true[:, j] == 0) & (y_pred[:, j] == 1))[0][:k]
        rows += [{"label": name, "type": "FN", "text": texts[i]} for i in fn]
        rows += [{"label": name, "type": "FP", "text": texts[i]} for i in fp]
    return pd.DataFrame(rows)


def confusion_pairs(y_true, y_pred, label_names):
    """C[a, b] = how often true category a was predicted as b while b was not true."""
    n = len(label_names)
    C = np.zeros((n, n), dtype=int)
    for a in range(n):
        for b in range(n):
            if a != b:
                C[a, b] = int(((y_true[:, a] == 1) & (y_pred[:, b] == 1) & (y_true[:, b] == 0)).sum())
    return pd.DataFrame(C, index=label_names, columns=label_names)


def co_occurrence(y, label_names):
    y = np.asarray(y)
    return pd.DataFrame(y.T @ y, index=label_names, columns=label_names)


def f1_by_length(texts, y_true, y_pred, bins=(0, 15, 30, 50, 10_000)):
    from sklearn.metrics import f1_score

    lengths = np.array([len(t.split()) for t in texts])
    rows = []
    for lo, hi in zip(bins[:-1], bins[1:]):
        m = (lengths >= lo) & (lengths < hi)
        if m.sum() == 0:
            continue
        rows.append({"words": f"{lo}-{hi - 1}" if hi < 10_000 else f"{lo}+",
                     "n": int(m.sum()), "n_unfair": int(y_true[m].any(axis=1).sum()),
                     "micro_f1_8": f1_score(y_true[m], y_pred[m], average="micro", zero_division=0)})
    return pd.DataFrame(rows)
