"""Loading the NASA SMAP/MSL telemetry and turning it into LSTM windows.

Layout expected under the data directory (created by scripts/download_data.py):

    data/
      labeled_anomalies.csv
      train/<chan_id>.npy     # shape (T_train, n_features), mostly nominal
      test/<chan_id>.npy      # shape (T_test,  n_features), contains labeled anomalies

Column 0 is the telemetry value (already scaled to [-1, 1] by NASA). The other columns are
one-hot encoded commands sent to the spacecraft.
"""

from __future__ import annotations

import ast
import os
from pathlib import Path

import numpy as np
import pandas as pd

PROJECT_ROOT = Path(__file__).resolve().parents[2]


def data_dir(path: str | os.PathLike | None = None) -> Path:
    if path is not None:
        return Path(path)
    return Path(os.environ.get("SAD_DATA_DIR", PROJECT_ROOT / "data"))


def _parse_list(s: str) -> list:
    # class column looks like "[point, contextual]" (unquoted), sequences are valid literals
    s = s.strip()
    try:
        return list(ast.literal_eval(s))
    except (ValueError, SyntaxError):
        return [x.strip() for x in s.strip("[]").split(",") if x.strip()]


def load_labels(path: str | os.PathLike | None = None) -> pd.DataFrame:
    """One row per channel with parsed `anomaly_sequences` and `class` lists.

    The CSV lists channel P-2 twice; the rows are merged so each channel appears once.
    """
    p = Path(path) if path is not None else data_dir() / "labeled_anomalies.csv"
    df = pd.read_csv(p)
    df["anomaly_sequences"] = df["anomaly_sequences"].map(_parse_list)
    df["class"] = df["class"].map(_parse_list)
    df = (
        df.groupby(["chan_id", "spacecraft"], sort=False)
        .agg({"anomaly_sequences": "sum", "class": "sum", "num_values": "first"})
        .reset_index()
    )
    return df


def channels(spacecraft: str, labels: pd.DataFrame | None = None) -> list[str]:
    labels = load_labels() if labels is None else labels
    return labels.loc[labels.spacecraft == spacecraft.upper(), "chan_id"].tolist()


def load_channel(chan_id: str, root: str | os.PathLike | None = None) -> tuple[np.ndarray, np.ndarray]:
    root = data_dir(root)
    train = np.load(root / "train" / f"{chan_id}.npy").astype(np.float32)
    test = np.load(root / "test" / f"{chan_id}.npy").astype(np.float32)
    return train, test


def make_windows(arr: np.ndarray, l_s: int, n_predictions: int) -> tuple[np.ndarray, np.ndarray]:
    """Sliding windows: X[i] = arr[i:i+l_s], y[i] = arr[i+l_s : i+l_s+n_pred, 0]."""
    n = len(arr) - l_s - n_predictions + 1
    if n <= 0:
        raise ValueError(f"series of length {len(arr)} too short for l_s={l_s}, h={n_predictions}")
    idx = np.arange(l_s)[None, :] + np.arange(n)[:, None]
    X = arr[idx]
    yidx = np.arange(n_predictions)[None, :] + np.arange(n)[:, None] + l_s
    y = arr[yidx, 0]
    return X.astype(np.float32), y.astype(np.float32)


def make_input_windows(arr: np.ndarray, l_s: int) -> np.ndarray:
    """All windows that end before the last timestep (used at test time, no targets needed)."""
    n = len(arr) - l_s
    idx = np.arange(l_s)[None, :] + np.arange(n)[:, None]
    return arr[idx].astype(np.float32)


def sequences_to_mask(sequences: list, length: int) -> np.ndarray:
    mask = np.zeros(length, dtype=bool)
    for start, end in sequences:
        mask[start : end + 1] = True
    return mask
