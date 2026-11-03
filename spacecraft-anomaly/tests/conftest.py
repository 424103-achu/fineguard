import numpy as np
import pandas as pd
import pytest


def _series(n, rng, anomaly=None):
    t = np.arange(n)
    x = 0.6 * np.sin(2 * np.pi * t / 100) + 0.02 * rng.standard_normal(n)
    cmds = np.zeros((n, 2), dtype=np.float32)
    cmds[::250, 0] = 1
    if anomaly:
        s, e = anomaly
        x[s:e] += 0.9
    return np.column_stack([np.clip(x, -1, 1), cmds]).astype(np.float32)


@pytest.fixture
def synthetic_data(tmp_path):
    """Two fake channels in the NASA layout, each with one injected anomaly."""
    rng = np.random.default_rng(0)
    (tmp_path / "train").mkdir()
    (tmp_path / "test").mkdir()
    rows = []
    for chan, anom in [("T-1", (1500, 1530)), ("T-2", (1200, 1260))]:
        np.save(tmp_path / "train" / f"{chan}.npy", _series(2000, rng))
        np.save(tmp_path / "test" / f"{chan}.npy", _series(2500, rng, anom))
        rows.append({"chan_id": chan, "spacecraft": "SMAP",
                     "anomaly_sequences": str([list(anom)]), "class": "[point]", "num_values": 2500})
    pd.DataFrame(rows).to_csv(tmp_path / "labeled_anomalies.csv", index=False)
    return tmp_path
