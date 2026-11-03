"""Plot one channel from a finished run: signal, forecast, smoothed error, threshold,
labeled anomalies (red) and detected anomalies (blue).

    python -m sad.plot results/lstm_SMAP_seed0 P-1
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt  # noqa: E402
import numpy as np  # noqa: E402
import pandas as pd  # noqa: E402


def plot_channel(run_dir: str | Path, chan: str, out: str | Path | None = None) -> Path:
    run_dir = Path(run_dir)
    d = np.load(run_dir / "channels" / f"{chan}.npz")
    row = pd.read_csv(run_dir / "per_channel.csv").set_index("chan_id").loc[chan]
    true_seqs, pred_seqs = json.loads(row["true_seqs"]), json.loads(row["pred_seqs"])

    fig, (a1, a2) = plt.subplots(2, 1, figsize=(14, 6), sharex=True)
    a1.plot(d["signal"], lw=0.7, color="black", label="telemetry")
    if "y_hat" in d:
        a1.plot(d["y_hat"], lw=0.7, color="tab:orange", label="LSTM forecast")
    a2.plot(d["scores"], lw=0.7, color="tab:purple", label="anomaly score")
    a2.plot(d["threshold"], lw=1, ls="--", color="gray", label="threshold")
    for ax in (a1, a2):
        for s, e in true_seqs:
            ax.axvspan(s, e, color="tab:red", alpha=0.2)
        for s, e in pred_seqs:
            ax.axvspan(s, e, color="tab:blue", alpha=0.15)
    a1.set_title(f"{chan}  (red = labeled anomaly, blue = detected)   "
                 f"tp={row['tp']} fp={row['fp']} fn={row['fn']}")
    a1.legend(loc="upper left")
    a2.legend(loc="upper left")
    a2.set_xlabel("test timestep")
    fig.tight_layout()
    out = Path(out) if out else run_dir / "plots" / f"{chan}.png"
    out.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(out, dpi=120)
    plt.close(fig)
    return out


def main(argv=None):
    p = argparse.ArgumentParser()
    p.add_argument("run_dir")
    p.add_argument("channels", nargs="*", help="default: all channels in the run")
    a = p.parse_args(argv)
    chans = a.channels or [f.stem for f in sorted((Path(a.run_dir) / "channels").glob("*.npz"))]
    for c in chans:
        print(plot_channel(a.run_dir, c))


if __name__ == "__main__":
    main()
