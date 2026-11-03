"""Combine several runs (e.g. 3 seeds) into a mean ± std table.

    python -m sad.summarize results/*/summary.json
"""

from __future__ import annotations

import json
import sys
from pathlib import Path

import pandas as pd


def main(argv=None):
    paths = [Path(p) for p in (argv if argv is not None else sys.argv[1:])]
    if not paths:
        paths = sorted(Path("results").glob("*/summary.json"))
    rows = [json.loads(p.read_text()) for p in paths]
    if not rows:
        print("no summary.json files found")
        return
    df = pd.DataFrame(rows)
    metrics = ["precision", "recall", "f1", "f0.5"]
    g = df.groupby(["method", "spacecraft"])
    table = g[metrics].agg(["mean", "std"])
    out = pd.DataFrame(index=table.index)
    for m in metrics:
        out[m] = [f"{a:.3f} ± {0 if pd.isna(b) else b:.3f}" for a, b in table[m].itertuples(index=False)]
    out["n_seeds"] = g.size()
    print(out.to_string())
    return out


if __name__ == "__main__":
    main()
