"""Build the results table (markdown) straight from logged MLflow runs.

Run from ml/:  python -m src.results_table [--out results.md]

Seeded runs of the same model (run names ending in -s<seed>) are grouped and shown as
mean ± std. If a run name was logged more than once, only the newest run counts.
Only numbers from logged runs appear; nothing is estimated.
"""
import argparse
import re
from collections import defaultdict

import numpy as np

from src.tracking import configure

COLUMNS = [("micro_f1_8", "μ-F1 (8)"), ("macro_f1_8", "m-F1 (8)"),
           ("micro_f1_lexglue", "μ-F1 (LexGLUE)"), ("macro_f1_lexglue", "m-F1 (LexGLUE)"),
           ("unfair_recall", "Unfair recall"), ("t05_macro_f1_8", "m-F1 (8) @0.5")]


def fetch_runs():
    import mlflow

    exp = configure()
    return mlflow.search_runs([exp.experiment_id],
                              order_by=["attributes.start_time DESC"]).to_dict("records")


def build_table(runs):
    groups, seen = defaultdict(list), set()
    for r in runs:   # newest first, so a re-run replaces the older run with the same name
        name = r.get("tags.mlflow.runName") or r["run_id"]
        if name in seen or name.endswith("-summary") or name == "score-validation":
            continue
        seen.add(name)
        if f"metrics.{COLUMNS[0][0]}" not in r or np.isnan(r[f"metrics.{COLUMNS[0][0]}"]):
            continue
        groups[re.sub(r"-s\d+$", "", name)].append(r)

    lines = ["| Model | runs | " + " | ".join(c[1] for c in COLUMNS) + " |",
             "|---|---|" + "---|" * len(COLUMNS)]
    for name in sorted(groups):
        cells = []
        for key, _ in COLUMNS:
            vals = [r.get(f"metrics.{key}") for r in groups[name]]
            vals = [100 * v for v in vals if v is not None and not np.isnan(v)]
            if not vals:
                cells.append("")
            elif len(vals) == 1:
                cells.append(f"{vals[0]:.1f}")
            else:
                cells.append(f"{np.mean(vals):.1f} ± {np.std(vals, ddof=1):.1f}")
        lines.append(f"| {name} | {len(groups[name])} | " + " | ".join(cells) + " |")
    return "\n".join(lines)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--out", default=None)
    args = ap.parse_args()
    table = build_table(fetch_runs())
    print(table)
    if args.out:
        with open(args.out, "w") as f:
            f.write(table + "\n")


if __name__ == "__main__":
    main()
