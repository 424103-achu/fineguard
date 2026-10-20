"""Phase 9 validation: risk score vs ToS;DR grades on your collected documents.

Run from ml/:
  python -m src.validate_score --model models/legal-bert-unfair \
      [--grades data/tosdr_grades.csv] [--docs data/tos_docs] [--write-weights]

Reports Spearman rho (with bootstrap 95% CI) for equal weights (no fitting) and for
fitted non-negative weights (leave-one-out). --write-weights saves the v2 weights to
<model>/score_weights.json (or next to a .joblib model), which the service then loads.
"""
import argparse
import csv
import json
import time
from pathlib import Path

import numpy as np
from scipy.stats import spearmanr

from src.pipeline import split_sentences
from src.predictor import load_predictor
from src.scoring import (TOSDR_TO_NUM, bootstrap_ci, doc_features, fit_weights, risk_score,
                         to_grade, validate_against_tosdr)
from src.tracking import mlflow_run


def weights_path(model_path):
    p = Path(model_path)
    return p / "score_weights.json" if p.is_dir() else p.with_suffix(".score_weights.json")


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--model", required=True)
    ap.add_argument("--grades", default="data/tosdr_grades.csv")
    ap.add_argument("--docs", default="data/tos_docs")
    ap.add_argument("--out", default="data/score_validation.csv")
    ap.add_argument("--write-weights", action="store_true")
    ap.add_argument("--no-mlflow", action="store_true")
    args = ap.parse_args()

    with open(args.grades, newline="") as f:
        rows = [r for r in csv.DictReader(f) if r.get("file")]
    if len(rows) < 5:
        raise SystemExit(f"Need at least 5 graded documents in {args.grades}, found {len(rows)}")

    predictor = load_predictor(args.model)
    labels = predictor.labels
    counts, grades, times, n_sents = [], [], [], []
    for r in rows:
        text = (Path(args.docs) / r["file"]).read_text(encoding="utf-8")
        t0 = time.perf_counter()
        sents = split_sentences(text)
        _, preds = predictor.predict(sents)
        times.append((time.perf_counter() - t0) * 1000)
        counts.append(preds.sum(axis=0) if len(sents) else np.zeros(len(labels)))
        n_sents.append(len(sents))
        grades.append(TOSDR_TO_NUM[r["tosdr_grade"].strip().upper()])
    counts, g = np.array(counts), np.array(grades)
    F = doc_features(counts)

    equal = np.array([risk_score(c, np.ones(len(labels))) for c in counts])
    rho_eq, p_eq = spearmanr(equal, g)
    ci_eq = bootstrap_ci(equal, g)
    rho_fit, p_fit, loo_pred = validate_against_tosdr(F, g)
    ci_fit = bootstrap_ci(loo_pred, g)
    w = fit_weights(F, g)

    print(f"documents: {len(rows)}   mean ms/doc: {np.mean(times):.0f}")
    print(f"equal weights : rho={rho_eq:.3f} (p={p_eq:.3g}) 95% CI [{ci_eq[0]:.3f}, {ci_eq[1]:.3f}]")
    print(f"fitted (LOO)  : rho={rho_fit:.3f} (p={p_fit:.3g}) 95% CI [{ci_fit[0]:.3f}, {ci_fit[1]:.3f}]")
    print("fitted weights:", dict(zip(labels, np.round(w, 3).tolist())))

    with open(args.out, "w", newline="") as f:
        wr = csv.writer(f)
        wr.writerow(["service", "tosdr_grade", "n_sentences", "score_equal", "grade_equal",
                     "loo_pred", "ms"] + labels)
        for r, ns, s, lp, ms, c in zip(rows, n_sents, equal, loo_pred, times, counts):
            wr.writerow([r["service"], r["tosdr_grade"], ns, round(s, 1), to_grade(s),
                         round(float(lp), 3), round(ms)] + c.astype(int).tolist())
    print(f"per-document table written to {args.out}")

    if args.write_weights:
        weights_path(args.model).write_text(json.dumps(
            {"labels": labels, "weights": w.tolist(), "n_docs": len(rows),
             "loo_spearman": float(rho_fit)}, indent=2))
        print(f"weights written to {weights_path(args.model)}")

    if not args.no_mlflow:
        with mlflow_run("score-validation", {"model": args.model, "n_docs": len(rows)}):
            import mlflow
            mlflow.log_metrics({"spearman_equal": rho_eq, "spearman_equal_ci_low": ci_eq[0],
                                "spearman_equal_ci_high": ci_eq[1], "spearman_loo": rho_fit,
                                "spearman_loo_ci_low": ci_fit[0], "spearman_loo_ci_high": ci_fit[1],
                                "ms_per_doc": float(np.mean(times))})
            mlflow.log_dict({"labels": labels, "weights": w.tolist()}, "score_weights.json")
            mlflow.log_artifact(args.out)


if __name__ == "__main__":
    main()
