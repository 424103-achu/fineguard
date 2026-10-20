"""Phase 3: TF-IDF + Logistic Regression / linear SVM baselines.

Run from ml/:  python -m src.train_classical [--ngram 1 2] [--no-balanced]
"""
import argparse
from pathlib import Path

import joblib
from sklearn.calibration import CalibratedClassifierCV
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.linear_model import LogisticRegression
from sklearn.multiclass import OneVsRestClassifier
from sklearn.svm import LinearSVC

from src.data import load_all_splits
from src.evaluate import evaluate_probs
from src.tracking import log_results, mlflow_run, print_metrics


def build_models(c_logreg=4.0, c_svm=0.5, balanced=True):
    cw = "balanced" if balanced else None
    return {
        "tfidf_logreg": OneVsRestClassifier(
            LogisticRegression(max_iter=2000, class_weight=cw, C=c_logreg)),
        # LinearSVC has no predict_proba; calibration gives probabilities
        # so thresholds can be tuned the same way as for every other model
        "tfidf_svm": OneVsRestClassifier(
            CalibratedClassifierCV(LinearSVC(class_weight=cw, C=c_svm), cv=3)),
    }


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--ngram", type=int, nargs=2, default=(1, 2))
    ap.add_argument("--min-df", type=int, default=2)
    ap.add_argument("--c-logreg", type=float, default=4.0)
    ap.add_argument("--c-svm", type=float, default=0.5)
    ap.add_argument("--no-balanced", action="store_true")
    ap.add_argument("--models", nargs="+", default=["tfidf_logreg", "tfidf_svm"])
    ap.add_argument("--out-dir", default="models")
    ap.add_argument("--no-mlflow", action="store_true")
    args = ap.parse_args()

    splits, labels = load_all_splits()
    (Xtr_txt, Ytr), (Xva_txt, Yva), (Xte_txt, Yte) = (
        splits["train"], splits["validation"], splits["test"])

    tfidf = TfidfVectorizer(ngram_range=tuple(args.ngram), min_df=args.min_df,
                            sublinear_tf=True, lowercase=True)
    Xtr = tfidf.fit_transform(Xtr_txt)   # fit on train only
    Xva, Xte = tfidf.transform(Xva_txt), tfidf.transform(Xte_txt)

    Path(args.out_dir).mkdir(parents=True, exist_ok=True)
    models = build_models(args.c_logreg, args.c_svm, not args.no_balanced)
    for name in args.models:
        clf = models[name]
        params = {"model": name, "ngram": f"{args.ngram[0]}-{args.ngram[1]}",
                  "min_df": args.min_df, "balanced": not args.no_balanced,
                  "C": args.c_logreg if name == "tfidf_logreg" else args.c_svm}
        clf.fit(Xtr, Ytr)
        metrics, th = evaluate_probs(Yva, clf.predict_proba(Xva),
                                     Yte, clf.predict_proba(Xte), labels)
        print_metrics(name, metrics)
        joblib.dump({"vectorizer": tfidf, "model": clf, "thresholds": th, "labels": labels},
                    Path(args.out_dir) / f"{name}.joblib")
        if not args.no_mlflow:
            run_name = (f"{name}-ngram{args.ngram[0]}{args.ngram[1]}-C{params['C']:g}"
                        + ("" if params["balanced"] else "-unbalanced"))
            with mlflow_run(run_name, params):
                log_results(metrics, th, labels)


if __name__ == "__main__":
    main()
