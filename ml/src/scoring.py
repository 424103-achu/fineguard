"""Phase 9: document risk score, weight fitting against ToS;DR, and statistics.

Per category: f = 1 - e^(-n), n = number of flagged sentences.
Score = 100 * sum(w * f) / sum(w). Grades: A < 20, B < 40, C < 60, D < 80, E >= 80.
"""
import numpy as np

GRADE_BANDS = [(20, "A"), (40, "B"), (60, "C"), (80, "D")]
TOSDR_TO_NUM = {"A": 1, "B": 2, "C": 3, "D": 4, "E": 5}


def doc_features(counts):
    """counts: flagged-sentence count per category for one document (or a 2-D array of docs)."""
    return 1.0 - np.exp(-np.asarray(counts, dtype=float))


def risk_score(counts, weights):
    w = np.asarray(weights, dtype=float)
    if w.sum() <= 0:
        w = np.ones_like(w)
    return float(100.0 * (w * doc_features(counts)).sum() / w.sum())


def to_grade(score):
    for upper, grade in GRADE_BANDS:
        if score < upper:
            return grade
    return "E"


def fit_weights(F, grades_num):
    """F: (n_docs, n_labels) features; grades_num: ToS;DR grade as 1..5.
    Non-negative weights, so no category can lower the risk. Returned normalised to sum 1."""
    from sklearn.linear_model import LinearRegression

    reg = LinearRegression(positive=True).fit(F, grades_num)
    w = reg.coef_
    return w / w.sum() if w.sum() > 0 else np.ones_like(w) / len(w)


def loo_predictions(F, grades_num):
    """Leave-one-out predictions, so the score is never judged on a document it was fitted on."""
    from sklearn.linear_model import LinearRegression
    from sklearn.model_selection import LeaveOneOut, cross_val_predict

    return cross_val_predict(LinearRegression(positive=True), F, grades_num, cv=LeaveOneOut())


def validate_against_tosdr(F, grades_num):
    from scipy.stats import spearmanr

    pred = loo_predictions(F, grades_num)
    rho, p = spearmanr(pred, grades_num)
    return rho, p, pred


def bootstrap_ci(x, y, n_boot=2000, seed=0):
    """95% confidence interval for Spearman's rho."""
    from scipy.stats import spearmanr

    rng = np.random.default_rng(seed)
    x, y = np.asarray(x), np.asarray(y)
    vals = []
    for _ in range(n_boot):
        idx = rng.integers(0, len(x), len(x))
        r, _ = spearmanr(x[idx], y[idx])
        if not np.isnan(r):
            vals.append(r)
    return np.percentile(vals, [2.5, 97.5])

