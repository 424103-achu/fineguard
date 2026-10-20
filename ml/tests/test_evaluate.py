import numpy as np
import pytest

from src.data import to_multi_hot
from src.evaluate import (add_none_column, apply_thresholds, evaluate, evaluate_probs,
                          tune_thresholds)


def test_to_multi_hot():
    y = to_multi_hot([[0, 3], [], [5]], 8)
    assert y.shape == (3, 8)
    assert y[0].tolist() == [1, 0, 0, 1, 0, 0, 0, 0]
    assert y[1].sum() == 0


def test_none_column():
    y = np.array([[1, 0], [0, 0]])
    assert add_none_column(y).tolist() == [[1, 0, 0], [0, 0, 1]]


def test_perfect_prediction():
    y = to_multi_hot([[0], [], [1, 2], []], 3)
    m = evaluate(y, y)
    assert m["micro_f1_8"] == 1.0 and m["macro_f1_8"] == 1.0
    assert m["unfair_recall"] == 1.0 and m["micro_f1_lexglue"] == 1.0


def test_all_fair_prediction():
    y = to_multi_hot([[0], [], [1], []], 2)
    m = evaluate(y, np.zeros_like(y))
    assert m["micro_f1_8"] == 0.0 and m["unfair_recall"] == 0.0
    assert m["micro_f1_lexglue"] > 0   # the 'none' column rewards predicting fair


def test_threshold_tuning_finds_separating_threshold():
    y = np.array([[1], [1], [0], [0]])
    p = np.array([[0.3], [0.25], [0.1], [0.05]])
    th = tune_thresholds(y, p)
    assert 0.1 < th[0] <= 0.25
    assert apply_thresholds(p, th).ravel().tolist() == [1, 1, 0, 0]


def test_evaluate_probs_reports_tuned_and_fixed():
    y = np.array([[1, 0], [0, 1], [0, 0], [1, 1]])
    p = np.array([[0.3, 0.1], [0.1, 0.4], [0.05, 0.05], [0.35, 0.45]])
    m, th = evaluate_probs(y, p, y, p, ["a b", "c"])
    assert m["macro_f1_8"] == pytest.approx(1.0)
    assert m["t05_macro_f1_8"] == 0.0
    assert "f1_a_b" in m and len(th) == 2
