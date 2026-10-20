import numpy as np
import pytest

from src.scoring import (bootstrap_ci, doc_features, fit_weights, risk_score, to_grade,
                         validate_against_tosdr)


def test_no_flags_is_zero():
    assert risk_score(np.zeros(8), np.ones(8)) == 0.0


def test_more_flags_never_lower_score():
    w = np.ones(8)
    assert risk_score([1, 0, 0, 0, 0, 0, 0, 0], w) <= risk_score([2, 0, 0, 0, 0, 0, 0, 0], w)


def test_feature_formula():
    assert doc_features([0, 1])[0] == 0.0
    assert doc_features([0, 1])[1] == pytest.approx(1 - np.exp(-1))


def test_score_bounded_and_equal_weight_value():
    s = risk_score([1, 0, 0, 0, 0, 0, 0, 0], np.ones(8))
    assert s == pytest.approx(100 * (1 - np.exp(-1)) / 8)
    assert 0 <= risk_score([50] * 8, np.ones(8)) <= 100


def test_zero_weights_fall_back_to_equal():
    assert risk_score([1] * 8, np.zeros(8)) == risk_score([1] * 8, np.ones(8))


def test_grade_bands():
    assert to_grade(0) == "A"
    assert to_grade(19.9) == "A"
    assert to_grade(20) == "B"
    assert to_grade(59.9) == "C"
    assert to_grade(79.9) == "D"
    assert to_grade(80) == "E"
    assert to_grade(100) == "E"


def test_fit_weights_non_negative_and_normalised():
    rng = np.random.default_rng(0)
    F = doc_features(rng.integers(0, 4, size=(30, 8)))
    g = np.clip(np.round(1 + 4 * F[:, 0] + rng.normal(0, 0.3, 30)), 1, 5)
    w = fit_weights(F, g)
    assert (w >= 0).all()
    assert w.sum() == pytest.approx(1.0)
    assert w.argmax() == 0


def test_validation_and_ci():
    rng = np.random.default_rng(1)
    F = doc_features(rng.integers(0, 4, size=(25, 8)))
    g = np.clip(np.round(1 + 4 * F.mean(axis=1) + rng.normal(0, 0.3, 25)), 1, 5)
    rho, p, pred = validate_against_tosdr(F, g)
    assert len(pred) == 25 and -1 <= rho <= 1
    lo, hi = bootstrap_ci(pred, g, n_boot=200)
    assert lo <= hi
