import numpy as np

from sad.config import Config
from sad.thresholding import detect, find_epsilon, group_sequences, prune


def test_group_sequences():
    assert group_sequences(np.array([1, 2, 3, 7, 8, 12])) == [(1, 3), (7, 8), (12, 12)]
    assert group_sequences(np.array([], dtype=int)) == []


def test_dynamic_threshold_finds_spike():
    rng = np.random.default_rng(1)
    e = np.abs(rng.normal(0, 0.05, 3000))
    e[2000:2040] += 1.0
    cfg = Config(error_buffer=5, error_window=3000)
    eps = find_epsilon(e, cfg)
    assert e[2000:2040].min() > eps > np.quantile(e[:1900], 0.99)
    seqs, _ = detect(e, cfg)
    assert len(seqs) == 1 and seqs[0][0] <= 2000 and seqs[0][1] >= 2039


def test_pruning_drops_small_bumps():
    e = np.full(1000, 0.1)
    e[100:110] = 1.0      # clear anomaly
    e[500:505] = 0.105    # barely above the nominal maximum
    seqs = prune(e, [(100, 109), (500, 504)], p=0.13)
    assert seqs == [(100, 109)]
