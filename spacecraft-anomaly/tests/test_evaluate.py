from sad.evaluate import aggregate, evaluate_channel


def test_event_level_counts():
    r = evaluate_channel([[10, 20], [50, 60]], [(15, 18), (100, 110)], ["point", "contextual"])
    assert (r["tp"], r["fp"], r["fn"]) == (1, 1, 1)
    assert r["by_class"]["point"]["tp"] == 1 and r["by_class"]["contextual"]["tp"] == 0


def test_aggregate_metrics():
    s = aggregate([{"tp": 3, "fp": 1, "fn": 1}, {"tp": 1, "fp": 0, "fn": 1}])
    assert s["precision"] == 0.8 and abs(s["recall"] - 4 / 6) < 1e-9
    assert 0 < s["f1"] < 1 and 0 < s["f0.5"] < 1
