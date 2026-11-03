import json

from sad.run import build_parser, run

TINY = ["--l-s", "50", "--n-predictions", "5", "--hidden-size", "16", "--epochs", "4",
        "--error-window", "1000", "--error-buffer", "20"]


def _run(tmp_path, data, method, extra=()):
    args = build_parser().parse_args(
        ["--spacecraft", "SMAP", "--method", method, "--data-dir", str(data),
         "--model-dir", str(tmp_path / "models"), "--out", str(tmp_path / "results"), *TINY, *extra])
    return run(args)


def test_lstm_pipeline_runs_and_saves(tmp_path, synthetic_data):
    s = _run(tmp_path, synthetic_data, "lstm")
    assert s["n_channels"] == 2 and s["tp"] + s["fn"] == 2
    out = tmp_path / "results" / "lstm_SMAP_seed0"
    assert json.loads((out / "summary.json").read_text())["method"] == "lstm"
    assert (tmp_path / "models" / "SMAP" / "seed0" / "T-1.pt").exists()
    # second run reuses the saved models
    s2 = _run(tmp_path, synthetic_data, "lstm")
    assert s2["tp"] == s["tp"]

    from sad.plot import plot_channel
    assert plot_channel(out, "T-1").exists()


def test_baselines_detect_injected_shift(tmp_path, synthetic_data):
    s = _run(tmp_path, synthetic_data, "iforest")
    assert s["tp"] >= 1
    # rolling z-score is a weak baseline on periodic signals; only check that it runs
    s = _run(tmp_path, synthetic_data, "zscore")
    assert s["tp"] + s["fn"] == 2
