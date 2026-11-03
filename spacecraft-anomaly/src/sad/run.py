"""Run a detector over every channel of a spacecraft and report event-level metrics.

Examples
--------
    python -m sad.run --spacecraft SMAP --method lstm --seed 0
    python -m sad.run --spacecraft MSL  --method lstm --channels M-1 M-2 --epochs 5
    python -m sad.run --spacecraft SMAP --method iforest
"""

from __future__ import annotations

import argparse
import json
import time
from dataclasses import fields
from pathlib import Path

import numpy as np
import pandas as pd

from . import baselines
from .config import Config
from .data import PROJECT_ROOT, channels, load_channel, load_labels
from .evaluate import aggregate, evaluate_channel
from .thresholding import detect, smooth_errors, static_threshold
from .train import default_device, load_model, predict_series, save_model, train_channel

METHODS = ["lstm", "lstm_static", "zscore", "iforest"]


def lstm_forecast(chan, train, test, cfg, seed, model_dir, device, retrain, verbose):
    path = Path(model_dir) / f"seed{seed}" / f"{chan}.pt"
    history = None
    if path.exists() and not retrain:
        model = load_model(path, cfg, device)
    else:
        model, history = train_channel(train, cfg, device=device, seed=seed, verbose=verbose)
        save_model(model, path, cfg, train.shape[1], history)
    y_hat = predict_series(model, test, cfg, device)
    return y_hat, history


def run(args) -> dict:
    cfg = Config(**{f.name: getattr(args, f.name) for f in fields(Config)
                    if getattr(args, f.name, None) is not None})
    labels = load_labels(Path(args.data_dir) / "labeled_anomalies.csv" if args.data_dir else None)
    chans = args.channels or channels(args.spacecraft, labels)
    labels = labels.set_index("chan_id")
    device = default_device()
    run_name = args.run_name or f"{args.method}_{args.spacecraft.upper()}_seed{args.seed}"
    out = Path(args.out) / run_name
    (out / "channels").mkdir(parents=True, exist_ok=True)
    model_dir = Path(args.model_dir) / args.spacecraft.upper()
    print(f"[{run_name}] {len(chans)} channels on {device}")

    rows = []
    t0 = time.time()
    for k, chan in enumerate(chans, 1):
        train, test = load_channel(chan, args.data_dir)
        true_seqs = labels.loc[chan, "anomaly_sequences"]
        classes = labels.loc[chan, "class"]
        extra = {}
        if args.method in ("lstm", "lstm_static"):
            y_hat, hist = lstm_forecast(chan, train, test, cfg, args.seed, model_dir, device,
                                        args.retrain, args.verbose)
            y_true = test[cfg.l_s:, 0]
            e_s = smooth_errors(y_true, y_hat, cfg)
            if args.method == "lstm":
                seqs, thr = detect(e_s, cfg, y_true=y_true)
            else:
                t = float(e_s.mean() + 3 * e_s.std())
                seqs, thr = static_threshold(e_s, t, cfg.error_buffer), np.full(len(e_s), t)
            pred_seqs = [(s + cfg.l_s, e + cfg.l_s) for s, e in seqs]  # back to test indices
            scores = np.r_[np.full(cfg.l_s, np.nan), e_s]
            thr = np.r_[np.full(cfg.l_s, np.nan), thr]
            extra["y_hat"] = np.r_[np.full(cfg.l_s, np.nan), y_hat]
            if hist:
                extra_meta = {"epochs_run": hist["epochs_run"], "best_val_loss": hist["best_val_loss"]}
            else:
                extra_meta = {}
        else:
            fn = baselines.zscore_scores if args.method == "zscore" else baselines.iforest_scores
            kwargs = {"seed": args.seed} if args.method == "iforest" else {}
            scores, t = fn(train, test, **kwargs)
            pred_seqs = static_threshold(scores, t, cfg.error_buffer)
            thr = np.full(len(scores), t)
            extra_meta = {}

        res = evaluate_channel(true_seqs, pred_seqs, classes)
        rows.append({"chan_id": chan, **res, "n_pred": len(pred_seqs),
                     "pred_seqs": json.dumps([list(map(int, s)) for s in pred_seqs]),
                     "true_seqs": json.dumps(true_seqs), **extra_meta})
        np.savez_compressed(out / "channels" / f"{chan}.npz", signal=test[:, 0], scores=scores,
                            threshold=thr, **extra)
        print(f"  [{k:2d}/{len(chans)}] {chan:6s} tp={res['tp']} fp={res['fp']} fn={res['fn']}"
              f"  ({time.time() - t0:.0f}s)")

    summary = {"run_name": run_name, "method": args.method, "spacecraft": args.spacecraft.upper(),
               "seed": args.seed, "n_channels": len(chans), "config": cfg.to_dict(),
               **aggregate(rows)}
    pd.DataFrame(rows).drop(columns=["by_class"], errors="ignore").to_csv(out / "per_channel.csv", index=False)
    (out / "summary.json").write_text(json.dumps(summary, indent=2))
    print(json.dumps({k: summary[k] for k in ("tp", "fp", "fn", "precision", "recall", "f1", "f0.5")
                      if k in summary}, indent=2))
    if "recall_by_class" in summary:
        print("recall by class:", summary["recall_by_class"])
    print(f"saved -> {out}")

    if args.mlflow:
        import mlflow
        mlflow.set_experiment("spacecraft-anomaly")
        with mlflow.start_run(run_name=run_name):
            mlflow.log_params({"method": args.method, "spacecraft": summary["spacecraft"],
                               "seed": args.seed, **cfg.to_dict()})
            mlflow.log_metrics({k: summary[k] for k in ("precision", "recall", "f1", "f0.5", "tp", "fp", "fn")})
            mlflow.log_artifact(str(out / "summary.json"))
            mlflow.log_artifact(str(out / "per_channel.csv"))
    return summary


def build_parser() -> argparse.ArgumentParser:
    p = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument("--spacecraft", choices=["SMAP", "MSL", "smap", "msl"], required=True)
    p.add_argument("--method", choices=METHODS, default="lstm")
    p.add_argument("--channels", nargs="*", help="subset of channel ids (default: all)")
    p.add_argument("--seed", type=int, default=0)
    p.add_argument("--data-dir", default=None, help="default: ./data or $SAD_DATA_DIR")
    p.add_argument("--model-dir", default=str(PROJECT_ROOT / "models"))
    p.add_argument("--out", default=str(PROJECT_ROOT / "results"))
    p.add_argument("--run-name", default=None)
    p.add_argument("--retrain", action="store_true", help="ignore saved models and retrain")
    p.add_argument("--mlflow", action="store_true", help="log the run to MLflow")
    p.add_argument("--verbose", action="store_true")
    for f in fields(Config):
        if f.name == "z_range":
            continue
        p.add_argument(f"--{f.name.replace('_', '-')}", dest=f.name, type=type(f.default), default=None)
    return p


def main(argv=None):
    run(build_parser().parse_args(argv))


if __name__ == "__main__":
    main()
