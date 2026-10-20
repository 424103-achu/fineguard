"""Thin MLflow wrapper. Every reported number must come from a logged run.

Runs go to a SQLite store in ml/mlruns/ (mlflow.db + artifacts/), unless
MLFLOW_TRACKING_URI is set. Browse with:
  mlflow ui --backend-store-uri sqlite:///mlruns/mlflow.db   (from ml/)
"""
import json
import os
from contextlib import contextmanager
from pathlib import Path

EXPERIMENT = "fine-print-guard"
ML_DIR = Path(__file__).resolve().parents[1]


def configure(tracking_uri=None):
    """Point MLflow at the project store and make sure the experiment exists."""
    import mlflow

    store = ML_DIR / "mlruns"
    store.mkdir(exist_ok=True)
    mlflow.set_tracking_uri(tracking_uri or os.environ.get("MLFLOW_TRACKING_URI")
                            or f"sqlite:///{store / 'mlflow.db'}")
    if mlflow.get_experiment_by_name(EXPERIMENT) is None:
        mlflow.create_experiment(EXPERIMENT, artifact_location=(store / "artifacts").as_uri())
    return mlflow.set_experiment(EXPERIMENT)


@contextmanager
def mlflow_run(run_name, params, tracking_uri=None):
    import mlflow

    configure(tracking_uri)
    with mlflow.start_run(run_name=run_name) as run:
        mlflow.log_params({k: str(v) for k, v in params.items()})
        yield run


def log_results(metrics, thresholds, labels, extra_dicts=None):
    import mlflow

    mlflow.log_metrics(metrics)
    mlflow.log_dict({"labels": labels, "thresholds": [float(t) for t in thresholds]},
                    "thresholds.json")
    for name, d in (extra_dicts or {}).items():
        mlflow.log_dict(d, name)


def print_metrics(name, metrics):
    main = {k: round(v, 4) for k, v in metrics.items()
            if not k.startswith(("f1_", "t05_"))}
    print(f"{name}: {json.dumps(main)}")
