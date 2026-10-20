"""Phase 6: fine-tune BERT / Legal-BERT for multi-label classification.

Run from ml/ (GPU recommended):
  python -m src.train_transformer --model-name nlpaueb/legal-bert-base-uncased \
      --seeds 42 43 44 --export-dir models/legal-bert-unfair
  python -m src.train_transformer --model-name bert-base-uncased --seeds 42 43 44

Each seed is saved to models/<short-name>-s<seed>/. With --export-dir, the seed with the
best *validation* macro-F1 is copied there for the web service (test is never used to pick).
"""
import argparse
import json
import shutil
import time
from pathlib import Path

import numpy as np
import torch
import torch.nn as nn
from sklearn.metrics import f1_score

from src.data import load_unfair_tos, to_multi_hot
from src.evaluate import apply_thresholds, evaluate_probs, tune_thresholds
from src.torch_utils import compute_pos_weight
from src.tracking import mlflow_run, print_metrics

SHORT_NAMES = {"nlpaueb/legal-bert-base-uncased": "legal-bert",
               "bert-base-uncased": "bert",
               "distilbert-base-uncased": "distilbert"}


def sigmoid(x):
    return 1 / (1 + np.exp(-x))


def train_one_seed(model_name, ds, labels, seed, args):
    from transformers import (AutoModelForSequenceClassification, AutoTokenizer,
                              DataCollatorWithPadding, EarlyStoppingCallback, Trainer,
                              TrainingArguments)

    n = len(labels)
    tok = AutoTokenizer.from_pretrained(model_name)

    def preprocess(batch):
        enc = tok(batch["text"], truncation=True, max_length=args.max_len)
        enc["labels"] = [[1.0 if i in labs else 0.0 for i in range(n)] for labs in batch["labels"]]
        return enc

    encoded = ds.map(preprocess, batched=True, remove_columns=ds["train"].column_names)
    base_collator = DataCollatorWithPadding(tok)

    def collate(features):
        batch = base_collator(features)
        batch["labels"] = batch["labels"].float()   # BCE needs float32 targets
        return batch

    model = AutoModelForSequenceClassification.from_pretrained(
        model_name, num_labels=n, problem_type="multi_label_classification",
        id2label=dict(enumerate(labels)), label2id={l: i for i, l in enumerate(labels)})

    class WeightedTrainer(Trainer):
        """BCE with per-label pos_weight to handle class imbalance."""

        def __init__(self, *a, pos_weight=None, **kw):
            super().__init__(*a, **kw)
            self.pos_weight = pos_weight

        def compute_loss(self, model, inputs, return_outputs=False, **kwargs):
            labels_t = inputs.pop("labels")
            outputs = model(**inputs)
            loss = nn.functional.binary_cross_entropy_with_logits(
                outputs.logits, labels_t, pos_weight=self.pos_weight.to(outputs.logits.device))
            return (loss, outputs) if return_outputs else loss

    def compute_metrics(eval_pred):
        logits, y = eval_pred
        preds = (sigmoid(logits) >= 0.5).astype(int)
        return {"micro_f1": f1_score(y, preds, average="micro", zero_division=0),
                "macro_f1": f1_score(y, preds, average="macro", zero_division=0)}

    short = SHORT_NAMES.get(model_name, model_name.split("/")[-1])
    targs = TrainingArguments(
        output_dir=f"runs/{short}-s{seed}",
        eval_strategy="epoch",
        save_strategy="epoch",
        load_best_model_at_end=True,
        metric_for_best_model="macro_f1",
        greater_is_better=True,
        learning_rate=args.lr,
        per_device_train_batch_size=args.batch_size,
        per_device_eval_batch_size=64,
        num_train_epochs=args.epochs,
        weight_decay=0.01,
        warmup_steps=0.1,          # transformers v5: a float < 1 means 10% of steps
        fp16=torch.cuda.is_available(),
        save_total_limit=2,
        seed=seed,
        report_to="none",          # final metrics are logged to MLflow by src.tracking
    )
    Ytr = to_multi_hot(ds["train"]["labels"], n)
    trainer = WeightedTrainer(
        model=model, args=targs,
        train_dataset=encoded["train"], eval_dataset=encoded["validation"],
        data_collator=collate, compute_metrics=compute_metrics,
        callbacks=[EarlyStoppingCallback(early_stopping_patience=2)],
        pos_weight=compute_pos_weight(Ytr),
    )
    t0 = time.time()
    trainer.train()
    train_minutes = (time.time() - t0) / 60

    p_va = sigmoid(trainer.predict(encoded["validation"]).predictions)
    p_te = sigmoid(trainer.predict(encoded["test"]).predictions)
    Yva = to_multi_hot(ds["validation"]["labels"], n)
    Yte = to_multi_hot(ds["test"]["labels"], n)
    metrics, th = evaluate_probs(Yva, p_va, Yte, p_te, labels)
    metrics["train_minutes"] = train_minutes
    val_macro = f1_score(Yva, apply_thresholds(p_va, tune_thresholds(Yva, p_va)),
                         average="macro", zero_division=0)
    metrics["val_macro_f1_tuned"] = float(val_macro)

    out_dir = Path(args.models_dir) / f"{short}-s{seed}"
    trainer.save_model(str(out_dir))
    tok.save_pretrained(str(out_dir))
    (out_dir / "thresholds.json").write_text(json.dumps(
        {"labels": labels, "thresholds": th.tolist(), "max_len": args.max_len,
         "base_model": model_name, "seed": seed}, indent=2))
    shutil.rmtree(targs.output_dir, ignore_errors=True)   # drop intermediate checkpoints
    return metrics, th, out_dir


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--model-name", default="nlpaueb/legal-bert-base-uncased")
    ap.add_argument("--seeds", type=int, nargs="+", default=[42, 43, 44])
    ap.add_argument("--max-len", type=int, default=128,
                    help="set from the EDA token-length percentiles")
    ap.add_argument("--lr", type=float, default=3e-5)
    ap.add_argument("--batch-size", type=int, default=16)
    ap.add_argument("--epochs", type=int, default=8)
    ap.add_argument("--models-dir", default="models")
    ap.add_argument("--export-dir", default=None,
                    help="copy the best-validation seed here (e.g. models/legal-bert-unfair)")
    ap.add_argument("--no-mlflow", action="store_true")
    args = ap.parse_args()

    ds, labels = load_unfair_tos()
    short = SHORT_NAMES.get(args.model_name, args.model_name.split("/")[-1])
    all_metrics, dirs = [], []
    for seed in args.seeds:
        metrics, th, out_dir = train_one_seed(args.model_name, ds, labels, seed, args)
        print_metrics(f"{short} seed={seed}", metrics)
        all_metrics.append(metrics)
        dirs.append(out_dir)
        if not args.no_mlflow:
            params = {"model": args.model_name, "seed": seed, "max_len": args.max_len,
                      "lr": args.lr, "batch_size": args.batch_size, "epochs": args.epochs}
            with mlflow_run(f"{short}-s{seed}", params):
                import mlflow
                mlflow.log_metrics(metrics)
                mlflow.log_dict({"labels": labels, "thresholds": th.tolist()}, "thresholds.json")

    keys = all_metrics[0].keys()
    summary = {f"{k}_mean": float(np.mean([m[k] for m in all_metrics])) for k in keys}
    summary.update({f"{k}_std": float(np.std([m[k] for m in all_metrics], ddof=1))
                    for k in keys if len(all_metrics) > 1})
    print(f"{short} over seeds {args.seeds}:")
    for k in ("micro_f1_8", "macro_f1_8", "micro_f1_lexglue", "macro_f1_lexglue", "unfair_recall"):
        std = summary.get(f"{k}_std", 0.0)
        print(f"  {k}: {summary[k + '_mean']:.4f} ± {std:.4f}")
    if not args.no_mlflow and len(args.seeds) > 1:
        with mlflow_run(f"{short}-summary", {"model": args.model_name, "seeds": args.seeds}):
            import mlflow
            mlflow.log_metrics(summary)

    if args.export_dir:
        best = int(np.argmax([m["val_macro_f1_tuned"] for m in all_metrics]))
        export = Path(args.export_dir)
        shutil.rmtree(export, ignore_errors=True)
        shutil.copytree(dirs[best], export)
        print(f"exported seed {args.seeds[best]} (best validation macro-F1) to {export}")


if __name__ == "__main__":
    main()
