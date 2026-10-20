"""Phases 4–5: Word2Vec/GloVe + MLP, and BiLSTM.

Run from ml/:
  python -m src.train_neural --model mlp --emb w2v
  python -m src.train_neural --model mlp --emb glove
  python -m src.train_neural --model bilstm --emb w2v [--freeze-emb] [--pooling last]
"""
import argparse
from pathlib import Path

import numpy as np
import torch
from torch.utils.data import DataLoader

from src.data import load_all_splits
from src.embeddings import average_vectors, load_glove, train_word2vec
from src.evaluate import evaluate_probs
from src.models import MLP, BiLSTMClassifier
from src.torch_utils import (SequenceDataset, VectorDataset, build_embedding_matrix,
                             build_vocab, compute_pos_weight, fit, predict_probs,
                             sequence_collate, set_seed, vector_collate)
from src.tracking import log_results, mlflow_run, print_metrics


def get_vectors(kind, train_texts, dim, seed):
    if kind == "w2v":
        return train_word2vec(train_texts, dim=dim, seed=seed).wv
    if kind == "glove":
        return load_glove(f"glove-wiki-gigaword-{dim}")
    if kind == "random":
        return None
    raise ValueError(kind)


def run_mlp(splits, kv, args, device):
    (tr_x, tr_y), (va_x, va_y), (te_x, te_y) = splits
    Xtr, Xva, Xte = (average_vectors(t, kv) for t in (tr_x, va_x, te_x))
    mk = lambda X, y, shuffle, bs: DataLoader(VectorDataset(X, y), batch_size=bs,
                                              shuffle=shuffle, collate_fn=vector_collate)
    model = MLP(Xtr.shape[1], tr_y.shape[1], hidden=args.hidden, dropout=args.dropout)
    model = fit(model, mk(Xtr, tr_y, True, 64), mk(Xva, va_y, False, 256),
                compute_pos_weight(tr_y), epochs=args.epochs, lr=args.lr, device=device)
    p_va, _ = predict_probs(model, mk(Xva, va_y, False, 256), device)
    p_te, _ = predict_probs(model, mk(Xte, te_y, False, 256), device)
    return model, p_va, p_te, {}


def run_bilstm(splits, kv, args, device):
    (tr_x, tr_y), (va_x, va_y), (te_x, te_y) = splits
    vocab = build_vocab(tr_x)
    M = build_embedding_matrix(vocab, kv, args.dim, seed=args.seed)
    mk = lambda x, y, shuffle, bs: DataLoader(SequenceDataset(x, y, vocab, args.max_len),
                                              batch_size=bs, shuffle=shuffle,
                                              collate_fn=sequence_collate)
    model = BiLSTMClassifier(M, tr_y.shape[1], hidden=args.hidden, dropout=args.dropout,
                             freeze_emb=args.freeze_emb, pooling=args.pooling)
    model = fit(model, mk(tr_x, tr_y, True, 32), mk(va_x, va_y, False, 128),
                compute_pos_weight(tr_y), epochs=args.epochs, lr=args.lr, device=device)
    p_va, _ = predict_probs(model, mk(va_x, va_y, False, 128), device)
    p_te, _ = predict_probs(model, mk(te_x, te_y, False, 128), device)
    return model, p_va, p_te, {"vocab": vocab}


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--model", choices=["mlp", "bilstm"], required=True)
    ap.add_argument("--emb", choices=["w2v", "glove", "random"], default="w2v")
    ap.add_argument("--dim", type=int, default=200)
    ap.add_argument("--hidden", type=int, default=None)
    ap.add_argument("--dropout", type=float, default=0.3)
    ap.add_argument("--epochs", type=int, default=30)
    ap.add_argument("--lr", type=float, default=1e-3)
    ap.add_argument("--max-len", type=int, default=128)
    ap.add_argument("--freeze-emb", action="store_true")
    ap.add_argument("--pooling", choices=["max", "last"], default="max")
    ap.add_argument("--seeds", type=int, nargs="+", default=[42])
    ap.add_argument("--out-dir", default="models")
    ap.add_argument("--no-mlflow", action="store_true")
    args = ap.parse_args()
    if args.hidden is None:
        args.hidden = 256 if args.model == "mlp" else 128

    device = "cuda" if torch.cuda.is_available() else "cpu"
    data, labels = load_all_splits()
    splits = (data["train"], data["validation"], data["test"])
    Path(args.out_dir).mkdir(parents=True, exist_ok=True)

    for seed in args.seeds:
        args.seed = seed
        set_seed(seed)
        kv = get_vectors(args.emb, data["train"][0], args.dim, seed)
        runner = run_mlp if args.model == "mlp" else run_bilstm
        model, p_va, p_te, extra = runner(splits, kv, args, device)
        metrics, th = evaluate_probs(data["validation"][1], p_va, data["test"][1], p_te, labels)

        name = f"{args.model}_{args.emb}"
        if args.model == "bilstm":
            name += f"_{args.pooling}" + ("_frozen" if args.freeze_emb else "")
        print_metrics(f"{name} seed={seed}", metrics)
        torch.save({"state_dict": model.state_dict(), "thresholds": th, "labels": labels,
                    "args": vars(args), **extra}, Path(args.out_dir) / f"{name}_s{seed}.pt")
        if not args.no_mlflow:
            params = {k: v for k, v in vars(args).items() if k not in ("seeds", "out_dir")}
            with mlflow_run(f"{name}-s{seed}", params):
                log_results(metrics, th, labels)


if __name__ == "__main__":
    main()
