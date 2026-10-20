"""Shared PyTorch training loop for the MLP and BiLSTM."""
import copy
import random
from collections import Counter

import numpy as np
import torch
import torch.nn as nn
from sklearn.metrics import f1_score
from torch.utils.data import Dataset

from src.embeddings import tokenize

PAD, UNK = 0, 1


def set_seed(seed):
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)


def compute_pos_weight(y_train, max_weight=20.0):
    """neg/pos ratio per label, so rare categories are not ignored."""
    y_train = np.asarray(y_train)
    pos = y_train.sum(axis=0)
    neg = len(y_train) - pos
    w = np.clip(neg / np.maximum(pos, 1), 1.0, max_weight)
    return torch.tensor(w, dtype=torch.float32)


def run_epoch(model, loader, loss_fn, optimizer=None, device="cpu"):
    training = optimizer is not None
    model.train(training)
    total_loss, all_logits, all_y = 0.0, [], []
    for inputs, y in loader:
        inputs = {k: v.to(device) for k, v in inputs.items()}
        y = y.to(device)
        with torch.set_grad_enabled(training):
            logits = model(**inputs)
            loss = loss_fn(logits, y)
            if training:
                optimizer.zero_grad()
                loss.backward()
                torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
                optimizer.step()
        total_loss += loss.item() * y.size(0)
        all_logits.append(logits.detach().cpu())
        all_y.append(y.cpu())
    probs = torch.sigmoid(torch.cat(all_logits)).numpy()
    return total_loss / len(loader.dataset), probs, torch.cat(all_y).numpy()


def fit(model, train_loader, val_loader, pos_weight, epochs=30, lr=1e-3,
        patience=4, device="cpu", verbose=True):
    """Train with early stopping on validation macro-F1 (threshold 0.5)."""
    model.to(device)
    loss_fn = nn.BCEWithLogitsLoss(pos_weight=pos_weight.to(device))
    optimizer = torch.optim.AdamW(model.parameters(), lr=lr, weight_decay=1e-4)
    best_f1, best_state, bad_epochs = -1.0, None, 0
    for epoch in range(epochs):
        train_loss, _, _ = run_epoch(model, train_loader, loss_fn, optimizer, device)
        val_loss, val_probs, val_y = run_epoch(model, val_loader, loss_fn, None, device)
        val_f1 = f1_score(val_y, (val_probs >= 0.5).astype(int), average="macro", zero_division=0)
        if verbose:
            print(f"epoch {epoch + 1}: train {train_loss:.4f}  val {val_loss:.4f}  "
                  f"macro-F1 {val_f1:.4f}")
        if val_f1 > best_f1:
            best_f1, best_state, bad_epochs = val_f1, copy.deepcopy(model.state_dict()), 0
        else:
            bad_epochs += 1
            if bad_epochs >= patience:
                break
    model.load_state_dict(best_state)
    return model


def predict_probs(model, loader, device="cpu"):
    model.to(device)
    _, probs, y = run_epoch(model, loader, lambda logits, y: torch.tensor(0.0), None, device)
    return probs, y


# ---- MLP data -------------------------------------------------------------

class VectorDataset(Dataset):
    def __init__(self, X, y):
        self.X = torch.tensor(X, dtype=torch.float32)
        self.y = torch.tensor(y, dtype=torch.float32)

    def __len__(self):
        return len(self.y)

    def __getitem__(self, i):
        return self.X[i], self.y[i]


def vector_collate(batch):
    X = torch.stack([b[0] for b in batch])
    y = torch.stack([b[1] for b in batch])
    return {"x": X}, y


# ---- BiLSTM data ----------------------------------------------------------

def build_vocab(texts, min_count=2):
    counts = Counter(w for t in texts for w in tokenize(t))
    itos = ["<pad>", "<unk>"] + [w for w, c in counts.items() if c >= min_count]
    return {w: i for i, w in enumerate(itos)}


def encode(text, vocab, max_len=128):
    ids = [vocab.get(w, UNK) for w in tokenize(text)][:max_len]
    return ids or [UNK]          # never return an empty sequence


def build_embedding_matrix(vocab, kv, dim, seed=42):
    rng = np.random.default_rng(seed)
    M = rng.normal(0, 0.1, size=(len(vocab), dim)).astype(np.float32)
    M[PAD] = 0.0
    for w, i in vocab.items():
        if kv is not None and w in kv:
            M[i] = kv[w]
    return M


class SequenceDataset(Dataset):
    def __init__(self, texts, y, vocab, max_len=128):
        self.ids = [encode(t, vocab, max_len) for t in texts]
        self.y = np.asarray(y).astype(np.float32)

    def __len__(self):
        return len(self.ids)

    def __getitem__(self, i):
        return self.ids[i], self.y[i]


def sequence_collate(batch):
    seqs = [torch.tensor(b[0], dtype=torch.long) for b in batch]
    lengths = torch.tensor([len(s) for s in seqs], dtype=torch.long)
    ids = nn.utils.rnn.pad_sequence(seqs, batch_first=True, padding_value=PAD)
    y = torch.tensor(np.stack([b[1] for b in batch]), dtype=torch.float32)
    return {"ids": ids, "lengths": lengths}, y
