import numpy as np
import torch
from torch.utils.data import DataLoader

from src.models import MLP, BiLSTMClassifier
from src.torch_utils import (SequenceDataset, VectorDataset, build_embedding_matrix, build_vocab,
                             compute_pos_weight, fit, predict_probs, sequence_collate,
                             vector_collate)


def test_pos_weight():
    y = np.array([[1, 0], [0, 0], [0, 0], [0, 1]])
    w = compute_pos_weight(y)
    assert w.tolist() == [3.0, 3.0]


def test_mlp_trains():
    torch.manual_seed(0)
    X = np.random.default_rng(0).normal(size=(40, 5)).astype(np.float32)
    y = np.stack([(X[:, 0] > 0), (X[:, 1] > 0)], axis=1).astype(int)
    dl = DataLoader(VectorDataset(X, y), batch_size=8, collate_fn=vector_collate)
    model = fit(MLP(5, 2, hidden=16), dl, dl, compute_pos_weight(y), epochs=3, verbose=False)
    probs, _ = predict_probs(model, dl)
    assert probs.shape == (40, 2) and ((probs >= 0) & (probs <= 1)).all()


def test_bilstm_shapes_and_pooling():
    texts = ["we may terminate your account", "you agree to arbitration", "hello"]
    vocab = build_vocab(texts, min_count=1)
    M = build_embedding_matrix(vocab, None, 8)
    y = np.zeros((3, 2), dtype=int)
    batch = sequence_collate([SequenceDataset(texts, y, vocab)[i] for i in range(3)])
    for pooling in ("max", "last"):
        out = BiLSTMClassifier(M, 2, hidden=4, pooling=pooling)(**batch[0])
        assert out.shape == (3, 2)
