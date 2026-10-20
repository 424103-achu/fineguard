"""Phases 4–5: MLP and BiLSTM classifiers. Both output raw logits (sigmoid at loss/inference)."""
import torch
import torch.nn as nn


class MLP(nn.Module):
    def __init__(self, d_in, n_labels, hidden=256, dropout=0.3):
        super().__init__()
        self.net = nn.Sequential(
            nn.Linear(d_in, hidden), nn.ReLU(), nn.Dropout(dropout),
            nn.Linear(hidden, n_labels),
        )

    def forward(self, x):
        return self.net(x)   # raw logits; sigmoid is applied in the loss / at inference


class BiLSTMClassifier(nn.Module):
    def __init__(self, emb_matrix, n_labels, hidden=128, dropout=0.3, freeze_emb=False,
                 pooling="max"):
        super().__init__()
        self.pooling = pooling
        self.emb = nn.Embedding.from_pretrained(
            torch.as_tensor(emb_matrix, dtype=torch.float32), freeze=freeze_emb, padding_idx=0)
        self.lstm = nn.LSTM(self.emb.embedding_dim, hidden, batch_first=True, bidirectional=True)
        self.drop = nn.Dropout(dropout)
        self.out = nn.Linear(2 * hidden, n_labels)

    def forward(self, ids, lengths):
        x = self.emb(ids)
        packed = nn.utils.rnn.pack_padded_sequence(
            x, lengths.cpu(), batch_first=True, enforce_sorted=False)
        out, (h, _) = self.lstm(packed)
        if self.pooling == "last":
            pooled = torch.cat([h[-2], h[-1]], dim=1)   # final forward + backward states
        else:
            out, _ = nn.utils.rnn.pad_packed_sequence(out, batch_first=True)   # (B, T, 2H)
            steps = torch.arange(out.size(1), device=out.device)[None, :]
            mask = (steps < lengths.to(out.device)[:, None]).unsqueeze(-1)
            pooled = out.masked_fill(~mask, float("-inf")).max(dim=1).values  # max-pool real tokens
        return self.out(self.drop(pooled))
