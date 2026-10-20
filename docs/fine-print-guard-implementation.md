# Fine-Print Guard — Implementation Guide

**Automated detection of potentially unfair clauses in Terms of Service, with a document-level risk score**

| | |
|---|---|
| **Task** | Sentence-level multi-label classification (8 unfair-clause categories) + document risk scoring |
| **Dataset** | UNFAIR-ToS (LexGLUE), 9,414 sentences from 50 Terms of Service |
| **Models** | TF-IDF + LR/SVM → Word2Vec + MLP → BiLSTM → BERT → Legal-BERT |
| **Stack** | Python (PyTorch, scikit-learn, Hugging Face, FastAPI) + React/Tailwind + Express + PostgreSQL |
| **Headline metrics** | Micro/macro-F1, unfair-clause recall, Spearman correlation of risk score vs ToS;DR grades, latency per document |

---

## Contents

1. [Problem statement and scope](#1-problem-statement-and-scope)
2. [Dataset](#2-dataset)
3. [System architecture](#3-system-architecture)
4. [Repository structure](#4-repository-structure)
5. [Environment setup](#5-environment-setup)
6. [Phase 1 — Data loading and EDA](#6-phase-1--data-loading-and-eda)
7. [Phase 2 — Shared evaluation module](#7-phase-2--shared-evaluation-module)
8. [Phase 3 — Classical baselines](#8-phase-3--classical-baselines)
9. [Phase 4 — Word2Vec + MLP](#9-phase-4--word2vec--mlp)
10. [Phase 5 — BiLSTM](#10-phase-5--bilstm)
11. [Phase 6 — BERT and Legal-BERT](#11-phase-6--bert-and-legal-bert)
12. [Phase 7 — Error analysis](#12-phase-7--error-analysis)
13. [Phase 8 — Document pipeline](#13-phase-8--document-pipeline)
14. [Phase 9 — Risk scoring engine](#14-phase-9--risk-scoring-engine)
15. [Phase 10 — FastAPI ML service](#15-phase-10--fastapi-ml-service)
16. [Phase 11 — Express + PostgreSQL](#16-phase-11--express--postgresql)
17. [Phase 12 — React dashboard](#17-phase-12--react-dashboard)
18. [Phase 13 — Docker Compose](#18-phase-13--docker-compose)
19. [Experiment tracking](#19-experiment-tracking)
20. [Testing](#20-testing)
21. [Results table and metrics checklist](#21-results-table-and-metrics-checklist)
22. [10-week timeline](#22-10-week-timeline)
23. [Risks and mitigations](#23-risks-and-mitigations)
24. [What is original in this project](#24-what-is-original-in-this-project)
25. [Viva preparation](#25-viva-preparation)
26. [References](#26-references)

---

## 1. Problem statement and scope

> Online users routinely accept Terms of Service (ToS) without reading them, because the documents are long and legalistic. This project builds an NLP system that reads a ToS document, identifies clauses that are potentially unfair under EU consumer law across 8 categories, and computes an overall risk score and grade for the service. The system flags clauses for the user's attention and does not constitute legal advice.

**Wording note:** the dataset labels sentences as *potentially unfair under EU consumer law*. It does not label legal violations. Use "potentially unfair" in the report, UI and viva.

### In scope

- Sentence-level detection of 8 categories of potentially unfair clauses
- A comparison of five model families under one evaluation protocol
- A document-level risk score (0–100) and grade (A–E)
- Validation of the score against an external rating source (ToS;DR)
- A web app: paste text, upload a PDF, or enter a URL, and see score, breakdown and flagged sentences

### Out of scope

- Legal advice or verdicts of illegality
- Plain-language rewriting or translation of clauses
- Non-English documents

### UI disclaimer text

> This tool highlights clauses that may be unfair to consumers based on a research dataset annotated under EU consumer law. It is not legal advice and may miss or misclassify clauses.

---

## 2. Dataset

### Source

UNFAIR-ToS (Lippi et al., 2019), distributed as part of the LexGLUE benchmark (Chalkidis et al., 2022). It is available on the Hugging Face Hub as `coastalcph/lex_glue`, config `unfair_tos`, in Parquet format (no loading script needed).

| Split | Sentences | Period of the ToS documents |
|---|---|---|
| Train | 5,532 | 2006–2016 |
| Validation | 2,275 | 2017 |
| Test | 1,607 | 2017 |
| **Total** | **9,414** | |

The split is **chronological**. Keep it. Do not reshuffle, because a random split would leak sentences from the same documents across train and test and inflate scores.

### Fields

- `text`: one ToS sentence (string)
- `labels`: a list of category indices; an **empty list means the sentence is fair**

The released sentences appear to be lowercased. Confirm this in EDA, use **uncased** models, and lowercase input at inference time.

### The 8 categories

| Category | Plain meaning |
|---|---|
| Limitation of liability | The provider limits or excludes its responsibility for losses or damages |
| Unilateral termination | The provider can suspend or close your account at its discretion |
| Unilateral change | The provider can change the terms or the service without your agreement |
| Content removal | The provider can delete your content at its discretion |
| Contract by using | Merely using the service counts as accepting the terms |
| Choice of law | A specific country's or state's law governs disputes |
| Jurisdiction | Disputes must be brought in a specific (often distant) court |
| Arbitration | Disputes go to private arbitration instead of court |

**Always read the index order from the dataset features** (`ds["train"].features["labels"].feature.names`) instead of hardcoding it, and spot-check a few examples per label during EDA.

### Published reference results (LexGLUE paper, test set)

| Model | μ-F1 | m-F1 |
|---|---|---|
| TF-IDF + SVM | 95.4 | 78.8 |
| BERT | 95.6 | 81.3 |
| Legal-BERT | 96.0 | 83.0 |

LexGLUE adds an extra "no unfair term" column when scoring this task, which is why micro-F1 is so high (most sentences are fair, and predicting "fair" correctly counts). The evaluation module below computes **both** the LexGLUE-style score (for comparison with the paper) and the 8-category score (the honest measure of unfair-clause detection). Report both and say which is which.

### Data you collect yourself

1. **30–50 real ToS documents** from services that ToS;DR has graded. Save the raw text and the date you collected it.
2. **ToS;DR grades (A–E)** for the same services. Check ToS;DR's terms of use before reusing their data, and note the date of each grade.
3. **Optional robustness set:** 200–300 sentences from current (2024–2026) or Indian app terms, labeled by you with the same 8 categories, to measure how well the model handles terms written after the dataset period.

---

## 3. System architecture

```
┌──────────────────────────┐
│  React + Tailwind client │  paste text / upload PDF / enter URL
└────────────┬─────────────┘
             │ REST (JSON / multipart)
┌────────────▼─────────────┐        ┌──────────────────────┐
│  Express API (Node)      │◄──────►│  PostgreSQL          │
│  /api/analyze/*          │  pg    │  analyses table      │
│  /api/analyses (history) │        └──────────────────────┘
└────────────┬─────────────┘
             │ HTTP (axios)
┌────────────▼──────────────────────────────────────────────┐
│  FastAPI ML service (Python)                              │
│  extract text → split sentences → classify → score        │
│  model + thresholds + scoring weights loaded once at boot │
└───────────────────────────────────────────────────────────┘
```

**Request flow**

1. The user submits text, a PDF or a URL in React.
2. Express forwards it to FastAPI.
3. FastAPI extracts text, splits it into sentences, runs the classifier in batches, applies the tuned per-label thresholds, and computes the risk score.
4. FastAPI returns JSON. Express stores it in PostgreSQL and returns it with the record id.
5. React renders the score, grade, category breakdown and flagged sentences.

Training happens offline in notebooks and `ml/src/` scripts. The service only loads saved artifacts.

---

## 4. Repository structure

```
fine-print-guard/
├── client/                          # React + Vite + Tailwind
│   └── src/
│       ├── api/client.js
│       ├── components/
│       │   ├── InputPanel.jsx
│       │   ├── ScoreGauge.jsx
│       │   ├── CategoryBreakdown.jsx
│       │   ├── FlaggedClauseList.jsx
│       │   └── DocumentView.jsx
│       └── pages/
│           ├── AnalyzePage.jsx
│           ├── ResultPage.jsx
│           └── HistoryPage.jsx
├── server/                          # Express + PostgreSQL
│   ├── db/
│   │   ├── pool.js
│   │   └── schema.sql
│   ├── routes/analyze.js
│   ├── app.js
│   └── Dockerfile
├── ml/                              # all Python
│   ├── data/
│   │   ├── tos_docs/                # collected ToS texts (gitignored if large)
│   │   └── tosdr_grades.csv
│   ├── notebooks/
│   │   ├── 01_eda.ipynb
│   │   ├── 02_baselines.ipynb
│   │   ├── 03_mlp_bilstm.ipynb
│   │   ├── 04_bert.ipynb
│   │   └── 05_scoring.ipynb
│   ├── src/
│   │   ├── __init__.py
│   │   ├── data.py                  # loading, multi-hot labels
│   │   ├── evaluate.py              # metrics, threshold tuning
│   │   ├── train_classical.py
│   │   ├── embeddings.py            # Word2Vec / GloVe, averaging
│   │   ├── torch_utils.py           # training loop, early stopping
│   │   ├── models.py                # MLP, BiLSTM
│   │   ├── train_neural.py
│   │   ├── train_transformer.py
│   │   ├── pipeline.py              # extraction + sentence splitting
│   │   ├── predictor.py             # loads model, batch inference
│   │   └── scoring.py               # risk score, weight fitting, stats
│   ├── service/
│   │   ├── __init__.py
│   │   ├── main.py                  # FastAPI app
│   │   └── schemas.py               # Pydantic models
│   ├── models/                      # saved weights (gitignored)
│   ├── tests/
│   ├── requirements.txt
│   ├── requirements-serve.txt
│   └── Dockerfile
├── docker-compose.yml
└── README.md
```

---

## 5. Environment setup

Train on Google Colab or Kaggle (free GPU). Run the service locally or in Docker.

`ml/requirements.txt` (training + development):

```text
numpy
pandas
scikit-learn
scipy
matplotlib
seaborn
datasets
transformers
accelerate
torch
gensim
spacy
pymupdf
trafilatura
fastapi
uvicorn[standard]
python-multipart
pydantic
mlflow
lime
pytest
httpx
```

`ml/requirements-serve.txt` (service only, keeps the Docker image small):

```text
numpy
scikit-learn
scipy
transformers
torch
spacy
pymupdf
trafilatura
fastapi
uvicorn[standard]
python-multipart
pydantic
```

Notes:

- The Python code in this guide was run end to end on synthetic data with transformers 5.19, PyTorch 2.14, scikit-learn 1.9, datasets 5.1 and FastAPI 0.143 (October 2026). If you install older versions, some argument names differ (noted where it matters).
- After everything works, run `pip freeze > requirements.lock.txt` so results are reproducible.
- `gensim` has had NumPy-version conflicts on Colab. If installing it breaks NumPy, run the Word2Vec step in its own environment and save the vectors to disk.
- For the CPU-only Docker image, install PyTorch from the CPU wheel index (`--index-url https://download.pytorch.org/whl/cpu`) to avoid downloading CUDA.

---

## 6. Phase 1 — Data loading and EDA

`ml/src/data.py`

```python
import numpy as np
from datasets import load_dataset

def load_unfair_tos():
    """Load UNFAIR-ToS and return (dataset_dict, label_names)."""
    ds = load_dataset("coastalcph/lex_glue", "unfair_tos")
    label_names = ds["train"].features["labels"].feature.names
    return ds, label_names

def to_multi_hot(label_lists, n_labels):
    """Convert [[0, 3], [], [5]] into a (n, n_labels) 0/1 matrix."""
    y = np.zeros((len(label_lists), n_labels), dtype=np.int64)
    for i, labs in enumerate(label_lists):
        for j in labs:
            y[i, j] = 1
    return y

def get_split(ds, split, n_labels):
    texts = list(ds[split]["text"])
    y = to_multi_hot(ds[split]["labels"], n_labels)
    return texts, y
```

**EDA checklist** (notebook `01_eda.ipynb`) — each item becomes a figure or table in your report:

- [ ] Sentences per split; percentage fair vs potentially unfair
- [ ] Count per category in each split (expect strong imbalance; some categories are rare)
- [ ] Number of labels per sentence (how often a sentence has 2+ categories)
- [ ] Sentence length distribution in words and in BERT tokens (95th/99th percentile decides `max_length`)
- [ ] Top TF-IDF terms per category
- [ ] Two or three example sentences per category (spot-check that the label order is what you think)
- [ ] Confirm whether the text is lowercased

```python
import pandas as pd
from src.data import load_unfair_tos, get_split

ds, LABELS = load_unfair_tos()
n = len(LABELS)
texts, y = get_split(ds, "train", n)

df = pd.DataFrame(y, columns=LABELS)
print("fair share:", (df.sum(axis=1) == 0).mean())
print(df.sum().sort_values(ascending=False))
print("labels per sentence:\n", df.sum(axis=1).value_counts())

lengths = pd.Series([len(t.split()) for t in texts])
print(lengths.describe(percentiles=[0.5, 0.95, 0.99]))
```

---

## 7. Phase 2 — Shared evaluation module

Every model is evaluated by the same code, so the results table is a fair comparison.

`ml/src/evaluate.py`

```python
import numpy as np
from sklearn.metrics import f1_score, precision_score, recall_score, classification_report

def add_none_column(y):
    """Append a 'no unfair term' column, as LexGLUE does for this task."""
    none = (y.sum(axis=1) == 0).astype(int).reshape(-1, 1)
    return np.hstack([y, none])

def evaluate(y_true, y_pred):
    y_true = np.asarray(y_true)
    y_pred = np.asarray(y_pred)
    yt9, yp9 = add_none_column(y_true), add_none_column(y_pred)
    any_true, any_pred = y_true.any(axis=1), y_pred.any(axis=1)
    return {
        # 8 unfair categories only: the honest measure of detection
        "micro_f1_8": f1_score(y_true, y_pred, average="micro", zero_division=0),
        "macro_f1_8": f1_score(y_true, y_pred, average="macro", zero_division=0),
        # LexGLUE-style (8 + 'none'): use when comparing with the paper
        "micro_f1_lexglue": f1_score(yt9, yp9, average="micro", zero_division=0),
        "macro_f1_lexglue": f1_score(yt9, yp9, average="macro", zero_division=0),
        # sentence level: is it potentially unfair at all?
        "unfair_precision": precision_score(any_true, any_pred, zero_division=0),
        "unfair_recall": recall_score(any_true, any_pred, zero_division=0),
    }

def per_label_report(y_true, y_pred, label_names):
    return classification_report(y_true, y_pred, target_names=label_names, zero_division=0)

def tune_thresholds(y_true, probs, grid=None):
    """Pick one threshold per label that maximises that label's F1.
    Tune on the VALIDATION set only, then apply the thresholds to test."""
    if grid is None:
        grid = np.arange(0.05, 0.96, 0.05)
    thresholds = np.full(probs.shape[1], 0.5)
    for j in range(probs.shape[1]):
        best_f1, best_t = -1.0, 0.5
        for t in grid:
            f = f1_score(y_true[:, j], (probs[:, j] >= t).astype(int), zero_division=0)
            if f > best_f1:
                best_f1, best_t = f, t
        thresholds[j] = best_t
    return thresholds

def apply_thresholds(probs, thresholds):
    return (probs >= thresholds).astype(int)
```

Before comparing your LexGLUE-style numbers to the paper's, check your `add_none_column` against the evaluation code in the LexGLUE GitHub repository.

---

## 8. Phase 3 — Classical baselines

`ml/src/train_classical.py`

```python
import json
import joblib
import numpy as np
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.linear_model import LogisticRegression
from sklearn.svm import LinearSVC
from sklearn.calibration import CalibratedClassifierCV
from sklearn.multiclass import OneVsRestClassifier

from src.data import load_unfair_tos, get_split
from src.evaluate import evaluate, tune_thresholds, apply_thresholds

ds, LABELS = load_unfair_tos()
n = len(LABELS)
Xtr_txt, Ytr = get_split(ds, "train", n)
Xva_txt, Yva = get_split(ds, "validation", n)
Xte_txt, Yte = get_split(ds, "test", n)

tfidf = TfidfVectorizer(ngram_range=(1, 2), min_df=2, sublinear_tf=True)
Xtr = tfidf.fit_transform(Xtr_txt)   # fit on train only
Xva = tfidf.transform(Xva_txt)
Xte = tfidf.transform(Xte_txt)

models = {
    "tfidf_logreg": OneVsRestClassifier(
        LogisticRegression(max_iter=2000, class_weight="balanced", C=4.0)
    ),
    # LinearSVC has no predict_proba; calibration gives probabilities
    # so thresholds can be tuned the same way as for every other model
    "tfidf_svm": OneVsRestClassifier(
        CalibratedClassifierCV(LinearSVC(class_weight="balanced", C=0.5), cv=3)
    ),
}

results = {}
for name, clf in models.items():
    clf.fit(Xtr, Ytr)
    th = tune_thresholds(Yva, clf.predict_proba(Xva))
    Yhat = apply_thresholds(clf.predict_proba(Xte), th)
    results[name] = evaluate(Yte, Yhat)
    joblib.dump({"vectorizer": tfidf, "model": clf, "thresholds": th}, f"models/{name}.joblib")

print(json.dumps(results, indent=2))
```

Things to try and record:

- `ngram_range` (1,1) vs (1,2); `C` values; with vs without `class_weight="balanced"`
- Default 0.5 threshold vs tuned thresholds (report both; the gain is a resume number)

---

## 9. Phase 4 — Word2Vec + MLP

`ml/src/embeddings.py`

```python
import re
import numpy as np
from gensim.models import Word2Vec

TOKEN_RE = re.compile(r"[a-z0-9']+")

def tokenize(text):
    return TOKEN_RE.findall(text.lower())

def train_word2vec(texts, dim=200, seed=42):
    sentences = [tokenize(t) for t in texts]
    return Word2Vec(sentences=sentences, vector_size=dim, window=5, min_count=2,
                    sg=1, epochs=20, workers=4, seed=seed)

def average_vectors(texts, kv):
    """kv: a gensim KeyedVectors object (w2v.wv or loaded GloVe vectors)."""
    out = np.zeros((len(texts), kv.vector_size), dtype=np.float32)
    for i, t in enumerate(texts):
        vecs = [kv[w] for w in tokenize(t) if w in kv]
        if vecs:
            out[i] = np.mean(vecs, axis=0)
    return out
```

Word2Vec trained on only 5.5K sentences is weak. Also compare pretrained GloVe (`gensim.downloader.load("glove-wiki-gigaword-200")`). Domain-trained vs general-purpose embeddings is a good row pair for the results table.

`ml/src/models.py` (MLP part)

```python
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
```

`ml/src/torch_utils.py` (shared by the MLP and BiLSTM)

```python
import copy
import numpy as np
import torch
import torch.nn as nn
from sklearn.metrics import f1_score

def compute_pos_weight(y_train, max_weight=20.0):
    """neg/pos ratio per label, so rare categories are not ignored."""
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
        patience=4, device="cpu"):
    """Train with early stopping on validation macro-F1 (threshold 0.5)."""
    model.to(device)
    loss_fn = nn.BCEWithLogitsLoss(pos_weight=pos_weight.to(device))
    optimizer = torch.optim.AdamW(model.parameters(), lr=lr, weight_decay=1e-4)
    best_f1, best_state, bad_epochs = -1.0, None, 0
    for epoch in range(epochs):
        train_loss, _, _ = run_epoch(model, train_loader, loss_fn, optimizer, device)
        val_loss, val_probs, val_y = run_epoch(model, val_loader, loss_fn, None, device)
        val_f1 = f1_score(val_y, (val_probs >= 0.5).astype(int), average="macro", zero_division=0)
        print(f"epoch {epoch + 1}: train {train_loss:.4f}  val {val_loss:.4f}  macro-F1 {val_f1:.4f}")
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
```

MLP data wiring:

```python
import numpy as np
import torch
from torch.utils.data import Dataset, DataLoader

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

# Xtr_vec = average_vectors(train_texts, kv) etc.
# train_loader = DataLoader(VectorDataset(Xtr_vec, Ytr), batch_size=64, shuffle=True, collate_fn=vector_collate)
# val_loader   = DataLoader(VectorDataset(Xva_vec, Yva), batch_size=256, shuffle=False, collate_fn=vector_collate)
# model = fit(MLP(Xtr_vec.shape[1], n_labels), train_loader, val_loader, compute_pos_weight(Ytr), device=device)
# then: val probs -> tune_thresholds -> test probs -> apply_thresholds -> evaluate
```

---

## 10. Phase 5 — BiLSTM

Add to `ml/src/models.py`:

```python
class BiLSTMClassifier(nn.Module):
    def __init__(self, emb_matrix, n_labels, hidden=128, dropout=0.3, freeze_emb=False):
        super().__init__()
        self.emb = nn.Embedding.from_pretrained(
            torch.tensor(emb_matrix, dtype=torch.float32), freeze=freeze_emb, padding_idx=0)
        self.lstm = nn.LSTM(emb_matrix.shape[1], hidden, batch_first=True, bidirectional=True)
        self.drop = nn.Dropout(dropout)
        self.out = nn.Linear(2 * hidden, n_labels)

    def forward(self, ids, lengths):
        x = self.emb(ids)
        packed = nn.utils.rnn.pack_padded_sequence(
            x, lengths.cpu(), batch_first=True, enforce_sorted=False)
        out, _ = self.lstm(packed)
        out, _ = nn.utils.rnn.pad_packed_sequence(out, batch_first=True)   # (B, T, 2H)
        steps = torch.arange(out.size(1), device=out.device)[None, :]
        mask = (steps < lengths.to(out.device)[:, None]).unsqueeze(-1)
        pooled = out.masked_fill(~mask, float("-inf")).max(dim=1).values  # max-pool real tokens
        return self.out(self.drop(pooled))
```

Vocabulary, encoding and batching:

```python
import numpy as np
import torch
import torch.nn as nn
from collections import Counter
from torch.utils.data import Dataset
from src.embeddings import tokenize

PAD, UNK = 0, 1

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
        if w in kv:
            M[i] = kv[w]
    return M

class SequenceDataset(Dataset):
    def __init__(self, texts, y, vocab, max_len=128):
        self.ids = [encode(t, vocab, max_len) for t in texts]
        self.y = y.astype(np.float32)
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
```

Train with the same `fit` function. Experiments worth a row each: frozen vs fine-tuned embeddings; max-pool vs last hidden state.

---

## 11. Phase 6 — BERT and Legal-BERT

`ml/src/train_transformer.py`

```python
import json
import numpy as np
import torch
import torch.nn as nn
from sklearn.metrics import f1_score
from transformers import (AutoTokenizer, AutoModelForSequenceClassification,
                          DataCollatorWithPadding, EarlyStoppingCallback,
                          Trainer, TrainingArguments)

from src.data import load_unfair_tos, to_multi_hot
from src.evaluate import evaluate, tune_thresholds, apply_thresholds
from src.torch_utils import compute_pos_weight

MODEL_NAME = "nlpaueb/legal-bert-base-uncased"   # also run "bert-base-uncased"
OUT_DIR = "models/legal-bert-unfair"
MAX_LEN = 128                                     # set from your EDA token-length percentiles

ds, LABELS = load_unfair_tos()
N = len(LABELS)
tok = AutoTokenizer.from_pretrained(MODEL_NAME)

def preprocess(batch):
    enc = tok(batch["text"], truncation=True, max_length=MAX_LEN)
    enc["labels"] = [[1.0 if i in labs else 0.0 for i in range(N)] for labs in batch["labels"]]
    return enc

encoded = ds.map(preprocess, batched=True, remove_columns=ds["train"].column_names)

base_collator = DataCollatorWithPadding(tok)
def collate(features):
    batch = base_collator(features)
    batch["labels"] = batch["labels"].float()   # BCE needs float32 targets
    return batch

model = AutoModelForSequenceClassification.from_pretrained(
    MODEL_NAME, num_labels=N, problem_type="multi_label_classification",
    id2label=dict(enumerate(LABELS)), label2id={l: i for i, l in enumerate(LABELS)})

class WeightedTrainer(Trainer):
    """BCE with per-label pos_weight to handle class imbalance."""
    def __init__(self, *args, pos_weight=None, **kwargs):
        super().__init__(*args, **kwargs)
        self.pos_weight = pos_weight

    def compute_loss(self, model, inputs, return_outputs=False, **kwargs):
        labels = inputs.pop("labels")
        outputs = model(**inputs)
        loss = nn.functional.binary_cross_entropy_with_logits(
            outputs.logits, labels, pos_weight=self.pos_weight.to(outputs.logits.device))
        return (loss, outputs) if return_outputs else loss

def compute_metrics(eval_pred):
    logits, labels = eval_pred
    preds = (1 / (1 + np.exp(-logits)) >= 0.5).astype(int)
    return {"micro_f1": f1_score(labels, preds, average="micro", zero_division=0),
            "macro_f1": f1_score(labels, preds, average="macro", zero_division=0)}

args = TrainingArguments(
    output_dir="runs/legal-bert",
    eval_strategy="epoch",
    save_strategy="epoch",
    load_best_model_at_end=True,
    metric_for_best_model="macro_f1",
    greater_is_better=True,
    learning_rate=3e-5,
    per_device_train_batch_size=16,
    per_device_eval_batch_size=64,
    num_train_epochs=8,
    weight_decay=0.01,
    warmup_steps=0.1,          # transformers v5: a float < 1 means 10% of steps (on v4 use warmup_ratio=0.1)
    fp16=torch.cuda.is_available(),
    save_total_limit=2,
    seed=42,
    report_to="none",          # or "mlflow"
)

Ytr = to_multi_hot(ds["train"]["labels"], N)
trainer = WeightedTrainer(
    model=model, args=args,
    train_dataset=encoded["train"], eval_dataset=encoded["validation"],
    data_collator=collate, compute_metrics=compute_metrics,
    callbacks=[EarlyStoppingCallback(early_stopping_patience=2)],
    pos_weight=compute_pos_weight(Ytr),
)
trainer.train()

def probs_for(split):
    logits = trainer.predict(encoded[split]).predictions
    return 1 / (1 + np.exp(-logits))

Yva = to_multi_hot(ds["validation"]["labels"], N)
Yte = to_multi_hot(ds["test"]["labels"], N)
thresholds = tune_thresholds(Yva, probs_for("validation"))
metrics = evaluate(Yte, apply_thresholds(probs_for("test"), thresholds))
print(json.dumps(metrics, indent=2))

trainer.save_model(OUT_DIR)
tok.save_pretrained(OUT_DIR)
with open(f"{OUT_DIR}/thresholds.json", "w") as f:
    json.dump({"labels": LABELS, "thresholds": thresholds.tolist(), "max_len": MAX_LEN}, f, indent=2)
```

Notes:

- Run **3 seeds** for BERT and Legal-BERT and report mean ± std. Single runs on small datasets vary a lot.
- If Colab runs out of memory: lower the batch size, keep `fp16`, or use `distilbert-base-uncased` and report it as an extra row.
- **Speed row (optional):** measure latency per full document for Legal-BERT vs DistilBERT on CPU. That gives an efficiency number for the resume.

---

## 12. Phase 7 — Error analysis

This section is what convinces examiners the numbers are real.

- Per-category precision, recall and F1 (`per_label_report`) for the best model
- A table of false negatives (missed unfair sentences) and false positives, 5 examples each, with your explanation of why the model failed
- Which pairs of categories get confused (for example, Choice of law vs Jurisdiction often appear in the same sentence)
- Performance by sentence length bucket
- Optional: LIME explanations for 3–4 sentences to show which words drove a prediction

```python
import numpy as np
import pandas as pd

def error_table(texts, y_true, y_pred, label_names, k=5):
    rows = []
    for j, name in enumerate(label_names):
        fn = np.where((y_true[:, j] == 1) & (y_pred[:, j] == 0))[0][:k]
        fp = np.where((y_true[:, j] == 0) & (y_pred[:, j] == 1))[0][:k]
        rows += [{"label": name, "type": "FN", "text": texts[i]} for i in fn]
        rows += [{"label": name, "type": "FP", "text": texts[i]} for i in fp]
    return pd.DataFrame(rows)
```

---

## 13. Phase 8 — Document pipeline

`ml/src/pipeline.py`

```python
import ipaddress
import re
import socket
from urllib.parse import urlparse

import pymupdf        # PyMuPDF (older code imports it as `fitz`)
import spacy
import trafilatura

_nlp = spacy.blank("en")
_nlp.add_pipe("sentencizer")
_nlp.max_length = 2_000_000

MIN_WORDS = 4

def text_from_pdf(data: bytes) -> str:
    with pymupdf.open(stream=data, filetype="pdf") as doc:
        return "\n".join(page.get_text() for page in doc)

def _is_public_host(hostname: str) -> bool:
    """Block localhost / private addresses so the URL feature can't probe your network."""
    try:
        for info in socket.getaddrinfo(hostname, None):
            ip = ipaddress.ip_address(info[4][0])
            if ip.is_private or ip.is_loopback or ip.is_link_local or ip.is_reserved:
                return False
        return True
    except socket.gaierror:
        return False

def text_from_url(url: str) -> str:
    parsed = urlparse(url)
    if parsed.scheme not in ("http", "https") or not parsed.hostname:
        raise ValueError("Only http(s) URLs are supported")
    if not _is_public_host(parsed.hostname):
        raise ValueError("URL host is not allowed")
    html = trafilatura.fetch_url(url)
    if html is None:
        raise ValueError("Could not download the page")
    text = trafilatura.extract(html, include_comments=False, include_tables=True)
    if not text:
        raise ValueError("Could not extract readable text (the page may need JavaScript)")
    return text

def split_sentences(text: str) -> list[str]:
    """Split on line breaks first (ToS use lists and headings), then on sentence boundaries."""
    sentences = []
    for block in re.split(r"\n\s*\n|\n(?=\s*(?:[-•*]|\d+[.)]|\([a-z0-9]+\))\s)", text):
        block = re.sub(r"\s+", " ", block).strip()
        if not block:
            continue
        for sent in _nlp(block).sents:
            s = sent.text.strip()
            if len(s.split()) >= MIN_WORDS:
                sentences.append(s)
    return sentences
```

`ml/src/predictor.py`

```python
import json
from pathlib import Path

import numpy as np
import torch
from transformers import AutoTokenizer, AutoModelForSequenceClassification

class ClausePredictor:
    def __init__(self, model_dir: str, device: str | None = None):
        cfg = json.loads((Path(model_dir) / "thresholds.json").read_text())
        self.labels = cfg["labels"]
        self.thresholds = np.array(cfg["thresholds"])
        self.max_len = cfg.get("max_len", 128)
        self.device = device or ("cuda" if torch.cuda.is_available() else "cpu")
        self.tok = AutoTokenizer.from_pretrained(model_dir)
        self.model = AutoModelForSequenceClassification.from_pretrained(model_dir)
        self.model.to(self.device).eval()

    @torch.inference_mode()
    def predict(self, sentences: list[str], batch_size: int = 32):
        """Returns (probs, preds), each of shape (n_sentences, n_labels)."""
        if not sentences:
            empty = np.zeros((0, len(self.labels)))
            return empty, empty.astype(int)
        chunks = []
        for i in range(0, len(sentences), batch_size):
            batch = [s.lower() for s in sentences[i:i + batch_size]]   # training text is lowercase
            enc = self.tok(batch, truncation=True, max_length=self.max_len,
                           padding=True, return_tensors="pt").to(self.device)
            chunks.append(torch.sigmoid(self.model(**enc).logits).cpu().numpy())
        probs = np.vstack(chunks)
        return probs, (probs >= self.thresholds).astype(int)
```

---

## 14. Phase 9 — Risk scoring engine

### Score design

For each category *c*, count the flagged sentences *n<sub>c</sub>* in the document. Repeated clauses of the same type add risk, but with diminishing returns:

- feature: *f<sub>c</sub> = 1 − e<sup>−n<sub>c</sub></sup>* (0 when absent, about 0.63 for one clause, close to 1 for several)
- score: *S = 100 × Σ w<sub>c</sub> f<sub>c</sub> / Σ w<sub>c</sub>* (0 = no flagged clauses, 100 = every category present repeatedly)
- grade: A (0–20), B (20–40), C (40–60), D (60–80), E (80–100)

**Version 1 weights:** all equal. **Version 2 weights:** fitted to ToS;DR grades (below). Report both and whether fitting improved agreement.

`ml/src/scoring.py`

```python
import numpy as np
from scipy.stats import spearmanr
from sklearn.linear_model import LinearRegression
from sklearn.model_selection import LeaveOneOut, cross_val_predict

GRADE_BANDS = [(20, "A"), (40, "B"), (60, "C"), (80, "D"), (101, "E")]
TOSDR_TO_NUM = {"A": 1, "B": 2, "C": 3, "D": 4, "E": 5}

def doc_features(counts):
    """counts: flagged-sentence count per category for one document."""
    return 1.0 - np.exp(-np.asarray(counts, dtype=float))

def risk_score(counts, weights):
    w = np.asarray(weights, dtype=float)
    if w.sum() <= 0:
        w = np.ones_like(w)
    return float(100.0 * (w * doc_features(counts)).sum() / w.sum())

def to_grade(score):
    for upper, grade in GRADE_BANDS:
        if score < upper:
            return grade
    return "E"

def fit_weights(F, grades_num):
    """F: (n_docs, n_labels) features; grades_num: ToS;DR grade as 1..5.
    Non-negative weights, so no category can lower the risk."""
    reg = LinearRegression(positive=True).fit(F, grades_num)
    w = reg.coef_
    return w / w.sum() if w.sum() > 0 else np.ones_like(w) / len(w)

def validate_against_tosdr(F, grades_num):
    """Leave-one-out predictions, so the score is never judged on a document it was fitted on."""
    reg = LinearRegression(positive=True)
    pred = cross_val_predict(reg, F, grades_num, cv=LeaveOneOut())
    rho, p = spearmanr(pred, grades_num)
    return rho, p, pred

def bootstrap_ci(x, y, n_boot=2000, seed=0):
    """95% confidence interval for Spearman's rho."""
    rng = np.random.default_rng(seed)
    x, y = np.asarray(x), np.asarray(y)
    vals = []
    for _ in range(n_boot):
        idx = rng.integers(0, len(x), len(x))
        r, _ = spearmanr(x[idx], y[idx])
        if not np.isnan(r):
            vals.append(r)
    return np.percentile(vals, [2.5, 97.5])
```

### Validation procedure (notebook `05_scoring.ipynb`)

1. Run every collected ToS document through `split_sentences` and `ClausePredictor`.
2. Build `F` (one row of `doc_features` per document) and `g` (ToS;DR grades as 1–5).
3. Report Spearman ρ for **equal weights** (no fitting) and for **fitted weights** (leave-one-out), each with a bootstrap 95% CI.
4. Plot your score vs ToS;DR grade (box plot per grade).

**Expect a moderate correlation, not a perfect one.** ToS;DR grades also cover things outside these 8 categories (for example, tracking and data sharing), and their grade may refer to a different version of the terms than the one you collected. Say this in the report. It is a strength to explain it, not a weakness.

---

## 15. Phase 10 — FastAPI ML service

`ml/service/schemas.py`

```python
from pydantic import BaseModel, Field, HttpUrl

DISCLAIMER = ("This tool highlights clauses that may be unfair to consumers based on a research "
              "dataset annotated under EU consumer law. It is not legal advice and may miss or "
              "misclassify clauses.")

class TextRequest(BaseModel):
    text: str = Field(min_length=20, max_length=500_000)
    service_name: str | None = None

class UrlRequest(BaseModel):
    url: HttpUrl
    service_name: str | None = None

class FlaggedClause(BaseModel):
    index: int                       # position of the sentence in the document
    sentence: str
    categories: list[str]
    confidence: dict[str, float]     # probability for each flagged category

class CategorySummary(BaseModel):
    category: str
    count: int

class AnalysisResponse(BaseModel):
    source: str
    service_name: str | None
    n_sentences: int
    n_flagged: int
    risk_score: float
    grade: str
    categories: list[CategorySummary]
    flagged: list[FlaggedClause]
    processing_ms: float
    model_version: str
    disclaimer: str = DISCLAIMER
```

`ml/service/main.py`

```python
import json
import os
import time
from contextlib import asynccontextmanager
from pathlib import Path

import numpy as np
from fastapi import FastAPI, File, Form, HTTPException, UploadFile

from service.schemas import (AnalysisResponse, CategorySummary, FlaggedClause,
                             TextRequest, UrlRequest)
from src.pipeline import split_sentences, text_from_pdf, text_from_url
from src.predictor import ClausePredictor
from src.scoring import risk_score, to_grade

MODEL_DIR = os.environ.get("MODEL_DIR", "models/legal-bert-unfair")
MAX_PDF_BYTES = 10 * 1024 * 1024
state = {}

@asynccontextmanager
async def lifespan(app: FastAPI):
    state["predictor"] = ClausePredictor(MODEL_DIR)
    weights_file = Path(MODEL_DIR) / "score_weights.json"
    n = len(state["predictor"].labels)
    state["weights"] = (np.array(json.loads(weights_file.read_text())["weights"])
                        if weights_file.exists() else np.ones(n))
    yield
    state.clear()

app = FastAPI(title="Fine-Print Guard ML service", lifespan=lifespan)

def run_analysis(text: str, source: str, service_name: str | None) -> AnalysisResponse:
    t0 = time.perf_counter()
    sentences = split_sentences(text)
    if not sentences:
        raise HTTPException(status_code=422, detail="No readable sentences found")
    predictor = state["predictor"]
    probs, preds = predictor.predict(sentences)
    counts = preds.sum(axis=0)
    score = risk_score(counts, state["weights"])

    flagged = []
    for i in np.where(preds.any(axis=1))[0]:
        cats = [predictor.labels[j] for j in np.where(preds[i])[0]]
        conf = {predictor.labels[j]: round(float(probs[i, j]), 4) for j in np.where(preds[i])[0]}
        flagged.append(FlaggedClause(index=int(i), sentence=sentences[i],
                                     categories=cats, confidence=conf))
    flagged.sort(key=lambda c: max(c.confidence.values()), reverse=True)

    return AnalysisResponse(
        source=source, service_name=service_name,
        n_sentences=len(sentences), n_flagged=len(flagged),
        risk_score=round(score, 1), grade=to_grade(score),
        categories=[CategorySummary(category=l, count=int(c))
                    for l, c in zip(predictor.labels, counts)],
        flagged=flagged,
        processing_ms=round((time.perf_counter() - t0) * 1000, 1),
        model_version=Path(MODEL_DIR).name,
    )

@app.get("/health")
def health():
    return {"status": "ok", "model_loaded": "predictor" in state}

@app.post("/analyze/text", response_model=AnalysisResponse)
def analyze_text(req: TextRequest):
    return run_analysis(req.text, "text", req.service_name)

@app.post("/analyze/url", response_model=AnalysisResponse)
def analyze_url(req: UrlRequest):
    try:
        text = text_from_url(str(req.url))
    except ValueError as e:
        raise HTTPException(status_code=422, detail=str(e))
    return run_analysis(text, str(req.url), req.service_name)

@app.post("/analyze/pdf", response_model=AnalysisResponse)
def analyze_pdf(file: UploadFile = File(...), service_name: str | None = Form(None)):
    data = file.file.read(MAX_PDF_BYTES + 1)
    if len(data) > MAX_PDF_BYTES:
        raise HTTPException(status_code=413, detail="PDF larger than 10 MB")
    try:
        text = text_from_pdf(data)
    except Exception:
        raise HTTPException(status_code=422, detail="Could not read the PDF")
    return run_analysis(text, file.filename or "upload.pdf", service_name)
```

Run locally from `ml/`: `uvicorn service.main:app --reload --port 8000`, then open `http://localhost:8000/docs` for the auto-generated API page (useful in the demo).

Endpoints are plain `def` (not `async def`) because inference is CPU-bound; FastAPI runs them in a thread pool so the server stays responsive.

---

## 16. Phase 11 — Express + PostgreSQL

`server/db/schema.sql`

```sql
CREATE TABLE IF NOT EXISTS analyses (
    id           SERIAL PRIMARY KEY,
    source       TEXT          NOT NULL,
    service_name TEXT,
    risk_score   NUMERIC(5, 1) NOT NULL,
    grade        CHAR(1)       NOT NULL,
    n_sentences  INT           NOT NULL,
    n_flagged    INT           NOT NULL,
    result       JSONB         NOT NULL,
    created_at   TIMESTAMPTZ   NOT NULL DEFAULT now()
);

CREATE INDEX IF NOT EXISTS idx_analyses_created ON analyses (created_at DESC);
```

`server/db/pool.js`

```js
import pg from "pg";
export const pool = new pg.Pool({ connectionString: process.env.DATABASE_URL });
```

`server/routes/analyze.js`

```js
import { Router } from "express";
import axios from "axios";
import multer from "multer";
import FormData from "form-data";
import { pool } from "../db/pool.js";

const router = Router();
const ML_URL = process.env.ML_URL || "http://localhost:8000";
const upload = multer({ limits: { fileSize: 10 * 1024 * 1024 } });

async function saveAnalysis(r) {
  const { rows } = await pool.query(
    `INSERT INTO analyses (source, service_name, risk_score, grade, n_sentences, n_flagged, result)
     VALUES ($1, $2, $3, $4, $5, $6, $7)
     RETURNING id, created_at`,
    [r.source, r.service_name, r.risk_score, r.grade, r.n_sentences, r.n_flagged, r]
  );
  return rows[0];
}

function forwardError(err, res, next) {
  if (err.response) {
    return res.status(err.response.status).json(err.response.data);
  }
  next(err);
}

router.post("/analyze/text", async (req, res, next) => {
  try {
    const { data } = await axios.post(`${ML_URL}/analyze/text`, req.body, { timeout: 120000 });
    const saved = await saveAnalysis(data);
    res.json({ ...data, id: saved.id, created_at: saved.created_at });
  } catch (err) { forwardError(err, res, next); }
});

router.post("/analyze/url", async (req, res, next) => {
  try {
    const { data } = await axios.post(`${ML_URL}/analyze/url`, req.body, { timeout: 120000 });
    const saved = await saveAnalysis(data);
    res.json({ ...data, id: saved.id, created_at: saved.created_at });
  } catch (err) { forwardError(err, res, next); }
});

router.post("/analyze/pdf", upload.single("file"), async (req, res, next) => {
  try {
    if (!req.file) return res.status(400).json({ detail: "No file uploaded" });
    const form = new FormData();
    form.append("file", req.file.buffer, {
      filename: req.file.originalname, contentType: req.file.mimetype,
    });
    if (req.body.service_name) form.append("service_name", req.body.service_name);
    const { data } = await axios.post(`${ML_URL}/analyze/pdf`, form, {
      headers: form.getHeaders(), timeout: 180000,
    });
    const saved = await saveAnalysis(data);
    res.json({ ...data, id: saved.id, created_at: saved.created_at });
  } catch (err) { forwardError(err, res, next); }
});

router.get("/analyses", async (_req, res, next) => {
  try {
    const { rows } = await pool.query(
      `SELECT id, source, service_name, risk_score, grade, n_flagged, created_at
       FROM analyses ORDER BY created_at DESC LIMIT 50`);
    res.json(rows);
  } catch (err) { next(err); }
});

router.get("/analyses/:id", async (req, res, next) => {
  try {
    const { rows } = await pool.query(
      `SELECT id, result, created_at FROM analyses WHERE id = $1`, [req.params.id]);
    if (!rows.length) return res.status(404).json({ detail: "Not found" });
    res.json({ ...rows[0].result, id: rows[0].id, created_at: rows[0].created_at });
  } catch (err) { next(err); }
});

export default router;
```

`server/app.js`

```js
import express from "express";
import cors from "cors";
import analyzeRouter from "./routes/analyze.js";

const app = express();
app.use(cors({ origin: process.env.CLIENT_ORIGIN || "http://localhost:5173" }));
app.use(express.json({ limit: "2mb" }));
app.use("/api", analyzeRouter);

app.use((err, _req, res, _next) => {
  console.error(err);
  res.status(500).json({ detail: "Internal server error" });
});

const port = process.env.PORT || 5000;
app.listen(port, () => console.log(`API listening on ${port}`));
```

Packages: `npm i express cors pg axios multer form-data` (set `"type": "module"` in `package.json`).

---

## 17. Phase 12 — React dashboard

### Pages

| Page | Contents |
|---|---|
| **Analyze** | Tabs for Paste text / Upload PDF / Enter URL, optional service name, Analyze button, loading state |
| **Result** | Score gauge with grade, category breakdown chart, flagged clause list, full document view, disclaimer |
| **History** | Table of past analyses (service, score, grade, date) linking to their Result pages |

### Components

- `ScoreGauge` — 0–100 semicircle with the grade letter; colour by band
- `CategoryBreakdown` — horizontal bar chart of flagged count per category (Recharts)
- `FlaggedClauseList` — cards with the sentence, category chips and confidence; filter by category; sorted by confidence
- `DocumentView` — all sentences in order, flagged ones highlighted; clicking a card scrolls to it
- Disclaimer banner on every Result page

### API client

```js
// client/src/api/client.js
const BASE = import.meta.env.VITE_API_URL || "http://localhost:5000/api";

async function handle(res) {
  const body = await res.json().catch(() => ({}));
  if (!res.ok) throw new Error(body.detail || `Request failed (${res.status})`);
  return body;
}

export const analyzeText = (text, serviceName) =>
  fetch(`${BASE}/analyze/text`, {
    method: "POST",
    headers: { "Content-Type": "application/json" },
    body: JSON.stringify({ text, service_name: serviceName || null }),
  }).then(handle);

export const analyzeUrl = (url, serviceName) =>
  fetch(`${BASE}/analyze/url`, {
    method: "POST",
    headers: { "Content-Type": "application/json" },
    body: JSON.stringify({ url, service_name: serviceName || null }),
  }).then(handle);

export const analyzePdf = (file, serviceName) => {
  const form = new FormData();
  form.append("file", file);
  if (serviceName) form.append("service_name", serviceName);
  return fetch(`${BASE}/analyze/pdf`, { method: "POST", body: form }).then(handle);
};

export const listAnalyses = () => fetch(`${BASE}/analyses`).then(handle);
export const getAnalysis = (id) => fetch(`${BASE}/analyses/${id}`).then(handle);
```

The response shape is exactly `AnalysisResponse` from section 15 plus `id` and `created_at`. To build `DocumentView`, either return all sentences from the API (add a `sentences` field) or show only the flagged ones; decide early.

---

## 18. Phase 13 — Docker Compose

`ml/Dockerfile`

```dockerfile
FROM python:3.11-slim
WORKDIR /app
COPY requirements-serve.txt .
RUN pip install --no-cache-dir torch --index-url https://download.pytorch.org/whl/cpu \
 && pip install --no-cache-dir -r requirements-serve.txt
COPY src ./src
COPY service ./service
EXPOSE 8000
CMD ["uvicorn", "service.main:app", "--host", "0.0.0.0", "--port", "8000"]
```

`docker-compose.yml`

```yaml
services:
  db:
    image: postgres:16
    environment:
      POSTGRES_USER: fpg
      POSTGRES_PASSWORD: fpg
      POSTGRES_DB: fpg
    volumes:
      - pgdata:/var/lib/postgresql/data
      - ./server/db/schema.sql:/docker-entrypoint-initdb.d/schema.sql:ro
    ports:
      - "5432:5432"

  ml:
    build: ./ml
    environment:
      MODEL_DIR: /models/legal-bert-unfair
    volumes:
      - ./ml/models:/models:ro
    ports:
      - "8000:8000"

  server:
    build: ./server
    environment:
      DATABASE_URL: postgres://fpg:fpg@db:5432/fpg
      ML_URL: http://ml:8000
      CLIENT_ORIGIN: http://localhost:5173
    depends_on:
      - db
      - ml
    ports:
      - "5000:5000"

  client:
    build: ./client
    ports:
      - "5173:80"
    depends_on:
      - server

volumes:
  pgdata:
```

The model weights are mounted, not copied into the image, so you can swap models without rebuilding.

---

## 19. Experiment tracking

Log every run from Phase 3 onward. This is your evidence for every number in the report and on the resume.

```python
import mlflow

mlflow.set_experiment("fine-print-guard")
with mlflow.start_run(run_name="tfidf-logreg-ngram12"):
    mlflow.log_params({"model": "tfidf+logreg", "ngram": "1-2", "C": 4.0, "class_weight": "balanced"})
    mlflow.log_metrics(metrics)                      # dict from evaluate()
    mlflow.log_dict({"thresholds": thresholds.tolist()}, "thresholds.json")
```

View runs with `mlflow ui`. For BERT, set `report_to="mlflow"` in `TrainingArguments`.

---

## 20. Testing

`ml/tests/test_scoring.py`

```python
import numpy as np
from src.scoring import risk_score, to_grade

def test_no_flags_is_zero():
    assert risk_score(np.zeros(8), np.ones(8)) == 0.0

def test_more_flags_never_lower_score():
    w = np.ones(8)
    assert risk_score([1, 0, 0, 0, 0, 0, 0, 0], w) <= risk_score([2, 0, 0, 0, 0, 0, 0, 0], w)

def test_grade_bands():
    assert to_grade(0) == "A"
    assert to_grade(59.9) == "C"
    assert to_grade(100) == "E"
```

`ml/tests/test_pipeline.py`

```python
from src.pipeline import split_sentences

def test_splits_lists_and_sentences():
    text = "Terms of use.\n\n1. We may change these terms at any time. You agree to this.\n2) We can delete your content."
    sents = split_sentences(text)
    assert "We may change these terms at any time." in sents
    assert any("delete your content" in s for s in sents)

def test_drops_tiny_fragments():
    assert split_sentences("Hi.\n\nOk.") == []
```

API smoke test (needs a trained model in `MODEL_DIR`):

```python
from fastapi.testclient import TestClient
from service.main import app

def test_health_and_analyze():
    with TestClient(app) as client:
        assert client.get("/health").json()["model_loaded"] is True
        r = client.post("/analyze/text", json={
            "text": "We may terminate your account at any time without notice. "
                    "We are not liable for any damages arising from your use of the service."})
        assert r.status_code == 200
        assert 0 <= r.json()["risk_score"] <= 100
```

---

## 21. Results table and metrics checklist

Fill this with **your own** numbers. Report the 8-category scores as your main result and the LexGLUE-style scores only for comparison with the paper.

| Model | μ-F1 (8) | m-F1 (8) | μ-F1 (LexGLUE) | m-F1 (LexGLUE) | Unfair recall | CPU ms / doc |
|---|---|---|---|---|---|---|
| TF-IDF + LogReg | | | | | | |
| TF-IDF + SVM | | | | | | |
| Word2Vec (own) + MLP | | | | | | |
| GloVe + MLP | | | | | | |
| BiLSTM | | | | | | |
| BERT (mean ± std, 3 seeds) | | | | | | |
| Legal-BERT (mean ± std, 3 seeds) | | | | | | |
| *Published: TF-IDF + SVM* | | | *95.4* | *78.8* | | |
| *Published: BERT* | | | *95.6* | *81.3* | | |
| *Published: Legal-BERT* | | | *96.0* | *83.0* | | |

### Other numbers to record

- [ ] Gain from threshold tuning vs a fixed 0.5 threshold
- [ ] Best model's gain over the TF-IDF baseline (macro-F1 points)
- [ ] Per-category F1 for the best model
- [ ] Spearman ρ (with 95% CI) of risk score vs ToS;DR grade: equal weights and fitted weights
- [ ] Number of real ToS documents analysed
- [ ] Average processing time per document (CPU)
- [ ] Optional: F1 on your hand-labeled current/Indian terms vs the test set

---

## 22. 10-week timeline

| Week | Build | Done when |
|---|---|---|
| 1 | Environment, data loading, EDA | EDA notebook with all checklist figures |
| 2 | Evaluation module, TF-IDF baselines, MLflow | First two rows of the results table |
| 3 | Embeddings + MLP | MLP rows logged |
| 4 | BiLSTM | BiLSTM row logged |
| 5 | BERT fine-tuning | BERT row (first seed) |
| 6 | Legal-BERT, 3 seeds each, error analysis | Transformer rows + error tables |
| 7 | Document pipeline; collect ToS docs and ToS;DR grades | 30+ documents processed end to end |
| 8 | Scoring engine, weight fitting, statistics | Spearman ρ with CI, score plots |
| 9 | FastAPI, Express, PostgreSQL, React, Docker | Full demo runs with `docker compose up` |
| 10 | Tests, polish, report, demo rehearsal | Report draft, results table final |

---

## 23. Risks and mitigations

| Risk | Mitigation |
|---|---|
| Some categories are rare, so their F1 is unstable | pos_weight, per-label thresholds, report per-category F1 and seed variance |
| Dataset ends in 2017; modern terms are written differently | Optional hand-labeled current/Indian robustness set; discuss as a limitation |
| Real ToS pages are messy (headings, lists, cookie banners) | Line-aware sentence splitting, minimum length filter, PDF and paste fallbacks |
| URL extraction fails on JavaScript-heavy pages | Clear error message telling the user to paste text or upload a PDF |
| ToS;DR grade covers more than the 8 categories, or a different version | Report correlation honestly, record dates, explain the gap |
| Colab GPU limits | fp16, smaller batch, DistilBERT fallback, save checkpoints to Drive |
| Over-claiming in the viva | "Potentially unfair", "not legal advice", and keep every number traceable in MLflow |

---

## 24. What is original in this project

Several public demos already classify sentences with this dataset, so do not claim the classifier itself is new. The contributions to state are:

1. **A controlled comparison** of five model families (classical, MLP, BiLSTM, BERT, Legal-BERT) under one evaluation protocol, with threshold tuning and seed variance
2. **A document-level risk score**, validated against an external rating source (ToS;DR) with leave-one-out evaluation and confidence intervals
3. **A real-world test** on terms collected now, years after the dataset period (and optionally on Indian apps' terms)
4. **A deployed end-to-end system** (React, Express, PostgreSQL, FastAPI, Docker) that analyses a full document in seconds

---

## 25. Viva preparation

Be ready to answer these without notes:

1. Why is this multi-label and not multi-class? What loss and output activation does that require?
2. What is the difference between micro-F1 and macro-F1, and why does the paper's micro-F1 look so high?
3. Why keep the chronological split instead of shuffling?
4. Why tune thresholds on the validation set and never on the test set?
5. What does `pos_weight` do in `BCEWithLogitsLoss`?
6. Why does a BiLSTM beat averaged embeddings? Why does BERT beat the BiLSTM?
7. What is self-attention? What is the [CLS] token used for in classification?
8. What did Legal-BERT add over BERT, and did it help on every category?
9. Which categories fail most and why? Show two real examples.
10. How is the risk score computed, and why are the weights constrained to be non-negative?
11. What does a Spearman ρ of your value mean, and why is a confidence interval needed with ~40 documents?
12. What are the system's limitations, and why is it not legal advice?

---

## 26. References

- Lippi, M. et al. (2019). *CLAUDETTE: an automated detector of potentially unfair clauses in online terms of service.* Artificial Intelligence and Law. arXiv:1805.01217
- Chalkidis, I. et al. (2022). *LexGLUE: A Benchmark Dataset for Legal Language Understanding in English.* ACL 2022. arXiv:2110.00976
- Chalkidis, I. et al. (2020). *LEGAL-BERT: The Muppets straight out of Law School.* Findings of EMNLP 2020
- Devlin, J. et al. (2019). *BERT: Pre-training of Deep Bidirectional Transformers for Language Understanding.* NAACL 2019
- Mikolov, T. et al. (2013). *Efficient Estimation of Word Representations in Vector Space.* arXiv:1301.3781
- Council Directive 93/13/EEC on unfair terms in consumer contracts
- Dataset: https://huggingface.co/datasets/coastalcph/lex_glue
- Evaluation code: https://github.com/coastalcph/lex-glue
- Terms of Service; Didn't Read: https://tosdr.org
