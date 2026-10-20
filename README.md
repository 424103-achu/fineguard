# Fine-Print Guard

An NLP + ML system that reads a Terms of Service (ToS) document, flags sentences that are **potentially unfair under EU consumer law** across 8 categories, and gives the service a 0–100 risk score and an A–E grade.

> This tool highlights clauses that may be unfair to consumers based on a research dataset annotated under EU consumer law. It is not legal advice and may miss or misclassify clauses.

- Full design and code plan: [`docs/fine-print-guard-implementation.md`](docs/fine-print-guard-implementation.md)
- Project rules for contributors and Claude: [`CLAUDE.md`](CLAUDE.md)
- Dataset: [UNFAIR-ToS (LexGLUE)](https://huggingface.co/datasets/coastalcph/lex_glue), 5,532 train / 2,275 validation / 1,607 test sentences

## Status

| Part | Status |
|---|---|
| Docs and plan | Done |
| ML code (`ml/src/`) | Not started |
| Training notebooks (`ml/notebooks/`) | Not started |
| Web app (FastAPI, Express, React, Docker) | Not started |

---

## Training the models on Google Colab

All training runs on Colab's free GPU. Each notebook clones this repo, installs packages, trains, and saves models and results to your Google Drive, so nothing is lost if Colab disconnects.

### Notebooks

Run them in this order. The notebooks are added to `ml/notebooks/` as each phase is built.

| # | Notebook | What it does | GPU? | Rough time |
|---|---|---|---|---|
| 1 | `01_eda.ipynb` | Dataset statistics and figures for the report | No | ~5 min |
| 2 | `02_baselines.ipynb` | TF-IDF + Logistic Regression / SVM | No | ~10 min |
| 3 | `03_mlp_bilstm.ipynb` | Word2Vec/GloVe + MLP, BiLSTM | Yes | ~20 min |
| 4 | `04_bert.ipynb` | BERT and Legal-BERT, 3 seeds each | Yes | Several hours in total |
| 5 | `05_scoring.ipynb` | Risk score vs ToS;DR grades (Spearman ρ, 95% CI) | No | ~10 min |

Times are estimates; your runs decide the real numbers.

### Step by step

1. **Open the notebook in Colab.** Go to [colab.research.google.com](https://colab.research.google.com), choose **File → Open notebook → GitHub**, type `424103-achu/fineguard`, and pick the notebook.
   - Or open it directly: `https://colab.research.google.com/github/424103-achu/fineguard/blob/main/ml/notebooks/<notebook>.ipynb`
   - If the repo is **private**, tick "Include private repos" and sign in to GitHub when Colab asks.
2. **Turn on the GPU** (notebooks 3 and 4): **Runtime → Change runtime type → T4 GPU → Save**.
3. **Run everything:** **Runtime → Run all**.
4. **Allow Google Drive access** when the popup appears. Outputs go to `My Drive/fineguard/`.
5. **Wait for it to finish.** Keep the tab open. Free Colab disconnects after long idle periods, and sessions are capped at a few hours.
6. **Check the last cell.** It prints the metrics table for the run.

### What every notebook does first

The first cells are the same in every notebook:

```python
# 1. Get the code
!git clone https://github.com/424103-achu/fineguard.git
%cd fineguard/ml
!pip install -q -r requirements.txt

# 2. Save outputs to Google Drive so they survive a disconnect
from google.colab import drive
drive.mount("/content/drive")
!mkdir -p /content/drive/MyDrive/fineguard/models /content/drive/MyDrive/fineguard/mlruns
!ln -sfn /content/drive/MyDrive/fineguard/models models
!ln -sfn /content/drive/MyDrive/fineguard/mlruns mlruns

# 3. Check the GPU (notebooks 3 and 4)
import torch
print("GPU:", torch.cuda.get_device_name(0) if torch.cuda.is_available() else "none")
```

If the repo is private, cloning needs a GitHub token. Create a fine-grained token with read-only access to this repo, add it in Colab's **Secrets** panel (key icon on the left) as `GITHUB_TOKEN`, and clone with:

```python
from google.colab import userdata
token = userdata.get("GITHUB_TOKEN")
!git clone https://{token}@github.com/424103-achu/fineguard.git
```

Never paste the token directly into a notebook cell.

### Where results end up (in your Google Drive)

```
My Drive/fineguard/
├── models/
│   ├── tfidf_logreg.joblib, tfidf_svm.joblib
│   ├── mlp_*.pt, bilstm_*.pt
│   └── legal-bert-unfair/        # weights + tokenizer + thresholds.json (used by the web app)
└── mlruns/                       # MLflow logs: every metric in the report comes from here
```

To browse the MLflow runs, download `mlruns/` and run `mlflow ui` locally, or in a Colab cell run `!mlflow ui --backend-store-uri mlruns` behind a tunnel.

### Rules for training runs

- Keep the chronological train/validation/test split. Never reshuffle.
- Tune per-label thresholds on **validation only**, then apply them to test.
- BERT and Legal-BERT: run 3 seeds (42, 43, 44) and report mean ± std.
- Every number in the report must come from a logged MLflow run.

### Troubleshooting

| Problem | Fix |
|---|---|
| `CUDA out of memory` | Lower `per_device_train_batch_size` to 8 and keep `fp16=True` |
| Colab disconnected mid-run | Re-run all cells; finished seeds are already saved in Drive. Re-run only the missing seed |
| `gensim` install breaks NumPy | Restart the runtime after installing, or run the Word2Vec cells in a separate notebook |
| "No GPU" printed | Runtime → Change runtime type → T4 GPU, then Runtime → Restart |
| GPU quota used up | Wait a day, or use Kaggle Notebooks (30 free GPU hours per week) with the same cells |

---

## Running the app locally (once built)

```bash
docker compose up
```

- Web app: http://localhost:5173
- ML service API docs: http://localhost:8000/docs

Copy `My Drive/fineguard/models/legal-bert-unfair/` into `ml/models/` first; the ML service loads it at startup.

## Layout

```
client/   React + Vite + Tailwind
server/   Express + PostgreSQL
ml/       data, evaluation, models, training, FastAPI service, notebooks
docs/     implementation guide
```
