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
| ML code (`ml/src/`): data, evaluation, all 5 model families, pipeline, predictor, scoring | Done, tested |
| Training notebooks (`ml/notebooks/01`–`05`) | Done, ready to run on Colab |
| Web app (FastAPI, Express, React, Docker Compose) | Done, tested |
| Trained Legal-BERT model, results table, ToS;DR validation | **Your runs**: train on Colab, collect ToS documents |

No results are committed. Every number in the report must come from your own logged MLflow runs (`python -m src.results_table` builds the table from them).

---

## Training the models on Google Colab

All training runs on Colab's free GPU. Each notebook clones this repo, installs packages, trains, and saves models and results to your Google Drive, so nothing is lost if Colab disconnects.

### Notebooks

Run them in this order.

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
├── mlruns/                       # MLflow store (mlflow.db + artifacts/): every metric comes from here
├── figures/                      # EDA and scoring plots for the report
└── results.md                    # results table generated from MLflow
```

To browse the MLflow runs, download `mlruns/` into `ml/` and run `mlflow ui --backend-store-uri sqlite:///mlruns/mlflow.db` from `ml/`.

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

## Running the app

### Which model the app uses

The ML service loads the first model it finds, in this order:

1. `ml/models/legal-bert-unfair/`: copy it from `My Drive/fineguard/models/` after running notebook 04
2. `ml/models/tfidf_svm.joblib`, then `ml/models/tfidf_logreg.joblib`: made in about a minute on any laptop with `cd ml && python -m src.train_classical`

So you can demo the full app with the TF-IDF baseline before Legal-BERT is trained. If `score_weights.json` (made by notebook 05) sits next to the model, the service uses the fitted v2 weights; otherwise equal v1 weights. The Result page shows which model and weights produced each analysis.

### With Docker

```bash
docker compose up --build
```

- Web app: http://localhost:5173
- ML service API docs: http://localhost:8000/docs
- Express API: http://localhost:5000/api/health

### Without Docker (development)

Needs Python 3.11+, Node 20+ and PostgreSQL 16.

```bash
# 1. Database
createdb fpg            # the server creates the table on startup

# 2. ML service
cd ml
python -m venv .venv && source .venv/bin/activate
pip install -r requirements.txt
python -m src.train_classical          # only if you have no Legal-BERT model yet
uvicorn service.main:app --reload --port 8000

# 3. API server (new terminal)
cd server && npm install
DATABASE_URL=postgres://localhost:5432/fpg npm run dev

# 4. Client (new terminal); /api is proxied to localhost:5000
cd client && npm install && npm run dev
```

### Tests

```bash
cd ml && pytest          # pipeline, scoring, evaluation, models, API (no downloads needed)
cd server && npm test    # Express routes with a fake database and ML service
```

## Training from the command line

The notebooks call these scripts; you can also run them directly from `ml/`:

| Script | What it does |
|---|---|
| `python -m src.train_classical` | TF-IDF + LogReg / SVM (`--ngram 1 1`, `--no-balanced`, `--c-logreg`, `--c-svm`) |
| `python -m src.train_neural --model mlp --emb w2v` | Word2Vec/GloVe + MLP (`--emb glove`, `--seeds 42 43 44`) |
| `python -m src.train_neural --model bilstm --emb w2v` | BiLSTM (`--freeze-emb`, `--pooling last`) |
| `python -m src.train_transformer --model-name nlpaueb/legal-bert-base-uncased --export-dir models/legal-bert-unfair` | BERT / Legal-BERT, 3 seeds, exports the best-validation seed |
| `python -m src.validate_score --model models/legal-bert-unfair --write-weights` | Risk score vs ToS;DR grades (Spearman ρ, bootstrap 95% CI, LOO) |
| `python -m src.results_table --out results.md` | Results table built from MLflow runs |

Every script tunes thresholds on validation only, evaluates on test with `src/evaluate.py`, and logs to MLflow (`--no-mlflow` to skip).

## Layout

```
client/          React + Vite + Tailwind (Analyze, Result, History pages)
server/          Express + PostgreSQL (analyses table, full result stored as JSONB)
ml/src/          data, evaluate, models, training scripts, pipeline, predictor, scoring
ml/service/      FastAPI: /health, /analyze/text, /analyze/pdf, /analyze/url
ml/notebooks/    01 EDA → 05 scoring (run on Colab)
ml/data/         collected ToS documents and ToS;DR grades
ml/tests/        pytest suite
docs/            implementation guide
```
