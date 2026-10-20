# Fine-Print Guard

6-credit college major project. An NLP + ML system that reads a Terms of Service (ToS) document, flags sentences that are **potentially unfair under EU consumer law** across 8 categories, and gives the service a 0–100 risk score and an A–E grade.

Full design, code skeletons, timeline and viva questions: `docs/fine-print-guard-implementation.md`. Read the relevant section before building a phase.

## Scope rules

- Say "potentially unfair", never "illegal" or "violates the law". It is not legal advice; the UI always shows the disclaimer.
- English only. No plain-language rewriting and no translations.
- Input is pasted text, a PDF upload, or a URL.

## Data

- UNFAIR-ToS from LexGLUE: `load_dataset("coastalcph/lex_glue", "unfair_tos")`. Splits: 5,532 train / 2,275 validation / 1,607 test sentences.
- Multi-label: `labels` is a list per sentence; an empty list means fair.
- Read label names from `ds["train"].features["labels"].feature.names`. Never hardcode the label order.
- Keep the chronological split. Never reshuffle train/validation/test.
- The text is lowercased: use uncased models and lowercase input at inference.

## ML conventions

- Model ladder: TF-IDF + LogReg/SVM → Word2Vec/GloVe + MLP → BiLSTM → BERT (`bert-base-uncased`) → Legal-BERT (`nlpaueb/legal-bert-base-uncased`).
- Output: sigmoid + `BCEWithLogitsLoss` with `pos_weight`. Never softmax.
- Tune per-label thresholds on validation only, then apply them to test.
- Evaluate every model with `ml/src/evaluate.py`. Main metrics: 8-category micro/macro-F1 and unfair-sentence recall. LexGLUE-style scores (extra "none" column) are only for comparing with the paper.
- Transformers: run 3 seeds, report mean ± std.
- Log every run in MLflow. Never invent or estimate results; every number comes from a logged run.
- Code targets transformers 5.x (`warmup_steps=0.1`, not `warmup_ratio`), PyTorch 2.x, and `import pymupdf` (not `fitz`).

## Risk score

- Per category: f = 1 − e^(−n), where n is the number of flagged sentences.
- Score = 100 × Σ w·f / Σ w. Grades: A < 20, B < 40, C < 60, D < 80, E ≥ 80.
- Weights: v1 equal; v2 non-negative linear regression fitted to ToS;DR grades, validated with leave-one-out Spearman ρ and a bootstrap 95% CI.

## Stack and layout

- `client/` React + Vite + Tailwind (Analyze, Result, History pages)
- `server/` Express + PostgreSQL (`analyses` table, full result stored as JSONB)
- `ml/src/` data, evaluation, models, training, `pipeline.py` (extraction + sentence splitting), `predictor.py`, `scoring.py`
- `ml/service/` FastAPI: `/health`, `/analyze/text`, `/analyze/pdf`, `/analyze/url`
- `ml/notebooks/` EDA and experiments; `ml/models/` saved weights (gitignored)
- Train on Colab/Kaggle GPU. The service loads saved artifacts once at startup.

## Commands

- ML service: `cd ml && uvicorn service.main:app --reload --port 8000` (API docs at `/docs`)
- ML tests: `cd ml && pytest`; server tests: `cd server && npm test`
- Results table from MLflow: `cd ml && python -m src.results_table` (store: `ml/mlruns/mlflow.db`)
- Everything: `docker compose up`
