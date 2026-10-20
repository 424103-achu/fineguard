"""FastAPI ML service: extract text -> split sentences -> classify -> score.

The model, thresholds and scoring weights are loaded once at startup.
Run from ml/:  uvicorn service.main:app --reload --port 8000   (docs at /docs)

Model selection: MODEL_DIR (if set), else the first existing path in MODEL_PATHS
(default: models/legal-bert-unfair, then the TF-IDF baselines).
"""
import json
import logging
import os
import time
from contextlib import asynccontextmanager
from pathlib import Path

import numpy as np
from fastapi import FastAPI, File, Form, HTTPException, UploadFile

from service.schemas import (AnalysisResponse, CategorySummary, FlaggedClause, HealthResponse,
                             TextRequest, UrlRequest)
from src.pipeline import split_sentences, text_from_pdf, text_from_url
from src.predictor import load_predictor
from src.scoring import risk_score, to_grade

log = logging.getLogger("fineguard")
DEFAULT_MODEL_PATHS = "models/legal-bert-unfair,models/tfidf_svm.joblib,models/tfidf_logreg.joblib"
MAX_PDF_BYTES = 10 * 1024 * 1024
state = {}


def candidate_paths():
    paths = os.environ.get("MODEL_PATHS", DEFAULT_MODEL_PATHS).split(",")
    if os.environ.get("MODEL_DIR"):
        paths.insert(0, os.environ["MODEL_DIR"])
    return [p.strip() for p in paths if p.strip()]


def load_weights(model_path, labels):
    """v2 fitted weights if score_weights.json exists next to the model, else v1 equal weights."""
    p = Path(model_path)
    f = p / "score_weights.json" if p.is_dir() else p.with_suffix(".score_weights.json")
    if f.exists():
        cfg = json.loads(f.read_text())
        if cfg.get("labels", labels) == labels and len(cfg["weights"]) == len(labels):
            return np.asarray(cfg["weights"], dtype=float), "v2-fitted"
        log.warning("Ignoring %s: labels do not match the model", f)
    return np.ones(len(labels)), "v1-equal"


@asynccontextmanager
async def lifespan(app: FastAPI):
    for path in candidate_paths():
        if Path(path).exists():
            predictor = load_predictor(path)
            state["predictor"] = predictor
            state["weights"], state["weights_version"] = load_weights(path, predictor.labels)
            log.warning("Loaded model %s (%s weights)", path, state["weights_version"])
            break
    else:
        log.warning("No model found in %s; analysis endpoints will return 503", candidate_paths())
    yield
    state.clear()


app = FastAPI(title="Fine-Print Guard ML service", version="1.0.0", lifespan=lifespan,
              description="Flags potentially unfair ToS clauses (not legal advice).")


def run_analysis(text: str, source: str, service_name: str | None) -> AnalysisResponse:
    predictor = state.get("predictor")
    if predictor is None:
        raise HTTPException(status_code=503, detail="No model loaded on the ML service")
    t0 = time.perf_counter()
    sentences = split_sentences(text)
    if not sentences:
        raise HTTPException(status_code=422, detail="No readable sentences found")
    probs, preds = predictor.predict(sentences)
    counts = preds.sum(axis=0)
    weights = state["weights"]
    score = risk_score(counts, weights)

    flagged = []
    for i in np.where(preds.any(axis=1))[0]:
        idx = np.where(preds[i])[0]
        flagged.append(FlaggedClause(
            index=int(i), sentence=sentences[i],
            categories=[predictor.labels[j] for j in idx],
            confidence={predictor.labels[j]: round(float(probs[i, j]), 4) for j in idx}))
    flagged.sort(key=lambda c: max(c.confidence.values()), reverse=True)

    norm_w = weights / weights.sum() if weights.sum() > 0 else np.ones_like(weights) / len(weights)
    return AnalysisResponse(
        source=source, service_name=service_name or None,
        n_sentences=len(sentences), n_flagged=len(flagged),
        risk_score=round(score, 1), grade=to_grade(score),
        categories=[CategorySummary(category=l, count=int(c), weight=round(float(w), 4))
                    for l, c, w in zip(predictor.labels, counts, norm_w)],
        flagged=flagged,
        sentences=sentences,
        processing_ms=round((time.perf_counter() - t0) * 1000, 1),
        model_version=predictor.version,
        weights_version=state["weights_version"],
    )


@app.get("/health", response_model=HealthResponse)
def health():
    p = state.get("predictor")
    return HealthResponse(status="ok", model_loaded=p is not None,
                          model_version=p.version if p else None,
                          weights_version=state.get("weights_version"),
                          labels=p.labels if p else [])


@app.post("/analyze/text", response_model=AnalysisResponse)
def analyze_text(req: TextRequest):
    return run_analysis(req.text, "text", req.service_name)


@app.post("/analyze/url", response_model=AnalysisResponse)
def analyze_url(req: UrlRequest):
    if "predictor" not in state:
        raise HTTPException(status_code=503, detail="No model loaded on the ML service")
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
