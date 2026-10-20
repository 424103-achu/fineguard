"""Loads a saved model once and classifies sentences in batches.

Supported artifacts:
  * a directory with a Hugging Face model + tokenizer + thresholds.json (BERT / Legal-BERT)
  * a .joblib file saved by src.train_classical (TF-IDF baselines)
"""
import json
from pathlib import Path

import numpy as np


class ClausePredictor:
    """Transformer predictor (BERT / Legal-BERT)."""

    def __init__(self, model_dir: str, device: str | None = None):
        import torch
        from transformers import AutoModelForSequenceClassification, AutoTokenizer

        self._torch = torch
        cfg = json.loads((Path(model_dir) / "thresholds.json").read_text())
        self.labels = list(cfg["labels"])
        self.thresholds = np.array(cfg["thresholds"])
        self.max_len = cfg.get("max_len", 128)
        self.version = Path(model_dir).name
        self.device = device or ("cuda" if torch.cuda.is_available() else "cpu")
        self.tok = AutoTokenizer.from_pretrained(model_dir)
        self.model = AutoModelForSequenceClassification.from_pretrained(model_dir)
        self.model.to(self.device).eval()

    def predict_probs(self, sentences: list[str], batch_size: int = 32) -> np.ndarray:
        torch = self._torch
        chunks = []
        with torch.inference_mode():
            for i in range(0, len(sentences), batch_size):
                batch = [s.lower() for s in sentences[i:i + batch_size]]   # training text is lowercase
                enc = self.tok(batch, truncation=True, max_length=self.max_len,
                               padding=True, return_tensors="pt").to(self.device)
                chunks.append(torch.sigmoid(self.model(**enc).logits).float().cpu().numpy())
        return np.vstack(chunks)

    def predict(self, sentences: list[str], batch_size: int = 32):
        """Returns (probs, preds), each of shape (n_sentences, n_labels)."""
        if not sentences:
            empty = np.zeros((0, len(self.labels)))
            return empty, empty.astype(int)
        probs = self.predict_probs(sentences, batch_size)
        return probs, (probs >= self.thresholds).astype(int)


class SklearnPredictor:
    """TF-IDF + LogReg/SVM predictor from src.train_classical."""

    def __init__(self, path: str):
        import joblib

        bundle = joblib.load(path)
        self.vectorizer = bundle["vectorizer"]
        self.model = bundle["model"]
        self.thresholds = np.asarray(bundle["thresholds"])
        self.labels = list(bundle["labels"])
        self.version = Path(path).stem

    def predict(self, sentences: list[str], batch_size: int = 0):
        if not sentences:
            empty = np.zeros((0, len(self.labels)))
            return empty, empty.astype(int)
        X = self.vectorizer.transform([s.lower() for s in sentences])
        probs = np.asarray(self.model.predict_proba(X))
        return probs, (probs >= self.thresholds).astype(int)


def load_predictor(path: str):
    p = Path(path)
    if p.is_dir() and (p / "thresholds.json").exists():
        return ClausePredictor(str(p))
    if p.is_file() and p.suffix == ".joblib":
        return SklearnPredictor(str(p))
    raise FileNotFoundError(
        f"No model found at {path}. Expected a directory with thresholds.json "
        f"(BERT/Legal-BERT) or a .joblib file from src.train_classical.")
