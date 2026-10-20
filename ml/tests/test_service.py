"""API tests against a tiny TF-IDF model trained on the fly (no downloads needed)."""
import joblib
import numpy as np
import pytest
from fastapi.testclient import TestClient
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.linear_model import LogisticRegression
from sklearn.multiclass import OneVsRestClassifier

LABELS = ["Limitation of liability", "Unilateral termination"]


def make_model(tmp_path):
    texts = ["we are not liable for any damages", "we shall not be liable for losses",
             "we may terminate your account at any time", "we can suspend your account",
             "you can contact us by email", "this page describes our service"]
    y = np.array([[1, 0], [1, 0], [0, 1], [0, 1], [0, 0], [0, 0]])
    vec = TfidfVectorizer().fit(texts)
    clf = OneVsRestClassifier(LogisticRegression(C=10)).fit(vec.transform(texts), y)
    path = tmp_path / "tiny.joblib"
    joblib.dump({"vectorizer": vec, "model": clf, "thresholds": np.array([0.5, 0.5]),
                 "labels": LABELS}, path)
    return path


@pytest.fixture()
def client(tmp_path, monkeypatch):
    path = make_model(tmp_path)
    monkeypatch.delenv("MODEL_DIR", raising=False)
    monkeypatch.setenv("MODEL_PATHS", str(path))
    from service.main import app
    with TestClient(app) as c:
        yield c


def test_health(client):
    body = client.get("/health").json()
    assert body["model_loaded"] is True
    assert body["labels"] == LABELS
    assert body["weights_version"] == "v1-equal"


def test_analyze_text(client):
    r = client.post("/analyze/text", json={
        "text": "We may terminate your account at any time without notice. "
                "We are not liable for any damages arising from your use of the service. "
                "You can contact us by email at any time.",
        "service_name": "Example"})
    assert r.status_code == 200
    body = r.json()
    assert 0 <= body["risk_score"] <= 100
    assert body["grade"] in "ABCDE"
    assert body["n_sentences"] == 3 == len(body["sentences"])
    assert body["n_flagged"] == len(body["flagged"]) >= 1
    assert [c["category"] for c in body["categories"]] == LABELS
    assert "not legal advice" in body["disclaimer"]


def test_text_too_short(client):
    assert client.post("/analyze/text", json={"text": "short"}).status_code == 422


def test_no_sentences(client):
    r = client.post("/analyze/text", json={"text": "Hi.\n\nOk.\n\nYes.\n\nNo.\n\nFine."})
    assert r.status_code == 422


def test_pdf(client):
    import pymupdf

    doc = pymupdf.open()
    doc.new_page().insert_text((72, 72), "We may terminate your account at any time.")
    r = client.post("/analyze/pdf", files={"file": ("tos.pdf", doc.tobytes(), "application/pdf")})
    assert r.status_code == 200
    assert r.json()["source"] == "tos.pdf"


def test_bad_pdf(client):
    r = client.post("/analyze/pdf", files={"file": ("x.pdf", b"not a pdf", "application/pdf")})
    assert r.status_code == 422


def test_private_url_rejected(client):
    r = client.post("/analyze/url", json={"url": "http://127.0.0.1:8000/health"})
    assert r.status_code == 422


def test_no_model_returns_503(tmp_path, monkeypatch):
    monkeypatch.delenv("MODEL_DIR", raising=False)
    monkeypatch.setenv("MODEL_PATHS", str(tmp_path / "missing"))
    from service.main import app
    with TestClient(app) as c:
        assert c.get("/health").json()["model_loaded"] is False
        r = c.post("/analyze/text", json={"text": "We may terminate your account at any time."})
        assert r.status_code == 503


def test_fitted_weights_are_loaded(tmp_path, monkeypatch):
    import json

    path = make_model(tmp_path)
    path.with_suffix(".score_weights.json").write_text(
        json.dumps({"labels": LABELS, "weights": [0.0, 1.0]}))
    monkeypatch.delenv("MODEL_DIR", raising=False)
    monkeypatch.setenv("MODEL_PATHS", str(path))
    from service.main import app
    with TestClient(app) as c:
        assert c.get("/health").json()["weights_version"] == "v2-fitted"
        body = c.post("/analyze/text", json={
            "text": "We are not liable for any damages arising from the service."}).json()
        # only liability is flagged, and its weight is 0, so the score is 0
        assert body["categories"][0]["count"] == 1
        assert body["risk_score"] == 0.0 and body["grade"] == "A"
