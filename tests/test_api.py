import sys
from pathlib import Path

import pytest
import requests
from fastapi.testclient import TestClient

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))
import api  # noqa: E402

CHUNK = {"id": "abc", "doc": "report2.pdf", "page_start": 14, "page_end": 14,
         "text": "The per-transaction limit for UPI LITE is ₹1000."}


def fake_answer(retriever, question, **kwargs):
    return {"answer": "The per-transaction limit for UPI LITE is ₹1000 [1].",
            "results": [(CHUNK, 0.9)], "cited": [1], "invalid_citations": [],
            "refused": False, "retrieval_s": 0.01, "generation_s": 0.02, "tokens_per_s": 50.0}


class FakeVerifier:
    def verify(self, answer, results):
        return [{"claim": "The per-transaction limit for UPI LITE is ₹1000.",
                 "score": 0.99, "supported": True, "level": "verified"}]


class FakeRedactor:
    def redact(self, text):
        return text, []


@pytest.fixture
def client(monkeypatch):
    monkeypatch.setattr(api, "Retriever", lambda *args, **kwargs: object())
    monkeypatch.setattr(api, "ClaimVerifier", lambda *args, **kwargs: FakeVerifier())
    monkeypatch.setattr(api, "Redactor", lambda *args, **kwargs: FakeRedactor())
    monkeypatch.setattr(api, "answer_question", fake_answer)
    monkeypatch.setattr(api, "log_interaction", lambda *args, **kwargs: None)
    with TestClient(api.app) as test_client:
        yield test_client


def test_health(client):
    response = client.get("/health")
    assert response.status_code == 200
    assert response.json()["models_loaded"] is True


def test_ask_returns_answer_sources_and_claims(client):
    response = client.post("/ask", json={"question": "What is the UPI LITE limit?"})
    assert response.status_code == 200
    data = response.json()
    assert data["refused"] is False
    assert data["sources"][0]["doc"] == "report2.pdf"
    assert data["claims"][0]["level"] == "verified"


def test_too_short_question_is_rejected(client):
    assert client.post("/ask", json={"question": "hi"}).status_code == 422


def test_llm_down_returns_503(client, monkeypatch):
    def llm_down(*args, **kwargs):
        raise requests.exceptions.ConnectionError()
    monkeypatch.setattr(api, "answer_question", llm_down)
    response = client.post("/ask", json={"question": "What is the UPI LITE limit?"})
    assert response.status_code == 503