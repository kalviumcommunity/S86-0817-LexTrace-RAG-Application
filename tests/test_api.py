import sys
from pathlib import Path
from unittest.mock import patch
import pytest
from fastapi.testclient import TestClient

PROJECT_ROOT = Path(__file__).resolve().parent.parent
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from src.api import app

client = TestClient(app)


def test_root_endpoint():
    response = client.get("/")
    assert response.status_code == 200
    data = response.json()
    assert data["service"] == "LexTrace RAG Backend API"
    assert "endpoints" in data


def test_health_check_endpoint():
    response = client.get("/health")
    assert response.status_code == 200
    data = response.json()
    assert data["status"] == "healthy"
    assert "collection_count" in data
    assert data["collection_count"] >= 0
    assert "chat_model" in data


def test_query_endpoint_valid_contract_question():
    payload = {
        "question": "What notice period is required for standard contract termination?"
    }
    response = client.post("/query", json=payload)
    assert response.status_code == 200
    data = response.json()
    assert "answer" in data
    assert "30 days" in data["answer"].lower()
    assert data["status"] == "answered"
    assert isinstance(data["sources"], list)
    assert len(data["sources"]) > 0
    assert any(s["source"] == "contract.txt" for s in data["sources"])


def test_query_endpoint_missing_context_refusal():
    payload = {
        "question": "What is the international travel reimbursement allowance for Paris?"
    }
    response = client.post("/query", json=payload)
    assert response.status_code == 200
    data = response.json()
    assert "could not find" in data["answer"].lower()
    assert data["status"] == "refused"
    assert data["sources"] == []


def test_query_endpoint_short_question_validation():
    payload = {"question": "Hi"}
    response = client.post("/query", json=payload)
    assert response.status_code == 422  # Pydantic validation error (min_length=3)


def test_query_endpoint_empty_payload_validation():
    response = client.post("/query", json={})
    assert response.status_code == 422


def test_query_endpoint_internal_error_handling():
    with patch("src.api.guarded_answer", side_effect=Exception("Database connection timeout")):
        payload = {"question": "When can the agreement be terminated?"}
        response = client.post("/query", json=payload)
        assert response.status_code == 500
        data = response.json()
        assert data["detail"] == "RAG service failed"


def test_query_endpoint_value_error_handling():
    with patch("src.api.guarded_answer", side_effect=ValueError("Invalid query syntax")):
        payload = {"question": "When can the agreement be terminated?"}
        response = client.post("/query", json=payload)
        assert response.status_code == 400
        data = response.json()
        assert data["detail"] == "Invalid query syntax"
