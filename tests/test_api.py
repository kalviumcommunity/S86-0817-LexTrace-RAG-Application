import io
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


def test_upload_unsupported_file_type():
    file_bytes = b"echo 'binary payload'"
    files = {"file": ("malicious_script.exe", file_bytes, "application/octet-stream")}
    response = client.post("/documents", files=files)
    assert response.status_code == 415
    data = response.json()
    assert "Unsupported file type" in data["detail"]


def test_upload_empty_file():
    empty_bytes = b""
    files = {"file": ("empty_policy.txt", empty_bytes, "text/plain")}
    response = client.post("/documents", files=files)
    assert response.status_code == 400
    data = response.json()
    assert "empty" in data["detail"].lower()


def test_upload_and_query_document_runtime():
    """
    Test uploading a new policy document at runtime, verifying indexing,
    and querying the newly indexed information immediately without restart.
    """
    sample_policy_content = (
        "# Remote Work and Equipment Policy\n\n"
        "## Equipment Stipend\n"
        "All remote employees receive a one-time home office equipment stipend of $1,500.\n\n"
        "## Internet Reimbursement\n"
        "Employees can expense up to $80 per month for high-speed home internet connectivity.\n"
    )

    filename = "test_remote_work_policy.md"
    files = {"file": (filename, sample_policy_content.encode("utf-8"), "text/markdown")}

    # 1. Upload the document
    upload_res = client.post("/documents", files=files)
    assert upload_res.status_code == 201
    upload_data = upload_res.json()
    assert upload_data["status"] == "indexed"
    assert upload_data["filename"] == filename
    assert upload_data["summary"]["chunks"] >= 1
    assert upload_data["summary"]["indexed"] >= 1

    # 2. Query the newly uploaded content at runtime
    query_payload = {
        "question": "What is the one-time home office equipment stipend amount for remote employees?"
    }
    query_res = client.post("/query", json=query_payload)
    assert query_res.status_code == 200
    query_data = query_res.json()

    assert "$1,500" in query_data["answer"] or "1,500" in query_data["answer"] or "1500" in query_data["answer"]
    assert query_data["status"] == "answered"
    assert any(filename in s["source"] for s in query_data["sources"])
