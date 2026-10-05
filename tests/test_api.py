"""Tests for FastAPI endpoints, request validation, error handling, and response schema.
"""

import pytest
from fastapi.testclient import TestClient
from src.api import app


@pytest.fixture
def client():
    """Provides TestClient with managed lifespan."""
    with TestClient(app) as test_client:
        yield test_client


def test_health_endpoint(client):
    """Verifies that /health returns status 200 and healthy."""
    resp = client.get("/health")
    assert resp.status_code == 200
    data = resp.json()
    assert data["status"] == "healthy"
    assert data["model_loaded"] is True


def test_model_info_endpoint(client):
    """Verifies that /model-info returns valid metadata."""
    resp = client.get("/model-info")
    assert resp.status_code == 200
    data = resp.json()
    assert "model_type" in data
    assert "decision_threshold" in data


def test_predict_endpoint_valid_request(client):
    """Verifies that POST /predict processes valid input and returns complete response schema."""
    payload = {
        "claim_id": "WC_TEST_01",
        "submitted_at": "2026-07-02 11:30",
        "partner_id": "SP3160",
        "sku": "KH-RV-03",
        "product_serial": "KH123456789",
        "days_since_purchase": 180,
        "claim_amount_inr": 1890.0,
        "photo_attached": "Y",
        "partner_inspected": "N",
        "claim_description": "motor not running",
        "customer_prior_claims": 2,
    }
    resp = client.post("/predict", json=payload)
    assert resp.status_code == 200
    data = resp.json()
    assert data["claim_id"] == "WC_TEST_01"
    assert 0.0 <= data["fraud_score"] <= 1.0
    assert data["risk_level"] in ["LOW", "MEDIUM", "HIGH"]
    assert data["decision"] in ["APPROVE", "REVIEW", "MANUAL REVIEW"]
    assert isinstance(data["reasons"], list)
    assert len(data["reasons"]) >= 1


def test_predict_endpoint_invalid_request(client):
    """Verifies that invalid input triggers Pydantic HTTP 422 error."""
    # Negative claim amount and missing mandatory claim_id
    invalid_payload = {
        "claim_amount_inr": -500.0,
        "days_since_purchase": -10,
    }
    resp = client.post("/predict", json=invalid_payload)
    assert resp.status_code == 422
