"""Test client script demonstrating end-to-end API inference and explainability.
Uses FastAPI TestClient to test without requiring an active external web server.
"""

import json
from pathlib import Path
from fastapi.testclient import TestClient
from src.api import app
from src.config import EXAMPLES_DIR


def run_api_demonstration():
    """Runs a complete test of the /health, /model-info, and /predict endpoints."""
    print("=" * 60)
    print("Testing Kestrel Warranty Fraud Review API Endpoints")
    print("=" * 60)

    with TestClient(app) as client:
        # 1. Test /health
        print("\n1. Testing GET /health...")
        health_resp = client.get("/health")
        print(f"Status: {health_resp.status_code}")
        print("Response:", json.dumps(health_resp.json(), indent=2))
        assert health_resp.status_code == 200

        # 2. Test /model-info
        print("\n2. Testing GET /model-info...")
        info_resp = client.get("/model-info")
        print(f"Status: {info_resp.status_code}")
        info_data = info_resp.json()
        print(f"Model Type: {info_data.get('model_type')}")
        print(f"Model Version: {info_data.get('model_version')}")
        print(f"Decision Threshold: {info_data.get('decision_threshold')}")
        assert info_resp.status_code == 200

        # 3. Test POST /predict with high-risk sample
        print("\n3. Testing POST /predict (High-Risk Sample)...")
        sample_path = EXAMPLES_DIR / "sample_claim.json"
        with open(sample_path, "r", encoding="utf-8") as f:
            sample_payload = json.load(f)

        pred_resp = client.post("/predict", json=sample_payload)
        print(f"Status: {pred_resp.status_code}")
        pred_data = pred_resp.json()
        print("Prediction Result:")
        print(json.dumps(pred_data, indent=2))
        assert pred_resp.status_code == 200
        assert "fraud_score" in pred_data
        assert "risk_level" in pred_data
        assert "reasons" in pred_data

        # 4. Test POST /predict with low-risk sample
        print("\n4. Testing POST /predict (Low-Risk Sample)...")
        low_risk_payload = {
            "claim_id": "WC799999",
            "submitted_at": "2026-07-05 14:20",
            "partner_id": "SP3022",
            "sku": "KH-CF-01",
            "product_serial": "KH998877665",
            "days_since_purchase": 120,
            "claim_amount_inr": 450.0,
            "photo_attached": "Y",
            "partner_inspected": "Y",
            "claim_description": "motor not running",
            "inspector_note": "Unit inspected, fault confirmed",
            "customer_prior_claims": 0,
            "source": "crm",
        }
        low_resp = client.post("/predict", json=low_risk_payload)
        print(f"Status: {low_resp.status_code}")
        low_data = low_resp.json()
        print("Prediction Result:")
        print(json.dumps(low_data, indent=2))
        assert low_resp.status_code == 200
        assert low_data["risk_level"] == "LOW"
        assert low_data["decision"] == "APPROVE"

    print("\n" + "=" * 60)
    print("ALL API ENDPOINT TESTS PASSED SUCCESSFULLY!")
    print("=" * 60)


if __name__ == "__main__":
    run_api_demonstration()
