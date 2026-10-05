"""Tests for trained model loading, score bounds, predictions, and submission schema.
"""

import pytest
import joblib
import pandas as pd
from src.config import MODEL_FILE, PREPROCESSOR_FILE, PREDICTIONS_FILE
from src.validate_submission import validate_submission


def test_model_artifacts_exist():
    """Checks that model and preprocessor are serialized."""
    assert MODEL_FILE.exists()
    assert PREPROCESSOR_FILE.exists()


def test_model_prediction_score_range():
    """Verifies that model generates valid probabilities between 0.0 and 1.0."""
    model = joblib.load(MODEL_FILE)
    preprocessor = joblib.load(PREPROCESSOR_FILE)

    dummy_df = pd.DataFrame([{
        "days_since_purchase": 100,
        "claim_amount_inr": 1500.0,
        "customer_prior_claims": 0,
        "partner_tenure_days": 200,
        "claim_to_price_ratio": 0.25,
        "warranty_elapsed_ratio": 0.3,
        "is_under_2000": 1,
        "partner_is_new": 1,
        "new_partner_small_claim": 1,
        "p_prior_claims": 5,
        "p_prior_frauds": 0,
        "p_prior_fraud_rate": 0.012,
        "photo_attached": "Y",
        "partner_inspected": "N",
        "partner_type": "authorised_service_centre",
        "city": "Pune",
        "family": "Air Fryer",
        "clean_claim_desc": "motor not running",
    }])

    X = preprocessor.transform(dummy_df)
    proba = model.predict_proba(X)[0, 1]
    assert 0.0 <= proba <= 1.0


def test_submission_validation_passes():
    """Ensures that the automated submission validator passes 10/10 checks."""
    assert PREDICTIONS_FILE.exists()
    assert validate_submission(PREDICTIONS_FILE) is True
