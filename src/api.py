"""FastAPI service for real-time warranty claim fraud risk assessment.
Provides /predict, /health, and /model-info endpoints with Pydantic validation.
"""

from contextlib import asynccontextmanager
import json
from pathlib import Path
from typing import List, Optional
import joblib
import pandas as pd
from fastapi import FastAPI, HTTPException, status
from pydantic import BaseModel, Field

from src.config import (
    DEFAULT_DECISION_THRESHOLD,
    METADATA_FILE,
    MODEL_FILE,
    MODEL_VERSION,
    PARTNERS_FILE,
    PARTNER_STATS_FILE,
    PREPROCESSOR_FILE,
    PRODUCTS_FILE,
)
from src.explain import (
    determine_risk_level_and_decision,
    generate_explanation_reasons,
)
from src.feature_engineering import PartnerHistoryTracker, engineer_base_features
from src.preprocessing import preprocess_claims

# In-memory artifact cache
ARTIFACTS = {
    "model": None,
    "preprocessor": None,
    "tracker": None,
    "metadata": None,
    "partners_dict": {},
    "products_dict": {},
}


def load_artifacts():
    """Loads model, preprocessor, and master lookup tables into memory once at startup."""
    print("Loading model artifacts into memory...")
    if not MODEL_FILE.exists() or not PREPROCESSOR_FILE.exists():
        raise RuntimeError("Model artifacts missing! Please run 'python -m src.train' first.")

    ARTIFACTS["model"] = joblib.load(MODEL_FILE)
    ARTIFACTS["preprocessor"] = joblib.load(PREPROCESSOR_FILE)
    ARTIFACTS["tracker"] = PartnerHistoryTracker.load(PARTNER_STATS_FILE)

    if METADATA_FILE.exists():
        with open(METADATA_FILE, "r", encoding="utf-8") as f:
            ARTIFACTS["metadata"] = json.load(f)
    else:
        ARTIFACTS["metadata"] = {"version": MODEL_VERSION}

    # Load master lookups
    partners_df = pd.read_csv(PARTNERS_FILE)
    for _, row in partners_df.iterrows():
        ARTIFACTS["partners_dict"][str(row["partner_id"])] = {
            "city": row["city"],
            "onboarded_date": row["onboarded_date"],
            "partner_type": row["partner_type"],
        }

    products_df = pd.read_csv(PRODUCTS_FILE)
    for _, row in products_df.iterrows():
        ARTIFACTS["products_dict"][str(row["sku"])] = {
            "family": row["family"],
            "list_price_inr": float(row["list_price_inr"]),
            "warranty_months": int(row["warranty_months"]),
        }
    print("All artifacts and reference tables loaded successfully!")


@asynccontextmanager
async def lifespan(app: FastAPI):
    """Lifespan event to initialize artifacts before requests arrive."""
    load_artifacts()
    yield
    print("Shutting down API service...")


app = FastAPI(
    title="Kestrel Home — Warranty Claim Review API",
    description="Real-time ML fraud scoring and explainability service for warranty claims.",
    version=MODEL_VERSION,
    lifespan=lifespan,
)


class ClaimRequest(BaseModel):
    claim_id: str = Field(..., json_schema_extra={"example": "WC711348"})
    submitted_at: str = Field(..., json_schema_extra={"example": "2026-07-01 10:45"})
    partner_id: str = Field(..., json_schema_extra={"example": "SP3160"})
    sku: str = Field(..., json_schema_extra={"example": "KH-RV-03"})
    product_serial: Optional[str] = Field("KH222151083", json_schema_extra={"example": "KH222151083"})
    days_since_purchase: int = Field(..., ge=0, json_schema_extra={"example": 240})
    claim_amount_inr: float = Field(..., gt=0, json_schema_extra={"example": 1850.0})
    photo_attached: str = Field("Y", json_schema_extra={"example": "Y"})
    partner_inspected: str = Field("N", json_schema_extra={"example": "N"})
    claim_description: str = Field(..., json_schema_extra={"example": "motor not running"})
    inspector_note: Optional[str] = Field("", json_schema_extra={"example": ""})
    customer_prior_claims: int = Field(0, ge=0, json_schema_extra={"example": 0})
    source: Optional[str] = Field("crm", json_schema_extra={"example": "crm"})


class ClaimResponse(BaseModel):
    claim_id: str
    fraud_score: float
    risk_level: str
    decision: str
    reasons: List[str]
    model_version: str


@app.get("/health", status_code=status.HTTP_200_OK)
def health_check():
    """Health check endpoint to verify service and model availability."""
    return {
        "status": "healthy",
        "model_loaded": ARTIFACTS["model"] is not None,
        "model_version": MODEL_VERSION,
    }


@app.get("/model-info", status_code=status.HTTP_200_OK)
def model_info():
    """Returns technical metadata, validation scores, and active feature sets."""
    if ARTIFACTS["metadata"] is None:
        raise HTTPException(status_code=500, detail="Metadata not initialized")
    return ARTIFACTS["metadata"]


@app.post("/predict", response_model=ClaimResponse, status_code=status.HTTP_200_OK)
def predict_claim(claim: ClaimRequest):
    """Evaluates one warranty claim, predicts continuous fraud probability,
    and returns human-readable audit reasons.
    """
    if ARTIFACTS["model"] is None:
        raise HTTPException(status_code=503, detail="Model is not loaded")

    # 1. Prepare raw dictionary
    raw_dict = claim.model_dump()

    # 2. Lookup partner and product metadata
    p_info = ARTIFACTS["partners_dict"].get(
        str(claim.partner_id),
        {"city": "Pune", "onboarded_date": "2024-01-01", "partner_type": "authorised_service_centre"},
    )
    prod_info = ARTIFACTS["products_dict"].get(
        str(claim.sku),
        {"family": "Robot Vacuum", "list_price_inr": 20000.0, "warranty_months": 12},
    )

    row_data = {
        **raw_dict,
        "city": p_info["city"],
        "onboarded_date": p_info["onboarded_date"],
        "partner_type": p_info["partner_type"],
        "family": prod_info["family"],
        "list_price_inr": prod_info["list_price_inr"],
        "warranty_months": prod_info["warranty_months"],
    }

    df_single = pd.DataFrame([row_data])

    # 3. Clean and feature engineer
    df_clean = preprocess_claims(df_single)
    df_feat = engineer_base_features(df_clean)
    df_transformed = ARTIFACTS["tracker"].transform_static(df_feat)

    # 4. Transform with preprocessor and predict probability
    X_input = ARTIFACTS["preprocessor"].transform(df_transformed)
    fraud_score = float(ARTIFACTS["model"].predict_proba(X_input)[0, 1])
    fraud_score = round(fraud_score, 4)

    # 5. Extract explainability features
    feat_dict = df_transformed.iloc[0].to_dict()
    status_info = determine_risk_level_and_decision(fraud_score, DEFAULT_DECISION_THRESHOLD)
    reasons = generate_explanation_reasons(raw_dict, feat_dict, fraud_score, DEFAULT_DECISION_THRESHOLD)

    return ClaimResponse(
        claim_id=claim.claim_id,
        fraud_score=fraud_score,
        risk_level=status_info["risk_level"],
        decision=status_info["decision"],
        reasons=reasons,
        model_version=MODEL_VERSION,
    )
