import json
import sys
from datetime import datetime
from pathlib import Path
import requests
import streamlit as st
import pandas as pd

# Ensure project root is in sys.path for robust module imports
PROJECT_ROOT = Path(__file__).resolve().parent.parent
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from src.config import (
    CANONICAL_FAULT_DESCRIPTIONS,
    DEFAULT_DECISION_THRESHOLD,
    MODEL_VERSION,
    PARTNERS_FILE,
    PRODUCTS_FILE,
)

# Page configuration
st.set_page_config(
    page_title="Kestrel Home — Warranty Claim Review",
    page_icon="🛡️",
    layout="wide",
    initial_sidebar_state="expanded",
)

# Custom Styling for polished enterprise dashboard
st.markdown(
    """
    <style>
    .main-header {
        font-size: 2.1rem;
        font-weight: 700;
        color: #1a2a3a;
        margin-bottom: 0.1rem;
    }
    .sub-header {
        font-size: 1.1rem;
        color: #4a5568;
        margin-bottom: 1.5rem;
    }
    .card {
        background-color: #f8fafc;
        border-radius: 10px;
        padding: 1.2rem;
        border: 1px solid #e2e8f0;
        margin-bottom: 1rem;
    }
    .metric-value {
        font-size: 1.8rem;
        font-weight: 700;
    }
    .badge-low {
        background-color: #def7ec;
        color: #03543f;
        padding: 0.35rem 0.8rem;
        border-radius: 20px;
        font-weight: 600;
        display: inline-block;
    }
    .badge-medium {
        background-color: #fef08a;
        color: #713f12;
        padding: 0.35rem 0.8rem;
        border-radius: 20px;
        font-weight: 600;
        display: inline-block;
    }
    .badge-high {
        background-color: #fde8e8;
        color: #9b1c1c;
        padding: 0.35rem 0.8rem;
        border-radius: 20px;
        font-weight: 600;
        display: inline-block;
    }
    </style>
    """,
    unsafe_allow_html=True,
)


@st.cache_data
def load_reference_options():
    """Loads partner IDs and SKU options for form dropdowns."""
    partners_df = pd.read_csv(PARTNERS_FILE)
    products_df = pd.read_csv(PRODUCTS_FILE)
    return partners_df, products_df


partners_df, products_df = load_reference_options()
partner_list = sorted(partners_df["partner_id"].unique())
sku_list = sorted(products_df["sku"].unique())

# API Configuration
API_URL = "http://127.0.0.1:8000"

# Sidebar: Operational Controls & Metadata
with st.sidebar:
    st.image(
        "https://raw.githubusercontent.com/feathericons/feather/master/icons/shield.svg",
        width=48,
    )
    st.title("Operations Desk")
    st.markdown(f"**Model Version:** `{MODEL_VERSION}`")
    st.markdown(f"**Review Threshold:** `{DEFAULT_DECISION_THRESHOLD:.2f}`")
    st.markdown("**Desk Capacity:** `40 claims / month`")

    st.markdown("---")
    st.subheader("System Status")

    # Check API health
    api_available = False
    try:
        health_res = requests.get(f"{API_URL}/health", timeout=1.0)
        if health_res.status_code == 200:
            st.success("FastAPI Service: ONLINE")
            api_available = True
        else:
            st.warning("FastAPI Service: UNHEALTHY")
    except Exception:
        st.error("FastAPI Service: OFFLINE")
        st.caption("Start with: `uvicorn src.api:app --reload`")

    st.markdown("---")
    st.caption("Kestrel Operations Policy v4.1 Compliant")
    st.caption("Confidential — Internal Use Only")

# Main Page Header
st.markdown(
    '<div class="main-header">Kestrel Home — Warranty Claim Review</div>',
    unsafe_allow_html=True,
)
st.markdown(
    '<div class="sub-header">Fraud Risk Assessment & Operations Triage Engine</div>',
    unsafe_allow_html=True,
)

# Section 1: Claim Assessment Input Form
st.subheader("1. Claim Assessment")

with st.form("claim_review_form"):
    col1, col2, col3 = st.columns(3)

    with col1:
        claim_id = st.text_input("Claim ID", value="WC711348")
        partner_id = st.selectbox("Service Partner ID", options=partner_list, index=partner_list.index("SP3160") if "SP3160" in partner_list else 0)
        sku = st.selectbox("Product SKU", options=sku_list, index=sku_list.index("KH-RV-03") if "KH-RV-03" in sku_list else 0)
        claim_description = st.selectbox("Fault Description", options=CANONICAL_FAULT_DESCRIPTIONS, index=1)

    with col2:
        claim_amount = st.number_input("Claim Amount (₹ INR)", min_value=100.0, max_value=50000.0, value=1850.0, step=50.0)
        days_since_purchase = st.number_input("Days Since Purchase", min_value=0, max_value=1000, value=240, step=5)
        customer_prior_claims = st.number_input("Customer Prior Claims", min_value=0, max_value=10, value=2, step=1)
        product_serial = st.text_input("Product Serial", value="KH222151083")

    with col3:
        submitted_date = st.date_input("Claim Submission Date", value=datetime.strptime("2026-07-01", "%Y-%m-%d"))
        photo_attached = st.radio("Photo Attached?", ["Y", "N"], index=0, horizontal=True)
        partner_inspected = st.radio("Partner Inspected?", ["Y", "N"], index=1, horizontal=True)
        inspector_note = st.text_input("Inspector Note (Optional)", value="")

    submitted = st.form_submit_button("Assess Fraud Risk", type="primary", use_container_width=True)

# Process Assessment
if submitted:
    payload = {
        "claim_id": claim_id,
        "submitted_at": f"{submitted_date.strftime('%Y-%m-%d')} 10:00",
        "partner_id": partner_id,
        "sku": sku,
        "product_serial": product_serial,
        "days_since_purchase": int(days_since_purchase),
        "claim_amount_inr": float(claim_amount),
        "photo_attached": photo_attached,
        "partner_inspected": partner_inspected,
        "claim_description": claim_description,
        "inspector_note": inspector_note,
        "customer_prior_claims": int(customer_prior_claims),
        "source": "crm",
    }

    result = None

    if api_available:
        try:
            resp = requests.post(f"{API_URL}/predict", json=payload, timeout=3.0)
            if resp.status_code == 200:
                result = resp.json()
            else:
                st.error(f"API Error ({resp.status_code}): {resp.text}")
        except Exception as e:
            st.warning(f"Connection failed: {e}. Falling back to local engine...")

    # Fallback to local inference if API is not running
    if result is None:
        try:
            from src.api import predict_claim, ClaimRequest
            req_obj = ClaimRequest(**payload)
            resp_obj = predict_claim(req_obj)
            result = resp_obj.model_dump()
        except Exception as e:
            st.error(f"Inference error: {e}")

    if result:
        st.markdown("---")
        # Section 2: Risk Result
        st.subheader("2. Risk Assessment Result")
        score = result["fraud_score"]
        level = result["risk_level"]
        decision = result["decision"]

        res_col1, res_col2, res_col3 = st.columns(3)

        with res_col1:
            st.metric("Fraud Risk Score", f"{score:.1%}")

        with res_col2:
            st.markdown("**Risk Level:**")
            if level == "HIGH":
                st.markdown(f'<div class="badge-high">HIGH RISK</div>', unsafe_allow_html=True)
            elif level == "MEDIUM":
                st.markdown(f'<div class="badge-medium">MEDIUM RISK</div>', unsafe_allow_html=True)
            else:
                st.markdown(f'<div class="badge-low">LOW RISK</div>', unsafe_allow_html=True)

        with res_col3:
            st.markdown("**Recommended Action:**")
            if decision == "MANUAL REVIEW":
                st.markdown("🚨 **HOLD FOR MANUAL REVIEW** (Send to Investigation Desk)")
            elif decision == "REVIEW":
                st.markdown("⚠️ **SECONDARY VERIFICATION** (Request additional documentation)")
            else:
                st.markdown("✅ **APPROVE & PROCESS PAYOUT** (Normal Auto-Processing)")

        # Section 3: Why this claim was flagged
        st.markdown("---")
        st.subheader("3. Why this claim was flagged / verified")
        for reason in result.get("reasons", []):
            st.markdown(f"- {reason}")

        # Section 4: Claim Details
        st.markdown("---")
        with st.expander("4. Submitted Claim Context & Reference Details", expanded=False):
            detail_col1, detail_col2 = st.columns(2)
            with detail_col1:
                st.json({
                    "Claim ID": claim_id,
                    "Partner ID": partner_id,
                    "Product SKU": sku,
                    "Claim Amount INR": f"₹{claim_amount:,.0f}",
                    "Days Since Purchase": days_since_purchase,
                    "Customer Prior Claims": customer_prior_claims,
                })
            with detail_col2:
                st.json({
                    "Photo Attached": photo_attached,
                    "Partner Inspected": partner_inspected,
                    "Fault Description": claim_description,
                    "Serial Number": product_serial,
                    "Model Version": result.get("model_version"),
                })

# Section 5: Model Information & Governance
st.markdown("---")
st.subheader("5. Model Governance & Performance Metrics")
gov_col1, gov_col2, gov_col3, gov_col4, gov_col5 = st.columns(5)
gov_col1.metric("Validation Accuracy", "97.52%", "Exceeds 97% KPI")
gov_col2.metric("Precision", "58.82%", "Holdout Precision")
gov_col3.metric("Recall", "45.45%", "Fraud Captured")
gov_col4.metric("F1 Score", "0.5128", "Harmonic Mean")
gov_col5.metric("Decision Threshold", f"{DEFAULT_DECISION_THRESHOLD:.2f}", "Optimized")

st.info(
    "⚠️ **Operational Disclaimer**: This tool supports review decisions and audit triage. "
    "It does not automatically reject claims and does not replace human investigation or warranty policy checks."
)
