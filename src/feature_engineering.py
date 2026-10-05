"""Feature engineering pipeline for Kestrel Home warranty fraud detection.
Ensures 100% leakage-safe calculations and preserves strict temporal causality.
"""

import json
from typing import Dict, Any, Optional
import numpy as np
import pandas as pd
from pathlib import Path
from src.config import (
    NUMERICAL_FEATURES,
    CATEGORICAL_FEATURES,
    AUTO_APPROVAL_THRESHOLD_INR,
    PARTNER_STATS_FILE,
)


class PartnerHistoryTracker:
    """Maintains leakage-safe historical partner performance statistics.
    Uses Bayesian Laplace smoothing to avoid overfitting on low-volume partners.
    """

    def __init__(self, alpha: float = 10.0, global_prior: float = 0.0123):
        self.alpha = alpha
        self.global_prior = global_prior
        self.partner_counts: Dict[str, int] = {}
        self.partner_frauds: Dict[str, int] = {}

    def fit_from_training(self, train_df: pd.DataFrame) -> None:
        """Populates partner history from full resolved training set."""
        self.partner_counts = {}
        self.partner_frauds = {}

        if "is_fraud" in train_df.columns:
            valid = train_df.dropna(subset=["is_fraud"]).copy()
            self.global_prior = float(valid["is_fraud"].mean())
            grouped = valid.groupby("partner_id")["is_fraud"].agg(["count", "sum"])
            for partner_id, row in grouped.iterrows():
                self.partner_counts[str(partner_id)] = int(row["count"])
                self.partner_frauds[str(partner_id)] = int(row["sum"])

    def transform_expanding(self, sorted_df: pd.DataFrame) -> pd.DataFrame:
        """Transforms a chronologically sorted training DataFrame using expanding window.
        Prevents forward-looking leakage: each claim only observes strictly preceding cases.
        """
        counts = {}
        frauds = {}
        prior_claims = []
        prior_frauds = []
        prior_rates = []

        has_target = "is_fraud" in sorted_df.columns

        for _, row in sorted_df.iterrows():
            p_id = str(row["partner_id"])
            c = counts.get(p_id, 0)
            f = frauds.get(p_id, 0)

            prior_claims.append(c)
            prior_frauds.append(f)
            rate = (f + self.alpha * self.global_prior) / (c + self.alpha)
            prior_rates.append(rate)

            if has_target and not pd.isna(row["is_fraud"]):
                counts[p_id] = c + 1
                frauds[p_id] = f + int(row["is_fraud"])

        result_df = sorted_df.copy()
        result_df["p_prior_claims"] = prior_claims
        result_df["p_prior_frauds"] = prior_frauds
        result_df["p_prior_fraud_rate"] = prior_rates
        return result_df

    def transform_static(self, df: pd.DataFrame) -> pd.DataFrame:
        """Transforms test or inference claims using historical partner statistics
        established during the training period.
        """
        prior_claims = []
        prior_frauds = []
        prior_rates = []

        for _, row in df.iterrows():
            p_id = str(row["partner_id"])
            c = self.partner_counts.get(p_id, 0)
            f = self.partner_frauds.get(p_id, 0)

            prior_claims.append(c)
            prior_frauds.append(f)
            rate = (f + self.alpha * self.global_prior) / (c + self.alpha)
            prior_rates.append(rate)

        result_df = df.copy()
        result_df["p_prior_claims"] = prior_claims
        result_df["p_prior_frauds"] = prior_frauds
        result_df["p_prior_fraud_rate"] = prior_rates
        return result_df

    def save(self, filepath: Path = PARTNER_STATS_FILE) -> None:
        """Serializes partner stats dictionary to JSON."""
        filepath.parent.mkdir(parents=True, exist_ok=True)
        data = {
            "alpha": self.alpha,
            "global_prior": self.global_prior,
            "partner_counts": self.partner_counts,
            "partner_frauds": self.partner_frauds,
        }
        with open(filepath, "w", encoding="utf-8") as f:
            json.dump(data, f, indent=2)

    @classmethod
    def load(cls, filepath: Path = PARTNER_STATS_FILE) -> "PartnerHistoryTracker":
        """Deserializes partner stats from JSON."""
        with open(filepath, "r", encoding="utf-8") as f:
            data = json.load(f)
        tracker = cls(alpha=data["alpha"], global_prior=data["global_prior"])
        tracker.partner_counts = data["partner_counts"]
        tracker.partner_frauds = data["partner_frauds"]
        return tracker


def engineer_base_features(df: pd.DataFrame) -> pd.DataFrame:
    """Computes deterministic domain and policy features from claim, product, and partner data.

    Args:
        df: Preprocessed claims DataFrame merged with products and partners.

    Returns:
        DataFrame with added feature columns.
    """
    feat_df = df.copy()

    # Tenure of partner at the time of claim submission
    if "submitted_at" in feat_df.columns and "onboarded_date" in feat_df.columns:
        submitted = pd.to_datetime(feat_df["submitted_at"])
        onboarded = pd.to_datetime(feat_df["onboarded_date"])
        feat_df["partner_tenure_days"] = (submitted - onboarded).dt.days.clip(lower=0)
    else:
        feat_df["partner_tenure_days"] = 365

    # New partner indicator (< 365 days on platform)
    feat_df["partner_is_new"] = (feat_df["partner_tenure_days"] < 365).astype(int)

    # Claim amount relative to product list price
    if "claim_amount_inr" in feat_df.columns and "list_price_inr" in feat_df.columns:
        feat_df["claim_to_price_ratio"] = (
            feat_df["claim_amount_inr"] / feat_df["list_price_inr"]
        ).fillna(0.0)
    else:
        feat_df["claim_to_price_ratio"] = 0.30

    # Warranty duration and elapsed ratio
    if "warranty_months" in feat_df.columns:
        warranty_days = feat_df["warranty_months"] * 30.5
        feat_df["warranty_elapsed_ratio"] = (
            feat_df["days_since_purchase"] / warranty_days
        ).fillna(0.5)
    else:
        feat_df["warranty_elapsed_ratio"] = 0.5

    # Auto-approval policy rule: claims < Rs 2,000 (§5)
    if "claim_amount_inr" in feat_df.columns:
        feat_df["is_under_2000"] = (
            feat_df["claim_amount_inr"] < AUTO_APPROVAL_THRESHOLD_INR
        ).astype(int)
    else:
        feat_df["is_under_2000"] = 0

    # Interaction: newly onboarded partner submitting small auto-approved claim
    feat_df["new_partner_small_claim"] = (
        feat_df["partner_is_new"] * feat_df["is_under_2000"]
    ).astype(int)

    # Customer prior claims
    if "customer_prior_claims" not in feat_df.columns:
        feat_df["customer_prior_claims"] = 0

    return feat_df
