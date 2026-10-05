"""Prediction pipeline for unlabelled test claims.
Loads trained artifacts and generates predictions.csv in the exact sample_submission format.
"""

from pathlib import Path
import joblib
import numpy as np
import pandas as pd

from src.config import (
    MODEL_FILE,
    PARTNER_STATS_FILE,
    PREDICTIONS_FILE,
    PREPROCESSOR_FILE,
    TEST_FILE,
)
from src.data_loader import load_clean_test_data
from src.feature_engineering import PartnerHistoryTracker, engineer_base_features
from src.preprocessing import preprocess_claims


def generate_predictions(output_path: Path = PREDICTIONS_FILE) -> pd.DataFrame:
    """Generates continuous fraud probability scores for all claims in test_unlabelled.csv.

    Args:
        output_path: Destination path for the predictions CSV file.

    Returns:
        DataFrame containing claim_id and continuous fraud score.
    """
    print(f"Loading test data from {TEST_FILE}...")
    test_df = load_clean_test_data()
    raw_test = pd.read_csv(TEST_FILE)

    print("Preprocessing and engineering test features...")
    processed_df = preprocess_claims(test_df)
    feat_df = engineer_base_features(processed_df)

    # Load partner history tracker from training and transform statically
    print(f"Applying historical partner registry from {PARTNER_STATS_FILE}...")
    tracker = PartnerHistoryTracker.load(PARTNER_STATS_FILE)
    feat_df = tracker.transform_static(feat_df)

    # Load fitted preprocessor and production model
    print(f"Loading preprocessor from {PREPROCESSOR_FILE} and model from {MODEL_FILE}...")
    preprocessor = joblib.load(PREPROCESSOR_FILE)
    model = joblib.load(MODEL_FILE)

    X_test = preprocessor.transform(feat_df)
    print("Generating fraud risk probability scores...")
    fraud_scores = model.predict_proba(X_test)[:, 1]

    # Create submission dataframe matching original row order
    submission_df = pd.DataFrame({
        "claim_id": raw_test["claim_id"],
        "score": np.round(fraud_scores, 4),
    })

    output_path.parent.mkdir(parents=True, exist_ok=True)
    submission_df.to_csv(output_path, index=False)
    print(f"Successfully generated predictions at {output_path} ({len(submission_df)} rows)")

    # Print summary statistics
    print("\nPrediction Score Summary:")
    print(submission_df["score"].describe())
    print(f"Claims with score >= 0.40: {(submission_df['score'] >= 0.40).sum()} ({(submission_df['score'] >= 0.40).mean():.2%})")
    print(f"Claims with score >= 0.50: {(submission_df['score'] >= 0.50).sum()} ({(submission_df['score'] >= 0.50).mean():.2%})")

    return submission_df


if __name__ == "__main__":
    generate_predictions()
