"""Data loading and ingestion utilities for Kestrel Home warranty claims.
"""

import pandas as pd
from typing import Tuple
from src.config import TRAIN_FILE, TEST_FILE, PARTNERS_FILE, PRODUCTS_FILE


def load_raw_data() -> Tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame, pd.DataFrame]:
    """Loads raw CSV files from disk.

    Returns:
        Tuple of (train_df, test_df, partners_df, products_df)
    """
    train_df = pd.read_csv(TRAIN_FILE)
    test_df = pd.read_csv(TEST_FILE)
    partners_df = pd.read_csv(PARTNERS_FILE)
    products_df = pd.read_csv(PRODUCTS_FILE)
    return train_df, test_df, partners_df, products_df


def load_clean_training_data() -> pd.DataFrame:
    """Loads and joins training claims with reference tables.
    Excludes undecided cases where `is_fraud` is NaN per Tanmay's email and policy.

    Returns:
        Resolved training DataFrame sorted chronologically by submitted_at.
    """
    train_df, _, partners_df, products_df = load_raw_data()

    # Filter out unresolved / undecided claims (NaN)
    resolved_df = train_df.dropna(subset=["is_fraud"]).copy()
    resolved_df["is_fraud"] = resolved_df["is_fraud"].astype(int)

    # Convert timestamps
    resolved_df["submitted_at"] = pd.to_datetime(resolved_df["submitted_at"])
    partners_df["onboarded_date"] = pd.to_datetime(partners_df["onboarded_date"])

    # Merges
    merged_df = resolved_df.merge(products_df, on="sku", how="left")
    merged_df = merged_df.merge(partners_df, on="partner_id", how="left")

    # Sort chronologically to preserve strict temporal order
    merged_df = merged_df.sort_values("submitted_at").reset_index(drop=True)
    return merged_df


def load_clean_test_data() -> pd.DataFrame:
    """Loads and joins unlabelled test claims with reference tables.

    Returns:
        Test DataFrame ready for feature engineering and prediction.
    """
    _, test_df, partners_df, products_df = load_raw_data()

    # Convert timestamps
    test_df["submitted_at"] = pd.to_datetime(test_df["submitted_at"])
    partners_df["onboarded_date"] = pd.to_datetime(partners_df["onboarded_date"])

    # Merges
    merged_df = test_df.merge(products_df, on="sku", how="left")
    merged_df = merged_df.merge(partners_df, on="partner_id", how="left")
    return merged_df
