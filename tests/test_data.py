"""Tests for data loading, merging, and integrity.
"""

import pytest
import pandas as pd
from src.config import TRAIN_FILE, TEST_FILE, PARTNERS_FILE, PRODUCTS_FILE
from src.data_loader import load_raw_data, load_clean_training_data, load_clean_test_data


def test_raw_files_exist():
    """Verifies that all raw files exist on disk."""
    assert TRAIN_FILE.exists()
    assert TEST_FILE.exists()
    assert PARTNERS_FILE.exists()
    assert PRODUCTS_FILE.exists()


def test_load_raw_data_dimensions():
    """Checks raw data dimensions and shapes."""
    train_df, test_df, partners_df, products_df = load_raw_data()
    assert len(train_df) == 12029
    assert len(test_df) == 2252
    assert len(partners_df) == 380
    assert len(products_df) == 21


def test_clean_training_data_no_nan_targets():
    """Verifies undecided claims (NaN is_fraud) are filtered out from training data."""
    train_clean = load_clean_training_data()
    assert train_clean["is_fraud"].isnull().sum() == 0
    assert len(train_clean) == 11814
    assert set(train_clean["is_fraud"].unique()) == {0, 1}


def test_clean_test_data_integrity():
    """Verifies test data joins and rows."""
    test_clean = load_clean_test_data()
    assert len(test_clean) == 2252
    assert "family" in test_clean.columns
    assert "city" in test_clean.columns
    assert "partner_type" in test_clean.columns
