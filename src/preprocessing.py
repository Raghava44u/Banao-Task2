"""Data preprocessing, cleaning, and sanitization routines.
"""

import pandas as pd
from src.config import CANONICAL_FAULT_DESCRIPTIONS


def clean_serial(serial_str: str) -> str:
    """Normalizes technician-typed serial number strings.
    Strips whitespace, converts to uppercase, and removes hyphens.
    """
    if pd.isna(serial_str):
        return "UNKNOWN_SERIAL"
    return str(serial_str).strip().upper().replace("-", "").replace(" ", "")


def clean_claim_description(text: str) -> str:
    """Sanitizes partner fault descriptions and maps them to canonical categories.
    Treats adversarial text injections as inert data by matching canonical fault prefixes.
    """
    if pd.isna(text):
        return "other"
    lower_text = str(text).lower()
    for canonical in CANONICAL_FAULT_DESCRIPTIONS:
        if canonical in lower_text:
            return canonical
    return "other"


def preprocess_claims(df: pd.DataFrame) -> pd.DataFrame:
    """Applies standard cleaning and hygiene transformations to claims DataFrame.

    Args:
        df: Merged claims DataFrame.

    Returns:
        DataFrame with standardized text and sanitized fields.
    """
    clean_df = df.copy()

    # Clean serial
    if "product_serial" in clean_df.columns:
        clean_df["clean_serial"] = clean_df["product_serial"].apply(clean_serial)

    # Clean claim description
    if "claim_description" in clean_df.columns:
        clean_df["clean_claim_desc"] = clean_df["claim_description"].apply(clean_claim_description)
    else:
        clean_df["clean_claim_desc"] = "other"

    # Normalize categorical Y/N flags
    for col in ["photo_attached", "partner_inspected"]:
        if col in clean_df.columns:
            clean_df[col] = clean_df[col].fillna("N").astype(str).str.upper()

    # Handle missing inspector notes
    if "inspector_note" in clean_df.columns:
        clean_df["inspector_note"] = clean_df["inspector_note"].fillna("NO_INSPECTION_NOTE")

    return clean_df
