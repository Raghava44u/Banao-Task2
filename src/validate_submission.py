"""Automated validation script for submission predictions.
Verifies row count, schema, claim_id integrity, missingness, score range, and format against sample_submission.csv.
"""

import sys
from pathlib import Path
import pandas as pd
from src.config import PREDICTIONS_FILE, SAMPLE_SUBMISSION_FILE, TEST_FILE


def validate_submission(pred_path: Path = PREDICTIONS_FILE) -> bool:
    """Validates predictions.csv against requirements and sample_submission.csv.

    Args:
        pred_path: Path to predictions.csv.

    Returns:
        True if all validation checks pass, False otherwise.
    """
    print("=" * 60)
    print("Kestrel Warranty Claim Review — Submission Validation")
    print("=" * 60)

    # 1. Check file exists
    if not pred_path.exists():
        print(f"[FAIL] Predictions file not found at {pred_path}")
        return False
    print(f"[PASS] Predictions file exists at {pred_path}")

    # 2. Load files
    pred_df = pd.read_csv(pred_path)
    sample_df = pd.read_csv(SAMPLE_SUBMISSION_FILE)
    test_df = pd.read_csv(TEST_FILE)

    # 3. Check Row Count
    expected_rows = len(test_df)
    actual_rows = len(pred_df)
    if actual_rows != expected_rows:
        print(
            f"[FAIL] Row count mismatch: expected {expected_rows}, got {actual_rows}"
        )
        return False
    print(
        f"[PASS] Row count matches test set exactly: {actual_rows} rows (sample_submission: {len(sample_df)})"
    )

    # 4. Check Column Names
    expected_cols = ["claim_id", "score"]
    if list(pred_df.columns) != expected_cols:
        print(
            f"[FAIL] Column mismatch: expected {expected_cols}, got {list(pred_df.columns)}"
        )
        return False
    print(f"[PASS] Column names match exactly: {expected_cols}")

    # 5. Check Missing Values
    null_counts = pred_df.isnull().sum()
    if null_counts.sum() > 0:
        print(f"[FAIL] Missing values found:\n{null_counts}")
        return False
    print("[PASS] Zero missing or null values in submission")

    # 6. Check Duplicate Claim IDs
    dup_count = pred_df["claim_id"].duplicated().sum()
    if dup_count > 0:
        print(f"[FAIL] Duplicate claim IDs found: {dup_count}")
        return False
    print("[PASS] Zero duplicate claim IDs")

    # 7. Check Claim ID Alignment with Test Set
    if not pred_df["claim_id"].equals(test_df["claim_id"]):
        # Check set equality
        if set(pred_df["claim_id"]) != set(test_df["claim_id"]):
            missing_ids = set(test_df["claim_id"]) - set(pred_df["claim_id"])
            extra_ids = set(pred_df["claim_id"]) - set(test_df["claim_id"])
            print(
                f"[FAIL] Claim ID mismatch! Missing: {len(missing_ids)}, Extra: {len(extra_ids)}"
            )
            return False
        else:
            print("[WARN] Claim IDs match set but have different row ordering.")
    else:
        print("[PASS] Claim IDs match test set exactly 1-to-1 in identical sequence")

    # 8. Check Score Data Type & Numeric Validity
    if not pd.api.types.is_numeric_dtype(pred_df["score"]):
        print("[FAIL] 'score' column is not numeric")
        return False
    print(f"[PASS] 'score' column is valid numeric dtype ({pred_df['score'].dtype})")

    # 9. Check Score Bounds [0.0, 1.0]
    min_score = pred_df["score"].min()
    max_score = pred_df["score"].max()
    if min_score < 0.0 or max_score > 1.0:
        print(
            f"[FAIL] Score values outside [0.0, 1.0] range: min={min_score}, max={max_score}"
        )
        return False
    print(f"[PASS] Score values bounded in [0.0, 1.0]: min={min_score:.4f}, max={max_score:.4f}")

    # 10. Check Non-Trivial Score Spread
    std_score = pred_df["score"].std()
    if std_score == 0 or pd.isna(std_score):
        print("[FAIL] Scores are degenerate/constant with zero variance")
        return False
    print(f"[PASS] Scores show active variance (mean={pred_df['score'].mean():.4f}, std={std_score:.4f})")

    print("\n" + "=" * 60)
    print("ALL SUBMISSION VALIDATION CHECKS PASSED (10/10)!")
    print("=" * 60)
    return True


if __name__ == "__main__":
    success = validate_submission()
    sys.exit(0 if success else 1)
