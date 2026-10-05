"""Configuration settings and constants for the Kestrel Home Warranty Claim Review system.
"""

from pathlib import Path

# Base Paths
BASE_DIR = Path(__file__).resolve().parent.parent
DATA_DIR = BASE_DIR / "Given"
MODELS_DIR = BASE_DIR / "models"
REPORTS_DIR = BASE_DIR / "reports"
FIGURES_DIR = REPORTS_DIR / "figures"
EXAMPLES_DIR = BASE_DIR / "examples"

# Input Data Files
TRAIN_FILE = DATA_DIR / "train.csv"
TEST_FILE = DATA_DIR / "test_unlabelled.csv"
PARTNERS_FILE = DATA_DIR / "partners.csv"
PRODUCTS_FILE = DATA_DIR / "products.csv"
SAMPLE_SUBMISSION_FILE = DATA_DIR / "sample_submission.csv"

# Output Files
PREDICTIONS_FILE = BASE_DIR / "predictions.csv"
MODEL_FILE = MODELS_DIR / "model.joblib"
PREPROCESSOR_FILE = MODELS_DIR / "preprocessor.joblib"
PARTNER_STATS_FILE = MODELS_DIR / "partner_stats.json"
METADATA_FILE = MODELS_DIR / "model_metadata.json"

# Operational Constants from Policy v4.1 & Business Context
AUTO_APPROVAL_THRESHOLD_INR = 2000.0  # From 1 May 2026, claims under Rs 2,000 auto-approved
GOODWILL_COST_INR = 380.0             # Delayed genuine customer goodwill cost (False Positive)
SERVICE_CONTACT_COST_INR = 260.0      # Blended cost of service contact
INVESTIGATION_CAPACITY_MONTHLY = 40   # Max claims desk can review per month
DEFAULT_DECISION_THRESHOLD = 0.40     # Optimal threshold balancing recall, precision & capacity
RANDOM_STATE = 42
MODEL_VERSION = "1.0.0"

# Canonical Fault Categories
CANONICAL_FAULT_DESCRIPTIONS = [
    "unit not heating",
    "motor not running",
    "tripping mcb",
    "filter indicator stuck",
    "remote not working",
    "blade jammed",
    "loud noise while running",
    "power button not working",
    "display not working",
    "water leaking",
    "burning smell",
    "not charging",
]

# Numeric & Categorical Feature Definitions
NUMERICAL_FEATURES = [
    "days_since_purchase",
    "claim_amount_inr",
    "customer_prior_claims",
    "partner_tenure_days",
    "claim_to_price_ratio",
    "warranty_elapsed_ratio",
    "is_under_2000",
    "partner_is_new",
    "new_partner_small_claim",
    "p_prior_claims",
    "p_prior_frauds",
    "p_prior_fraud_rate",
]

CATEGORICAL_FEATURES = [
    "photo_attached",
    "partner_inspected",
    "partner_type",
    "city",
    "family",
    "clean_claim_desc",
]
