"""Model training pipeline for Kestrel Home warranty fraud classification.
Trains multiple candidate models, evaluates on temporal validation, and saves production artifacts.
"""

import json
from datetime import datetime
import joblib
import numpy as np
import pandas as pd
from sklearn.compose import ColumnTransformer
from sklearn.ensemble import (
    GradientBoostingClassifier,
    HistGradientBoostingClassifier,
    RandomForestClassifier,
)
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import (
    accuracy_score,
    average_precision_score,
    confusion_matrix,
    f1_score,
    precision_score,
    recall_score,
    roc_auc_score,
)
from sklearn.preprocessing import OneHotEncoder, StandardScaler

from src.config import (
    CATEGORICAL_FEATURES,
    DEFAULT_DECISION_THRESHOLD,
    METADATA_FILE,
    MODEL_FILE,
    MODEL_VERSION,
    MODELS_DIR,
    NUMERICAL_FEATURES,
    PARTNER_STATS_FILE,
    PREPROCESSOR_FILE,
    RANDOM_STATE,
)
from src.data_loader import load_clean_training_data
from src.feature_engineering import PartnerHistoryTracker, engineer_base_features
from src.preprocessing import preprocess_claims


def build_preprocessor() -> ColumnTransformer:
    """Builds the scikit-learn ColumnTransformer for numerical and categorical features."""
    return ColumnTransformer(
        transformers=[
            ("num", StandardScaler(), NUMERICAL_FEATURES),
            (
                "cat",
                OneHotEncoder(handle_unknown="ignore", sparse_output=False),
                CATEGORICAL_FEATURES,
            ),
        ]
    )


def train_and_evaluate_candidates():
    """Runs end-to-end model training, temporal evaluation, and artifact serialization."""
    print("=" * 60)
    print("Kestrel Home Warranty Fraud Detection — Training Pipeline")
    print("=" * 60)

    # 1. Load clean resolved claims
    print("\n[1/6] Loading resolved historical claims...")
    df = load_clean_training_data()
    print(f"Total resolved claims: {len(df)}")
    print(f"Total historical frauds: {df['is_fraud'].sum()} ({df['is_fraud'].mean():.2%})")

    # 2. Preprocess
    print("\n[2/6] Preprocessing and sanitizing claims...")
    df = preprocess_claims(df)

    # 3. Engineer features
    print("\n[3/6] Engineering policy and domain features...")
    df = engineer_base_features(df)

    # 4. Expanding partner tracker (leakage-safe)
    print("\n[4/6] Computing leakage-safe expanding partner history...")
    tracker = PartnerHistoryTracker(alpha=10.0)
    df = tracker.transform_expanding(df)

    # Save full historical tracker for test and live API inference
    tracker.fit_from_training(df)
    tracker.save(PARTNER_STATS_FILE)
    print(f"Partner history registry saved to {PARTNER_STATS_FILE}")

    # 5. Temporal Validation (Holdout: June 2026)
    print("\n[5/6] Performing Temporal Validation (Holdout: June 2026)...")
    val_cutoff = "2026-06-01"
    train_mask = df["submitted_at"] < val_cutoff
    val_mask = df["submitted_at"] >= val_cutoff

    train_df = df[train_mask].copy()
    val_df = df[val_mask].copy()

    print(f"Training split (Apr 2025 – May 2026): {len(train_df)} rows, {train_df['is_fraud'].sum()} frauds")
    print(f"Validation split (June 2026 holdout): {len(val_df)} rows, {val_df['is_fraud'].sum()} frauds")

    preprocessor = build_preprocessor()
    X_train = preprocessor.fit_transform(train_df)
    y_train = train_df["is_fraud"].values
    X_val = preprocessor.transform(val_df)
    y_val = val_df["is_fraud"].values

    candidates = {
        "Logistic Regression": LogisticRegression(
            class_weight="balanced", max_iter=1000, random_state=RANDOM_STATE
        ),
        "Random Forest": RandomForestClassifier(
            n_estimators=150,
            class_weight="balanced",
            max_depth=6,
            random_state=RANDOM_STATE,
        ),
        "HistGradientBoosting": HistGradientBoostingClassifier(
            max_iter=100,
            max_leaf_nodes=15,
            class_weight="balanced",
            random_state=RANDOM_STATE,
        ),
        "Gradient Boosting": GradientBoostingClassifier(
            n_estimators=100, max_depth=3, random_state=RANDOM_STATE
        ),
    }

    eval_results = {}
    print("\nCandidate Model Performance on Temporal Validation:")
    print("-" * 80)
    print(
        f"{'Model':<24} | {'ROC-AUC':<8} | {'PR-AUC':<8} | {'Acc@0.40':<8} | {'Prec@0.40':<8} | {'Rec@0.40':<8} | {'F1@0.40':<8}"
    )
    print("-" * 80)

    for name, model in candidates.items():
        model.fit(X_train, y_train)
        probs = model.predict_proba(X_val)[:, 1]
        preds = (probs >= DEFAULT_DECISION_THRESHOLD).astype(int)

        roc = roc_auc_score(y_val, probs)
        pr = average_precision_score(y_val, probs)
        acc = accuracy_score(y_val, preds)
        prec = precision_score(y_val, preds, zero_division=0)
        rec = recall_score(y_val, preds, zero_division=0)
        f1 = f1_score(y_val, preds, zero_division=0)

        eval_results[name] = {
            "roc_auc": float(roc),
            "pr_auc": float(pr),
            "accuracy": float(acc),
            "precision": float(prec),
            "recall": float(rec),
            "f1": float(f1),
        }
        print(
            f"{name:<24} | {roc:<8.4f} | {pr:<8.4f} | {acc:<8.4f} | {prec:<8.4f} | {rec:<8.4f} | {f1:<8.4f}"
        )

    # 6. Fit Production Model & Save Artifacts
    print("\n[6/6] Fitting Production Model (Gradient Boosting Classifier)...")
    production_model = GradientBoostingClassifier(
        n_estimators=100, max_depth=3, random_state=RANDOM_STATE
    )

    # Fit preprocessor and production model on full training set
    full_preprocessor = build_preprocessor()
    X_full = full_preprocessor.fit_transform(df)
    y_full = df["is_fraud"].values
    production_model.fit(X_full, y_full)

    # Save artifacts
    MODELS_DIR.mkdir(parents=True, exist_ok=True)
    joblib.dump(production_model, MODEL_FILE)
    joblib.dump(full_preprocessor, PREPROCESSOR_FILE)
    print(f"Model saved to {MODEL_FILE}")
    print(f"Preprocessor saved to {PREPROCESSOR_FILE}")

    # Best model validation metrics
    best_metrics = eval_results["Gradient Boosting"]
    metadata = {
        "model_version": MODEL_VERSION,
        "model_type": "GradientBoostingClassifier",
        "training_timestamp": datetime.now().isoformat(),
        "random_state": RANDOM_STATE,
        "decision_threshold": DEFAULT_DECISION_THRESHOLD,
        "validation_strategy": "Temporal Holdout (June 2026)",
        "validation_metrics": best_metrics,
        "all_candidate_metrics": eval_results,
        "features": {
            "numerical": NUMERICAL_FEATURES,
            "categorical": CATEGORICAL_FEATURES,
        },
    }

    with open(METADATA_FILE, "w", encoding="utf-8") as f:
        json.dump(metadata, f, indent=2)
    print(f"Metadata saved to {METADATA_FILE}")
    print("\nTraining completed successfully!")


if __name__ == "__main__":
    train_and_evaluate_candidates()
