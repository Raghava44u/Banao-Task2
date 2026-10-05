# Kestrel Home — Warranty Claim Review System

[![Python 3.10+](https://img.shields.io/badge/python-3.10+-blue.svg)](https://www.python.org/downloads/)
[![FastAPI](https://img.shields.io/badge/FastAPI-0.110+-green.svg)](https://fastapi.tiangolo.com)
[![Streamlit](https://img.shields.io/badge/Streamlit-1.32+-red.svg)](https://streamlit.io)
[![Scikit-Learn](https://img.shields.io/badge/scikit--learn-1.4+-orange.svg)](https://scikit-learn.org)
[![Tests Passing](https://img.shields.io/badge/pytest-15%2F15%20passed-brightgreen.svg)]()

> **Production Warranty Fraud Classification, Operational Triage, and Explainability Engine**  
> Tailored for Kestrel Home Appliances (Pune, Maharashtra, India).

---

## 1. Overview
Kestrel Home Appliances manufactures and distributes consumer appliances (Air Fryers, Mixer Grinders, Water Purifiers, Robot Vacuums, Induction Cooktops, Fans, and Room Heaters) across India via an extensive network of approximately 380 authorized service centers, franchises, and freelance technicians.

This repository provides an end-to-end, reproducible machine learning system designed to detect fraudulent warranty claims **before reimbursement payout occurs**. It delivers real-time fraud scoring, human-readable operational audit explanations, a high-throughput FastAPI service, and a Streamlit operations review desk.

---

## 2. Business Problem & Operational Reality
- **Target KPI Paradox**: Executive management requested overall accuracy above **97%**. However, because baseline fraud is heavily imbalanced (**1.23%** across historical claims), a naive model predicting zero fraud trivially achieves **98.77% accuracy** while catching ₹0 in fraud. Our system surpasses the 97% requirement (**97.52% holdout accuracy**) while actively intercepting **45.5% to 68.2% of all fraud cases**.
- **Investigation Desk Capacity**: Per Kestrel Operations Policy v4.1 (§5) and Finance direction (Farhan Sheikh), the claims review desk can investigate at most **40 claims per month**.
- **Cost Tradeoff**:
  - Genuine claim held for audit delays repair: Kestrel pays **₹380 goodwill compensation** (§4).
  - Fraudulent claim paid out costs the full claim amount (average **₹1,868 – ₹4,449**, up to ₹21,148).
- **The May 1 Policy Loophole**: From 1 May 2026, claims under ₹2,000 were auto-approved without physical inspection. Fraudsters immediately exploited this, causing small claims to explode—post-May, **97.3% of all fraud claims** were concentrated just below ₹2,000.

---

## 3. High-Level Solution
The solution addresses these challenges through:
1. **Leakage-Safe Temporal Pipeline**: Chronological expanding window calculating Bayesian Laplace-smoothed partner fraud rates without lookahead leakage.
2. **Policy-Aware Feature Engineering**: Explicit modeling of the ₹2,000 auto-approval loophole, partner tenure, claim-to-list-price ratios, and repeat claimant patterns.
3. **Capacity-Constrained Optimization**: Prioritizing high-risk audits to maximize Net Rupee Benefit under the 40-claims/month review cap.
4. **Transparent Explainability**: Plain-English audit rationales (3–4 bullet points) translating technical tree splits into operational guidance.
5. **Zero External Paid API Dependency**: Built entirely on standard local Python tools (Scikit-Learn, FastAPI, Streamlit) with ₹0 runtime cost.

---

## 4. Architecture

```
[ Incoming Warranty Claim ]
          │
          ▼
┌────────────────────────────────────────┐
│      FastAPI REST API (/predict)       │
└──────────────────┬─────────────────────┘
                   │
                   ▼
┌────────────────────────────────────────┐
│     Preprocessing & Sanitization       │
│  - Serial formatting (KH...)           │
│  - Prompt-injection text neutralization│
└──────────────────┬─────────────────────┘
                   │
                   ▼
┌────────────────────────────────────────┐
│     Feature Engineering Pipeline       │
│  - Auto-approval rule (< ₹2,000)       │
│  - Claim-to-price ratio                │
│  - Customer repeat claims              │
│  - Bayesian Partner History Lookup     │
└──────────────────┬─────────────────────┘
                   │
                   ▼
┌────────────────────────────────────────┐
│  Gradient Boosting Classifier (v1.0.0) │
│  - Outputs continuous fraud probability│
└──────────────────┬─────────────────────┘
                   │
                   ▼
┌────────────────────────────────────────┐
│      Explainability & Triage Layer     │
│  - Decision: APPROVE / REVIEW / MANUAL │
│  - 3–4 business-friendly explanations  │
└──────────────────┬─────────────────────┘
                   │
                   ▼
┌────────────────────────────────────────┐
│       Streamlit Operations Desk        │
└────────────────────────────────────────┘
```

---

## 5. Dataset Profile

| Dataset | Dimensions | Date Range | Description |
| :--- | :--- | :--- | :--- |
| `train.csv` | 12,029 × 14 | 2025-04-01 to 2026-06-30 | 11,814 resolved claims; 215 undecided cases excluded from training |
| `test_unlabelled.csv` | 2,252 × 13 | 2026-07-01 to 2026-09-30 | Future deployment claims for submission |
| `partners.csv` | 380 × 4 | Onboarded 2021 to 2026 | Partner type, city, and onboarding date |
| `products.csv` | 21 × 4 | N/A | 7 appliance families, list prices, warranty durations |
| `sample_submission.csv` | 2,252 × 2 | N/A | Target format: `claim_id,score` |

---

## 6. Data Leakage Controls
- **Strict Target Isolation**: `is_fraud` is used only as the training target.
- **Expanding Historical Aggregations**: Partner claim volume and fraud counts are calculated strictly chronologically. Claim $i$ at timestamp $t_i$ only accesses claims resolved strictly prior to $t_i$.
- **Bayesian Prior Smoothing**: Partner rates are smoothed with $\alpha = 10.0$ against the platform global prior (1.23%), ensuring new or low-volume partners do not leak extreme rates.
- **Exclusion of Post-Resolution Data**: Technician sign-off canned notes (`inspector_note`) and potential resolution outcomes are excluded from model inputs.

---

## 7. Feature Engineering
- **Domain & Policy**:
  - `is_under_2000`: Binary indicator for claims below ₹2,000 (§5 auto-approval rule).
  - `partner_tenure_days`: Elapsed days between partner onboarding and claim submission.
  - `partner_is_new`: Indicator for partners onboarded < 365 days.
  - `new_partner_small_claim`: Interaction capturing new partners filing claims just under ₹2,000.
  - `claim_to_price_ratio`: Claim amount divided by product retail price.
  - `warranty_elapsed_ratio`: Days since purchase divided by total product warranty days.
  - `customer_prior_claims`: Repeat claimant frequency.
- **Historical Partner Reputation**:
  - `p_prior_claims`: Cumulative claims filed by partner prior to current claim.
  - `p_prior_frauds`: Cumulative fraud claims by partner prior to current claim.
  - `p_prior_fraud_rate`: Laplace-smoothed historical fraud rate.

---

## 8. Model Selection & Temporal Validation
Validation was performed on a **Strict Temporal Holdout Split**:
- **Train Split**: 1 April 2025 to 31 May 2026 (11,047 claims, 123 frauds).
- **Validation Holdout**: June 2026 (767 claims, 22 frauds, 2.87% fraud prevalence).

### Benchmark Comparison (June 2026 Holdout @ Threshold 0.40):
| Model | ROC-AUC | PR-AUC | Accuracy | Precision | Recall | F1-Score |
| :--- | :--- | :--- | :--- | :--- | :--- | :--- |
| Logistic Regression (Balanced) | 0.8868 | 0.3702 | 61.54% | 6.39% | 90.91% | 0.1194 |
| Random Forest (Balanced, depth=6) | 0.9256 | 0.3494 | 94.00% | 28.57% | 72.73% | 0.4103 |
| HistGradientBoosting (Balanced) | 0.8886 | 0.2712 | 90.09% | 17.86% | 68.18% | 0.2830 |
| **Gradient Boosting Classifier (Selected)** | **0.8826** | **0.4201** | **97.52%** | **58.82%** | **45.45%** | **0.5128** |

---

## 9. Expected Hidden-Test Performance (`test_unlabelled.csv`)
- **Expected Accuracy**: **96.5% – 97.8%** (Point: 97.2%)
- **Expected Fraud Recall**: **40.0% – 58.0%** (Point: 48.0%)
- **Expected Precision**: **42.0% – 62.0%** (Point: 52.0%)
- **Expected ROC-AUC**: **0.860 – 0.915** (Point: 0.885)
- **Expected PR-AUC**: **0.340 – 0.480** (Point: 0.410)

---

## 10. Partner Hypothesis Findings
Ritu Deshpande suggested newer partners are the primary cause of fraud.
- **Empirical Reality**:
  - The 2025–2026 partner cohorts exhibit a higher aggregate fraud rate (2.9%–4.5% vs 1.2% baseline).
  - However, **91.1% of newer partners had zero fraudulent claims**.
  - Fraud is concentrated in **fewer than 10 rogue outlets** (both older franchises and 7 newly onboarded bad actors exploiting the ₹2,000 rule).
  - **Recommendation**: Do not implement blanket bans on new partners; instead, audit individual repeat-offender outlets and claims under ₹2,000.

---

## 11. Project Structure

```
Banao-Task2/
├── Given/                          # Raw inputs
│   ├── train.csv
│   ├── test_unlabelled.csv
│   ├── partners.csv
│   ├── products.csv
│   ├── ops-policy.pdf
│   ├── email-thread.txt
│   ├── README.txt
│   └── sample_submission.csv
├── src/                            # Production ML Pipeline
│   ├── __init__.py
│   ├── config.py                   # Constants, file paths, feature lists
│   ├── data_loader.py              # Ingestion and schema joining
│   ├── preprocessing.py            # Sanitization and serial cleaning
│   ├── feature_engineering.py      # Policy features and partner tracker
│   ├── train.py                    # Model training and artifact export
│   ├── evaluate.py                 # Evaluation charts and reporting
│   ├── predict.py                  # Batch prediction pipeline
│   ├── validate_submission.py      # 10-point submission validator
│   ├── explain.py                  # Human-readable explainability engine
│   └── api.py                      # FastAPI REST service
├── app/
│   └── streamlit_app.py            # Streamlit Operations Desk UI
├── models/                         # Serialized Model Artifacts
│   ├── model.joblib
│   ├── preprocessor.joblib
│   ├── partner_stats.json
│   └── model_metadata.json
├── reports/                        # Documentation & Visuals
│   ├── data_profile.md
│   ├── model_evaluation.md
│   ├── error_analysis.md
│   ├── expected_score.md
│   ├── partner_hypothesis.md
│   ├── memo_to_ritu.md
│   └── figures/
│       ├── class_distribution.png
│       ├── roc_curve.png
│       ├── pr_curve.png
│       ├── confusion_matrix.png
│       ├── threshold_tradeoff.png
│       └── fraud_by_category.png
├── tests/                          # Automated Pytest Suite
│   ├── test_data.py
│   ├── test_features.py
│   ├── test_model.py
│   └── test_api.py
├── examples/
│   ├── sample_claim.json
│   └── test_api.py
├── predictions.csv                 # Final test predictions
├── submission-form.md              # Completed submission form
├── requirements.txt                # Python dependencies
├── pytest.ini                      # Pytest configuration
├── .gitignore
└── README.md
```

---

## 12. Installation & Quickstart

### Step 1: Environment Setup
```bash
# Clone the repository
git clone https://github.com/Raghava44u/Banao-Task2.git
cd Banao-Task2

# Create virtual environment (Python 3.10+)
python -m venv .venv

# Activate environment (Windows)
.venv\Scripts\activate

# Activate environment (Linux/macOS)
source .venv/bin/activate

# Install dependencies
pip install -r requirements.txt
```

---

## 13. Step-by-Step Execution

### Step 2: Train Model & Save Artifacts
```bash
python -m src.train
```

### Step 3: Run Comprehensive Evaluation & Generate Figures
```bash
python -m src.evaluate
```

### Step 4: Generate Final Test Predictions (`predictions.csv`)
```bash
python -m src.predict
```

### Step 5: Validate Submission Schema & Integrity
```bash
python -m src.validate_submission
```

### Step 6: Run Pytest Test Suite
```bash
pytest -v
```

---

## 14. Starting the API & Web Dashboard

### Run FastAPI Service
```bash
uvicorn src.api:app --reload
```
- API Docs: `http://127.0.0.1:8000/docs`
- Health: `http://127.0.0.1:8000/health`
- Model Info: `http://127.0.0.1:8000/model-info`

### Test API via Example Client
```bash
python -m examples.test_api
```

### Run Streamlit Operations Desk UI
```bash
streamlit run app/streamlit_app.py
```
Open your browser at `http://localhost:8501`.

---

## 15. AI Tools Disclosure
- **Development Assistant**: Claude Code (Agentic coding pair-programmer).
- **Runtime Inference Dependencies**: 100% Local Scikit-learn (Zero paid external LLM or cloud API dependencies).
- **Paid API Cost**: ₹0.00.
- **What Was Discarded**:
  - Global target encoding was discarded due to severe future lookahead leakage.
  - Deep learning models were discarded due to inference latency and lack of native tree explainability.
  - Random train/test split was discarded because it fails to capture concept drift from the May 1 policy change.

---

## 16. Assumptions & Limitations
1. **Capacity Cap**: Assumes Farhan Sheikh's cap of 40 manual reviews per month remains operational policy.
2. **Goodwill Cost**: Assumes ₹380 per delayed genuine customer per §4.
3. **Cold-Start Partners**: For 14 new partners in the test set with no training history, historical fraud rate safely defaults to the platform prior (1.23%).

---

## 17. GitHub Repository & Verification
- **Repository**: [https://github.com/Raghava44u/Banao-Task2.git](https://github.com/Raghava44u/Banao-Task2.git)
- **Branch**: `main`
