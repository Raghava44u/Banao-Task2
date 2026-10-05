# Kestrel Home Warranty Claim Review — Task 2 V3 (Variant C)
## Official Submission Form

---

### 1. General Project Information
- **Project Name:** Kestrel Home Warranty Claim Review System (Task 2 V3 — Variant C)
- **Candidate / Lead Engineer:** AI Pair Programmer & Machine Learning Engineering Lead
- **Target Organization:** Kestrel Home Appliances (Pune, Maharashtra, India)
- **Repository URL:** https://github.com/Raghava44u/Banao-Task2.git
- **Target Git Branch:** `main`

---

### 2. High-Level Approach
We developed a complete, reproducible, production-ready machine learning fraud detection pipeline tailored to Kestrel Home's operational warranty workflow. The solution models pre-payout risk to prevent fraudulent disbursements before funds leave the company. It incorporates:
1. Strict leakage-free chronological data processing and expanding Bayesian partner history tracking.
2. Direct operational modeling of Kestrel Operations Policy v4.1 (specifically §5 auto-approval rules and §4 goodwill cost matrices).
3. Realistic temporal holdout validation (June 2026 holdout after observing 14 months of historical claims).
4. A production FastAPI REST service returning risk scores, triage decisions, and plain-English explainability reasons.
5. An interactive Streamlit operations desk UI connected to the live API with local fallback.
6. Zero external paid API runtime dependencies (₹0 inference cost).

---

### 3. Data Used
- `train.csv`: 12,029 historical warranty claims (11,814 resolved ground truth cases; 215 undecided cases excluded from training).
- `test_unlabelled.csv`: 2,252 recent unlabelled claims (July 1, 2026 to September 30, 2026).
- `partners.csv`: 380 service partner outlets across 15 cities with onboarding dates and partner types.
- `products.csv`: 21 product SKUs across 7 appliance families with list prices and warranty durations.
- `ops-policy.pdf`: Operations policy v4.1 defining capacity (40 reviews/month), costs (₹380 goodwill, ₹260 contact), and auto-approval thresholds (₹2,000).
- `email-thread.txt`: Executive operational context from Ritu Deshpande, Farhan Sheikh, Meenal Joshi, and Tanmay Kulkarni.

---

### 4. Machine Learning Model Architecture
- **Primary Production Model:** `GradientBoostingClassifier` (`n_estimators=100`, `max_depth=3`, `random_state=42`).
- **Feature Transformation:** Scikit-learn `ColumnTransformer` applying `StandardScaler` to 12 numerical features and `OneHotEncoder(handle_unknown='ignore')` to 6 categorical features.
- **Candidate Models Benchmarked:**
  1. Logistic Regression (Balanced class weights)
  2. Random Forest Classifier (Balanced, depth=6)
  3. HistGradientBoostingClassifier (Balanced, max_leaf_nodes=15)
  4. Gradient Boosting Classifier (Selected)

---

### 5. Validation Methodology & Primary Results
- **Validation Strategy:** Strict Temporal Holdout (Training: 1 April 2025 to 31 May 2026 [11,047 claims, 123 frauds]; Validation: June 2026 [767 claims, 22 frauds]).
- **Primary Holdout Performance (@ Threshold 0.40):**
  - **Validation Accuracy:** **97.52%** (Exceeds the Board's >97.0% mandate)
  - **Fraud Precision:** **58.82%** (10 True Positives, 7 False Positives)
  - **Fraud Recall:** **45.45%** (Catches nearly half of all fraud cases)
  - **F1 Score:** **0.5128**
  - **ROC-AUC:** **0.8826**
  - **PR-AUC:** **0.4201** (34x higher than random base rate of 1.23%)
- **Capacity-Constrained Prioritization (Farhan Sheikh's Top-40 Audit Cap):**
  - **Audited:** Top 40 ranked claims in June 2026.
  - **Frauds Intercepted:** 15 out of 22 (**68.2% Recall**).
  - **Precision:** **37.5%** (15 TP, 25 FP).
  - **Net Monthly Benefit:** **+₹12,043 / month** (₹21,543 fraud stopped − ₹9,500 goodwill compensation).

---

### 6. Expected Hidden-Test Performance
- **Expected Accuracy:** **96.5% – 97.8%** (Point estimate: 97.2%)
- **Expected Recall:** **40.0% – 58.0%** (Point estimate: 48.0%)
- **Expected Precision:** **42.0% – 62.0%** (Point estimate: 52.0%)
- **Expected ROC-AUC:** **0.860 – 0.915** (Point estimate: 0.885)
- **Expected PR-AUC:** **0.340 – 0.480** (Point estimate: 0.410)
- **Rationale:** Derived from June 2026 temporal holdout and 5-fold cross-validation. Buffering by high genuine prevalence guarantees high accuracy, while cold-start latency on 14 unseen partners moderates recall lower bound.

---

### 7. Data Leakage Controls
- **Target Exclusion:** `is_fraud` is strictly used as target and excluded from all feature pipelines.
- **Chronological Aggregation:** Partner historical claim counts, fraud counts, and fraud rates are computed using an expanding temporal window where claim $i$ only accesses claims resolved strictly prior to its submission timestamp ($t_j < t_i$).
- **Laplace / Empirical Bayes Smoothing:** Historical partner rates are smoothed ($\alpha = 10.0$) against the global platform mean (1.23%), ensuring new or low-volume partners do not leak extreme rates.
- **Text & Notes:** Post-investigation text is omitted; canned technician notes and adversarial text prompt injections are neutralized to inert categorical mappings.

---

### 8. Key Predictive Features
1. `p_prior_fraud_rate`: Leakage-safe historical partner fraud rate (Bayesian smoothed).
2. `p_prior_frauds`: Cumulative historical fraudulent claims committed by this partner.
3. `new_partner_small_claim`: Interaction between partner tenure < 365 days and claim amount < ₹2,000 (capturing May 1 auto-approval loophole arbitrage).
4. `customer_prior_claims`: Repeat claimant frequency counter.
5. `claim_to_price_ratio`: Claim amount as percentage of retail list price.
6. `partner_inspected`: Indicator of whether partner inspection sign-off occurred.
7. `warranty_elapsed_ratio`: Days since purchase relative to product warranty duration.

---

### 9. Partner Hypothesis Conclusion
- **Hypothesis:** Ritu Deshpande proposed that newer partners are the primary source of fraud.
- **Empirical Finding:** Partially true in recent timing, but dangerous as a blanket policy.
- **Details:** 
  - Before May 2026, fraud was concentrated in older franchises onboarded in 2021–2023 submitting large claims.
  - After May 1, 2026, a small group of 7 newly onboarded partners exploited the uninspected ₹2,000 auto-approval rule.
  - However, **over 91% of newly onboarded partners are completely genuine**.
  - Management must **not** penalize all new partners; instead, target individual repeat-offender outlets and audit claims under ₹2,000.

---

### 10. API Specification
- **Framework:** FastAPI with Pydantic schema validation and in-memory artifact caching.
- **Endpoints:**
  - `POST /predict`: Ingests claim JSON; returns `claim_id`, `fraud_score`, `risk_level`, `decision`, `reasons`, and `model_version`.
  - `GET /health`: Returns service health status and model availability.
  - `GET /model-info`: Returns model architecture, validation metrics, and active feature sets.

---

### 11. User Interface
- **Framework:** Streamlit enterprise operations dashboard (`app/streamlit_app.py`).
- **Features:** Dynamic claim assessment form, real-time API connectivity with local fallback, clear visual risk badges (LOW/MEDIUM/HIGH), 3–5 human-readable explanation bullet points, and governance metrics.

---

### 12. Known Failure Modes
1. **Cold-Start Rogue Partners:** A newly onboarded rogue partner's initial 1–2 claims blend in because historical fraud counters start at platform baseline.
2. **False Positives in Major Repairs:** Genuine customers experiencing catastrophic failures on expensive appliances (e.g. Robot Vacuum motherboard) mimic high claim-to-price fraud patterns.
3. **Low-Value Electrical Claims:** Legitimate ceiling fan or heater repairs (₹400–₹800) submitted without photos are difficult to distinguish from fake small claims.

---

### 13. AI Tools & Costs Disclosure
- **Development Coding Assistant:** Claude Code (Agentic coding pair-programmer)
- **Runtime Inference Dependencies:** 100% Local Python / Scikit-learn (Zero external LLMs or cloud inference APIs)
- **Paid API Cost:** ₹0.00
- **What Was Discarded:**
  - Discarded complex deep learning models (excessive latency, zero interpretability).
  - Discarded global target encoding (caused severe future-to-past data leakage).
  - Discarded random train/test split (ignored concept drift caused by the May 1 policy change).

---

### 14. Reproducibility & Quick Run Instructions
```bash
# 1. Environment Setup
pip install -r requirements.txt

# 2. Train Pipeline & Save Artifacts
python -m src.train

# 3. Comprehensive Evaluation & Reporting
python -m src.evaluate

# 4. Generate Predictions
python -m src.predict

# 5. Validate Submission
python -m src.validate_submission

# 6. Run Test Suite
pytest -v

# 7. Start FastAPI Service
uvicorn src.api:app --reload

# 8. Start Streamlit UI
streamlit run app/streamlit_app.py
```
