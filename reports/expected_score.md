# Expected Performance on Hidden Evaluation Test Set

## 1. Executive Prediction Summary

Based on rigorous temporal holdout evaluation on the June 2026 cohort (767 claims) and 5-Fold Stratified Cross-Validation across 11,814 resolved historical claims, here is the honest, unvarnished expected performance on `test_unlabelled.csv` (July 1, 2026 to September 30, 2026; 2,252 claims):

| Metric | Point Estimate | Expected 95% Confidence Interval | Primary Driving Factor |
| :--- | :--- | :--- | :--- |
| **Hidden-Test Accuracy** | **97.2%** | **96.5% – 97.8%** | Extreme class imbalance (~1.5%–2.8% true fraud rate) and high specificity. |
| **Fraud Recall** | **48.0%** | **40.0% – 58.0%** | Interception of repeat rogue partners and auto-approval loophole claims. |
| **Fraud Precision** | **52.0%** | **42.0% – 62.0%** | Concentrated scoring on high-risk partner-customer clusters. |
| **ROC-AUC** | **0.885** | **0.860 – 0.915** | Robust rank-ordering of risk probabilities across product families. |
| **PR-AUC** | **0.410** | **0.340 – 0.480** | ~25x to 35x enrichment over the background fraud prevalence. |

---

## 2. Quantitative Rationale by Metric

### Expected Hidden-Test Accuracy: ~96.5% – 97.8%
- **Empirical Foundation**: In our June 2026 temporal holdout, the selected Gradient Boosting model achieved **97.52% accuracy** at decision threshold $0.40$. Across 5-fold CV, accuracy remained consistently between 96.8% and 97.6%.
- **Why It Holds**: Because over 97% of incoming claims are genuine, maintaining high specificity (>98%) guarantees overall accuracy stays above or right around the Board's stated 97.0% threshold.
- **Critical Caveat**: As documented, accuracy is heavily buffered by genuine claims. Management must not evaluate fraud efficacy solely on accuracy.

### Expected Fraud Recall: ~40.0% – 58.0%
- **Empirical Foundation**: On the June 2026 holdout, the model intercepted 10 out of 22 frauds at threshold 0.40 (45.5% recall) and 15 out of 22 frauds within the top-40 ranked audit queue (68.2% recall).
- **Why It Holds**: The primary fraud mechanism in the test era is policy arbitrage under the May 1 auto-approval rule (< ₹2,000 without inspection). The model strongly penalizes small claims from newer partners with elevated historical rates.
- **Potential Variation**: If new rogue partners emerge who had zero prior claims in the training dataset, cold-start latency will moderate recall toward the lower end of the interval (~40%).

### Expected Fraud Precision: ~42.0% – 62.0%
- **Empirical Foundation**: Holdout precision at threshold 0.40 was **58.8%** (10 true positives, 7 false positives).
- **Why It Holds**: The model combines multiple risk factors (partner fraud rate, repeat customer claims, claim-to-price ratio) before crossing the 0.40 probability threshold, keeping false positives very low (~7 to 15 per month).

---

## 3. Operational Risk Factors & Potential Distribution Shift

1. **Rogue Partner Onboarding During Test Period (July–Sept 2026)**:
   - 14 service partners in `test_unlabelled.csv` did not submit any claims during the training period.
   - For these 14 partners, historical fraud counters default to the platform prior ($\sim 1.23\%$). If one of these partners was established purely for fraud, the model will not flag their initial claims until volume/velocity signals trigger.
2. **Shift in Fraud Claim Amounts**:
   - If fraudsters learn that claims under ₹2,000 are being audited and shift their claim amounts back upward (or split invoices), model weights on the ₹2,000 threshold will lose marginal efficacy.
3. **Capacity Constraints**:
   - At Farhan Sheikh's cap of 40 audits per month, Kestrel can review at most 120 claims across the 3-month test period. Ranking by continuous risk score ensures that these 120 slots are allocated to the top 5.3% highest-risk claims.
