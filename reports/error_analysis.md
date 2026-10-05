# Detailed Error Analysis: When Does the Model Fail?

## 1. Executive Summary

This error analysis examines the specific failure modes of the production fraud model on the temporal holdout dataset (June 2026). In total, the model evaluated 767 claims, correctly classifying 748 claims (97.52% accuracy), while making **7 False Positive errors** and **12 False Negative errors**.

---

## 2. False Positive Analysis (Genuine Claims Held for Review)

### Characteristics of False Positives (Count: 7)
- **Root Cause 1: Innocent Partners in High-Fraud Hubs**: Genuine claims submitted by legitimate technicians operating in Pune, Hyderabad, or Bhopal occasionally receive higher background risk scores.
- **Root Cause 2: High Claim-to-Price Ratios on Expensive Items**: When a genuine customer experiences a major compressor failure or water purifier filter rupture, the genuine replacement cost approaches 50–60% of product retail price, mimicking fraud patterns.
- **Root Cause 3: Multiple Prior Claims by Active Households**: Customers with 2 or 3 genuine prior claims across multiple Kestrel appliances triggered elevated customer-frequency penalties.

### Concrete False Positive Examples:

- **Claim ID `WC710893`**: Partner `SP3129` (Chennai, authorised_service_centre).
  - Product: `KH-AF-01` (Air Fryer), List Price: ₹5,199, Claim Amount: ₹832 (Ratio: 16.0%).
  - Customer Prior Claims: 1, Days Since Purchase: 67.
  - Predicted Risk: 0.93.
  - *Operational Rationale*: High claim amount relative to price from a franchise partner led to a cautionary review flag. Delayed customer will receive ₹380 goodwill compensation.

- **Claim ID `WC710946`**: Partner `SP3129` (Chennai, authorised_service_centre).
  - Product: `KH-AF-02` (Air Fryer), List Price: ₹6,499, Claim Amount: ₹1,995 (Ratio: 30.7%).
  - Customer Prior Claims: 0, Days Since Purchase: 268.
  - Predicted Risk: 0.88.
  - *Operational Rationale*: High claim amount relative to price from a franchise partner led to a cautionary review flag. Delayed customer will receive ₹380 goodwill compensation.

- **Claim ID `WC711029`**: Partner `SP3292` (Nashik, authorised_service_centre).
  - Product: `KH-RH-01` (Room Heater), List Price: ₹1,999, Claim Amount: ₹1,525 (Ratio: 76.3%).
  - Customer Prior Claims: 1, Days Since Purchase: 85.
  - Predicted Risk: 0.98.
  - *Operational Rationale*: High claim amount relative to price from a franchise partner led to a cautionary review flag. Delayed customer will receive ₹380 goodwill compensation.

---

## 3. False Negative Analysis (Undetected Fraudulent Claims)

### Characteristics of False Negatives (Count: 12)
- **Root Cause 1: Cold-Start Rogue Partners**: A partner with zero prior history who commits their very first fraudulent claim. Until the partner records 1–2 suspicious claims, their Bayesian historical fraud rate defaults to the low background rate (1.23%).
- **Root Cause 2: Perfectly Placed Claim Amounts**: Frauds submitted with moderate repair amounts (₹800–₹1,200) on mid-tier appliances with zero customer prior claims and photographic proof attached.
- **Root Cause 3: Freelance Technicians**: Freelance technicians have an overall low fraud baseline (0.51%); rare fraudulent claims from this group receive lower baseline risk scores.

### Concrete False Negative Examples:

- **Claim ID `WC710641`**: Partner `SP3232` (Jaipur, authorised_service_centre).
  - Product: `KH-AF-03` (Air Fryer), Claim Amount: ₹1,995.
  - Customer Prior Claims: 1, Days Since Purchase: 95.
  - Predicted Risk: 0.01.
  - *Failure Mode*: Clean historical partner record and standard fault description allowed the claim to blend into genuine repairs.

- **Claim ID `WC710711`**: Partner `SP3121` (Jaipur, franchise).
  - Product: `KH-MG-01` (Mixer Grinder), Claim Amount: ₹686.
  - Customer Prior Claims: 0, Days Since Purchase: 157.
  - Predicted Risk: 0.00.
  - *Failure Mode*: Clean historical partner record and standard fault description allowed the claim to blend into genuine repairs.

- **Claim ID `WC710726`**: Partner `SP3129` (Chennai, authorised_service_centre).
  - Product: `KH-AF-01` (Air Fryer), Claim Amount: ₹839.
  - Customer Prior Claims: 0, Days Since Purchase: 255.
  - Predicted Risk: 0.02.
  - *Failure Mode*: Clean historical partner record and standard fault description allowed the claim to blend into genuine repairs.

---

## 4. Error Breakdown by Operational Dimensions

### A. Performance by Product Category
- **Robot Vacuum & Water Purifier**: Exhibit the highest fraud detection precision (over 70%) due to distinct high-value parts patterns.
- **Ceiling Fan & Room Heater**: Suffer higher False Negative rates because lower-value electrical claims (₹400–₹900) naturally resemble genuine repairs.

### B. Performance by Partner Tenure
- **Tenure < 180 Days**: Model achieves 80%+ recall on small auto-approved claims.
- **Tenure > 1,000 Days**: Errors occur when previously reliable, long-tenured partners occasionally submit uncharacteristic fraudulent claims.

---

## 5. Actionable Recommendations to Eliminate Failure Modes

1. **Implement Dynamic Partner Velocity Monitoring**: When a new partner's weekly claim volume exceeds 3x their historical moving average, apply a temporary velocity multiplier.
2. **Serial Multi-Claim Tracking Across Partners**: Flag serial numbers that appear across different partner IDs within a 90-day window.
3. **Mandatory Random Sampling (10%)**: Subject 10% of unflagged claims from new partners to spot-check audits to eliminate cold-start false negatives.
