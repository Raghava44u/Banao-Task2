# Partner Hypothesis Analysis: Empirical Investigation of Service Partner Fraud

## 1. Context & Hypothesis

Ritu Deshpande (Head of D2C Operations) stated:
> *"My own view is that the newer partners are the problem, but let the data say."*

Kestrel Home Appliances onboarded approximately 60 new service partners over the past year (late 2025 through mid-2026) to expand into tier-2 and tier-3 cities across India. Meanwhile, Service Desk Manager Meenal Joshi cautioned:
> *"Since the May change the small claims have exploded - my team sees the same few outlets again and again. But please don't paint all the new partners with one brush, most of the new ones are fine and we need them in the smaller cities."*

This report explicitly evaluates whether newer partners are inherently higher-risk, how partner tenure correlates with fraud, and what operational conclusions management must draw.

---

## 2. Empirical Findings: Cohort & Tenure Analysis

### A. Partner Fraud Rate by Onboarding Year

| Onboarding Cohort | Active Partners | Total Resolved Claims | Fraudulent Claims | Fraud Rate (%) | Avg Claim Amount (₹) |
| :--- | :--- | :--- | :--- | :--- | :--- |
| **2021** | 52 | 1,882 | 24 | 1.28% | ₹2,710 |
| **2022** | 76 | 2,676 | 36 | 1.35% | ₹2,645 |
| **2023** | 95 | 3,357 | 35 | 1.04% | ₹2,608 |
| **2024** | 78 | 2,800 | 14 | 0.50% | ₹2,624 |
| **2025** | 33 | 853 | 25 | **2.93%** | ₹2,398 |
| **2026** | 46 | 246 | 11 | **4.47%** | ₹2,102 |
| **Total** | 380 | 11,814 | 145 | 1.23% | ₹2,617 |

#### Key Observation:
On the surface, the aggregate fraud rate for the 2025 cohort (2.93%) and 2026 cohort (4.47%) is **2.4x to 3.6x higher** than the baseline rate of older cohorts (~1.2%). This gives initial credence to Ritu's intuition.

---

### B. Partner Fraud by Continuous Tenure at Claim Time

Breaking partner tenure into quintiles at the exact moment each claim was filed:

| Tenure Quintile | Tenure Range (Days) | Total Claims | Fraud Claims | Fraud Rate (%) |
| :--- | :--- | :--- | :--- | :--- |
| **Q1 (Youngest)** | 1 to 467 days | 2,365 | 36 | **1.52%** |
| **Q2** | 468 to 765 days | 2,370 | 19 | 0.80% |
| **Q3** | 766 to 1,028 days | 2,353 | 28 | 1.19% |
| **Q4** | 1,029 to 1,337 days | 2,363 | 29 | 1.23% |
| **Q5 (Oldest)** | 1,338 to 1,850 days | 2,363 | 33 | **1.40%** |

#### Crucial Nuance:
Across continuous tenure quintiles, the oldest partners (Q5: 1,338+ days tenure) have a fraud rate of **1.40%**, virtually indistinguishable from the youngest quintile (Q1: **1.52%**). 
Tenure alone has a correlation of **-0.006** with fraud across the overall dataset.

---

## 3. The Root Cause: Partner Concentration & The May 1 Policy Loophole

When we inspect individual service partners responsible for fraud, a striking pattern emerges:

### Top 10 Fraud Partners by Volume

| Partner ID | City | Onboarded Date | Partner Type | Total Claims | Fraud Claims | Partner Fraud Rate | Timing of Fraud |
| :--- | :--- | :--- | :--- | :--- | :--- | :--- | :--- |
| **SP3207** | Hyderabad | 2022-02-23 (Old) | Franchise | 40 | 14 | **35.0%** | April 2025 – April 2026 |
| **SP3376** | Mysuru | 2023-04-08 (Old) | Franchise | 40 | 12 | **30.0%** | April 2025 – April 2026 |
| **SP3095** | Bhopal | 2021-11-17 (Old) | Franchise | 38 | 11 | **28.9%** | April 2025 – April 2026 |
| **SP3160** | Pune | 2025-12-27 (New) | Service Centre | 42 | 10 | **23.8%** | May 2026 – June 2026 |
| **SP3103** | Jaipur | 2022-07-27 (Old) | Franchise | 37 | 9 | **24.3%** | April 2025 – April 2026 |
| **SP3350** | Hubballi | 2023-02-20 (Old) | Franchise | 33 | 9 | **27.3%** | April 2025 – April 2026 |
| **SP3232** | Jaipur | 2025-11-26 (New) | Service Centre | 42 | 7 | **16.7%** | May 2026 – June 2026 |
| **SP3228** | Pune | 2021-08-11 (Old) | Freelance Tech | 26 | 7 | **26.9%** | April 2025 – April 2026 |
| **SP3292** | Nashik | 2024-01-04 (Mid) | Service Centre | 32 | 7 | **21.9%** | April 2025 – April 2026 |
| **SP3318** | Hubballi | 2025-12-06 (New) | Franchise | 31 | 6 | **19.4%** | May 2026 – June 2026 |

### Two Distinct Fraud Epochs:
1. **Epoch 1 (Pre-May 2026)**: Fraud was driven primarily by established franchise partners onboarded between 2021 and 2023 (e.g., SP3207, SP3376, SP3095). These partners submitted large, high-value fraudulent repair claims (average ₹4,500+).
2. **Epoch 2 (Post-May 2026 Policy Change)**: On 1 May 2026, Kestrel enacted an auto-approval rule for claims under ₹2,000 without physical inspection (§5). A small cohort of partners onboarded in late 2025/2026 (SP3160, SP3232, SP3318, SP3129, SP3319, SP3118, SP3286) immediately discovered this loophole and flooded the CRM with claims just below ₹2,000 (average ₹1,400–₹1,995).

---

## 4. Controlled Multivariate Analysis

When controlling for:
1. **Claim Amount Regime (< ₹2,000 vs ≥ ₹2,000)**
2. **Partner Type (Franchise vs Service Centre vs Freelance)**
3. **Product Family (Mixer Grinder, Robot Vacuum, etc.)**

We find:
- **Franchise status** is inherently higher risk than freelance technicians (1.97% fraud rate for franchises vs 0.51% for freelance).
- **Out of 79 partners onboarded in 2025–2026**, exactly **7 partners** generated 34 of the 36 post-May frauds (94.4%).
- The remaining **72 newer partners (91.1%)** had a **0.0% fraud rate** with exemplary service records across Tier-2 and Tier-3 cities.

---

## 5. What Should Ritu Conclude?

1. **Ritu's hypothesis is partially true in recent timing, but dangerous as a blanket operational policy**:
   - Newer partners *as a statistical cohort* showed elevated fraud in May–June 2026 because 7 bad actors exploited the uninspected ₹2,000 auto-approval rule.
   - However, **over 91% of newer partners are completely genuine**.
2. **Blanket restrictions on new partners will severely damage Kestrel's D2C expansion**:
   - Restricting or terminating all new partners would strangle Kestrel's customer service turnaround in Tier-2/3 cities.
3. **The true threat is Partner Repeat Offense and Policy Arbitrage**:
   - 10 rogue partner outlets (both old franchises and newly onboarded bad actors) account for **over 63% of all fraud rupees**.
   - The model must target individual partner historical behavior, small-claim velocity, and repeat customer combinations rather than coarse onboarding age.
