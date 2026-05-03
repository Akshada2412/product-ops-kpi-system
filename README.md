# Product Operations Intelligence Platform

> End-to-end SaaS product analytics system — 
> revenue trends, feature adoption metrics, and 
> ML-powered churn risk scoring across 1,000 users.

🔴 **[View Live Dashboard](https://appuct-ops-kpi-system-xzeyg9ghwb9zwrd4tkowof.streamlit.app/)**
💻 **[View Source Code](https://github.com/Akshada2412)**

---

## The Problem

SaaS product teams are drowning in data but starving 
for insight. They need a system that connects revenue 
trends, user engagement signals, and churn risk into 
one actionable view — and flags at-risk accounts 
before it's too late.

## The Solution

A full-stack product analytics platform that:
- Tracks MRR trends and revenue concentration by plan
- Measures feature adoption and session depth by cohort
- Scores every user's churn probability using a trained
  Random Forest model (ROC-AUC: 0.883)
- Surfaces high-risk accounts in a live alert panel

---

## Live Dashboard

**[Click to open →](https://appuct-ops-kpi-system-xzeyg9ghwb9zwrd4tkowof.streamlit.app/)**

Features:
- MRR trend by plan tier (stacked area chart)
- Revenue concentration — 63.5% from Enterprise
- Feature adoption vs session depth by plan
- Support ticket burden analysis
- NPS satisfaction scores by cohort
- Monthly active user retention curve
- ML churn risk panel with intervention table

Filters: Plan tier · Industry (live, updates all panels)

---

## Key Findings

| Metric | Finding |
|--------|---------|
| MRR trend | Declining — $63K → $45K over 12 months |
| Revenue concentration | Enterprise = 63.5% of all revenue |
| Highest churn risk | Free plan — 12% monthly churn rate |
| Top churn predictor | Support tickets (32.7% importance) |
| NPS leaders | Pro (7.14) and Enterprise (7.06) |
| Free plan cost | $0 revenue, 1,885 support tickets |

**The Pro plan problem:** Highest support burden 
(7.96 tickets/user) but only $99/month — users are 
outgrowing the plan without upgrading.

---

## ML Model — Churn Prediction

**Algorithm:** Random Forest Classifier  
**Features:** Sessions, features used, support tickets,
NPS score, plan tier, industry, company size,
acquisition channel  
**Performance:**

| Metric | Score |
|--------|-------|
| Accuracy | 80% |
| ROC-AUC | 0.883 |
| Precision (churn) | 86% |
| Recall (churn) | 78% |

**Top predictors:**
1. Support tickets (32.7%) — frustrated users leave
2. Avg sessions (19.4%) — disengaged users leave
3. Features used (18.5%) — users not finding value leave

**Note on data leakage:** Initial model achieved 
ROC-AUC 1.0 using `months_active` as a feature — 
correctly identified and removed as it directly encodes 
the outcome. Final model uses only behavioral signals 
available at prediction time.

---

## Data Methodology

The dataset is synthetically generated using 
statistically realistic parameters calibrated against 
published SaaS industry benchmarks:

- **Churn rates** — Bessemer Venture Partners SaaS 
  metrics (free: 10-15%, enterprise: 1-3%)
- **NPS distributions** — ProfitWell benchmark data
- **Session patterns** — OpenView Product Benchmarks
- **Revenue tiers** — Standard SaaS pricing 
  ($0 / $29 / $99 / $399)

The analytical methodology — cohort segmentation, 
churn modeling, and feature importance analysis — is 
identical to what practitioners apply on production data.

---

## Tech Stack

| Layer | Tools |
|-------|-------|
| Data Generation | Python, NumPy, Pandas |
| Database | SQLite, SQL (CTEs, aggregations) |
| ML Model | Scikit-learn (Random Forest) |
| Dashboard | Streamlit, Plotly |
| Deployment | Streamlit Cloud |
| Version Control | Git, GitHub |

---

## Project Structure

| product-ops-kpi-system/
| ├── data/
| │   ├── raw/                   # Generated datasets
| │   └── processed/             # Model outputs
| ├── notebooks/
| │   ├── 01_generate_data.py    # Data generation
| │   ├── 02_sql_analysis.py     # SQL business queries
| │   └── 03_churn_model.py      # ML pipeline
| ├── dashboard/
| │   └── app.py                 # Streamlit application
| └── docs/
| └── product_memo.md            # PM deliverable

---

## PM Governance Layer

This project includes full PM deliverables:
- Product insight memo with recommendations
- Feature prioritization framework
- Stakeholder-ready visualizations

---

## Author

**Akshada Karade**
MS Engineering Management, UMass Amherst
[LinkedIn](https://www.linkedin.com/in/akshadakarade2412/) |
[Email](mailto:akshadakarade@gmail.com)