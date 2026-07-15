

```markdown
# Customer Purchase Prediction — Mahi & Bhumika

A machine learning project that predicts whether an e-commerce session 
will result in a purchase, built using the UCI Online Shoppers 
Purchasing Intention dataset.

---

## Project Overview

E-commerce platforms lose significant revenue when potential buyers 
leave without purchasing. This project builds and compares three 
classification models to predict purchase intent from session-level 
behavioural data, enabling businesses to proactively target high-intent 
users with personalised interventions.

---

## Dataset

- **Source:** UCI Online Shoppers Purchasing Intention Dataset
- **Size:** 12,330 sessions (after removing 125 duplicates)
- **Features:** 18 (10 numerical, 6 categorical/boolean)
- **Target:** Revenue (0 = No Purchase, 1 = Purchase)
- **Class imbalance:** 15.47% purchase rate — addressed using SMOTE

---

## Project Structure

```
├── notebooks/
│   ├── step1_data_loading.ipynb
│   ├── step2_data_cleaning.ipynb
│   ├── step3_eda.ipynb
│   ├── step4_feature_engineering.ipynb
│   ├── step5_logistic_regression.ipynb
│   ├── step6_random_forest.ipynb
│   ├── step7_xgboost.ipynb
│   ├── step8_evaluation.ipynb        (in progress)
│   ├── step9_shap.ipynb              (in progress)
├── app/
│   └── app.py                        (in progress)
├── paper/
│   └── research_paper.pdf            (in progress)
└── README.md
```

---

## Pipeline

```
Raw Data
    ↓
Step 1 — Data Loading & Exploration
    ↓
Step 2 — Data Cleaning & Encoding
    ↓
Step 3 — Exploratory Data Analysis
    ↓
Step 4 — Feature Engineering
    ↓
Step 5 — Logistic Regression (Baseline)
    ↓
Step 6 — Random Forest
    ↓
Step 7 — XGBoost
    ↓
Step 8 — Model Evaluation & Comparison
    ↓
Step 9 — SHAP Interpretability
    ↓
Step 10 — Streamlit Demo App
```

---

## Key Findings (EDA)

- **PageValues** is the strongest predictor of purchase (correlation = 0.492)
- **ExitRates** is the strongest negative predictor (correlation = -0.204)
- **BounceRates and ExitRates** are highly correlated (0.902) — multicollinearity flagged
- Purchase rate varies significantly by month — seasonality plays a real role
- Users who browse more product pages and spend longer on them are significantly more likely to buy

---

## Model Results

| Model | Accuracy | F1 | Precision | Recall | AUC |
|---|---|---|---|---|---|
| Logistic Regression | 0.8636 | 0.6279 | 0.5478 | 0.7356 | 0.8918 |
| **Random Forest** | **0.8865** | **0.6768** | **0.6105** | **0.7592** | **0.9255** |
| XGBoost | 0.8808 | 0.6403 | 0.6066 | 0.6780 | 0.9215 |

**Random Forest is our best performing model**, selected for final 
evaluation and SHAP interpretability analysis.

Notably, XGBoost — typically the stronger algorithm for tabular data 
— was outperformed by Random Forest here, likely due to the relatively 
small and clean nature of the dataset (12,000 rows, 21 features) where 
RF's bagging approach generalises better than XGBoost's aggressive boosting.

---

## Technologies Used

- Python 3
- pandas, numpy
- scikit-learn
- XGBoost
- imbalanced-learn (SMOTE)
- matplotlib, seaborn
- SHAP
- Streamlit (Step 10)
- Google Colab + Google Drive (collaborative workflow)
- GitHub (version control)

---

## Team

| Name | Steps |
|---|---|
| Mahi | 1, 3, 5, 7, 9 |
| Bhumika | 2, 4, 6, 8, 10 |

---

## Status

- [x] Step 1 — Data Loading
- [x] Step 2 — Data Cleaning
- [x] Step 3 — EDA
- [x] Step 4 — Feature Engineering
- [x] Step 5 — Logistic Regression
- [x] Step 6 — Random Forest
- [x] Step 7 — XGBoost
- [ ] Step 8 — Evaluation
- [ ] Step 9 — SHAP
- [ ] Step 10 — Streamlit App
- [ ] Research Paper

---

