# Predicting Loan Payback - Playground Series S5E11

![Competition](https://img.shields.io/badge/Competition-Playground%20Series%20S5E11-blue)
![Score](https://img.shields.io/badge/Private%20LB%20Score-404-brightgreen)
![Ranking](https://img.shields.io/badge/Ranking-Top%2011%25-success)
![Status](https://img.shields.io/badge/Status-Late%20Submission-orange)
![Metric](https://img.shields.io/badge/Metric-ROC--AUC-informational)

## Table of Contents

- [Overview](#overview)
- [Personal Results](#personal-results)
- [Competition Details](#competition-details)
- [Problem Description](#problem-description)
- [Solution Approach](#solution-approach)
- [Notebook Summary](#notebook-summary)
- [Feature Engineering Highlights](#feature-engineering-highlights)
- [Modeling & Ensembling](#modeling--ensembling)
- [Dataset Information](#dataset-information)
- [Evaluation Metric](#evaluation-metric)
- [Timeline](#timeline)
- [Prizes](#prizes)
- [Resources & References](#resources--references)

---

## Overview

**Predicting Loan Payback** is a Kaggle Tabular Playground Series competition (Season 5, Episode 11). The goal is to predict the probability that a borrower will pay back their loan using a synthetic dataset generated from a real-world loan prediction dataset. The challenge is optimized for fast iteration and model experimentation while maintaining realistic feature relationships.

---

## Personal Results

| Metric | Value |
|--------|-------|
| **Private Leaderboard Score** | 404 |
| **Ranking** | Top 11% |
| **Submission Type** | Late Submission |
| **Evaluation Metric** | ROC-AUC |
| **Notebook** | [View Public Notebook](https://www.kaggle.com/code/adilshamim8/predicting-loan-payback-101) |

---

## Competition Details

- **Series:** Playground Series - Season 5, Episode 11
- **Goal:** Predict the probability that a borrower pays back a loan
- **Metric:** Area Under the ROC Curve (ROC-AUC)
- **Start Date:** November 1, 2025
- **Final Submission Deadline:** November 30, 2025
- **Dataset Type:** Synthetic, derived from the Loan Prediction dataset
- **Participation:** 11,450 entrants, 3,724 teams, 31,791 submissions

---

## Problem Description

Given borrower demographics, loan characteristics, and financial indicators, predict `loan_paid_back` for each test row. Predictions must be probabilities between 0 and 1. Performance is measured by ROC-AUC, rewarding a good ranking of positives vs negatives across the dataset.

**Submission format**:

```csv
id,loan_paid_back
593994,0.5
593995,0.2
593996,0.1
```

---

## Solution Approach

The notebook follows a complete modeling pipeline:

1. **Setup & Config** - Reproducible environment, seeded runs, and configuration class for model weights and folds.
2. **Data Loading** - Train/test ingestion and quick shape checks.
3. **EDA** - Data previews, data types, and missing values analysis.
4. **Feature Engineering** - A rich 12-stage feature pipeline for risk, affordability, ratios, bins, and interactions.
5. **Model Training** - LightGBM, XGBoost, and CatBoost with stratified CV.
6. **Model Evaluation** - OOF AUC comparison, ROC curves, and feature importance.
7. **Ensemble Methods** - Multiple blending strategies, auto-selecting the best ensemble.
8. **Submission Generation** - Final prediction distribution checks and submission file creation.

---

## Notebook Summary

The notebook is a full tutorial-style workflow with the following sections:

1. **Introduction & Setup** - Imports, configuration, and reproducibility.
2. **Data Loading & Overview** - Train/test shapes and sample previews.
3. **EDA** - Data types, missing values, and summary checks.
4. **Feature Engineering** - Extensive feature creation (12 stages).
5. **Complete Feature Engineering** - Apply engineered features to train/test.
6. **Model Training** - LightGBM, XGBoost, CatBoost with 5-fold stratified CV.
7. **Model Evaluation** - OOF AUC comparison and ROC curves.
8. **Ensemble Methods** - Simple, weighted, rank, power mean, optimized, blended.
9. **Submission Generation** - Create `submission.csv` and visualize predictions.
10. **Conclusion** - Key insights, top features, and improvement ideas.

---

## Feature Engineering Highlights

The pipeline expands from 13 original features to **70+ engineered features**, including:

- **Ratios & Burden Metrics:** `debt_to_income_ratio`, `loan_to_income_ratio`, `payment_to_income_ratio`
- **Risk Scores:** `risk_score_v1`, `risk_score_v2`, `risk_score_v3`, `financial_health_score`
- **Loan Amount Features:** log/sqrt transforms and loan size bins
- **Binning & Quantiles:** deciles/quintiles for income, credit, loan, and interest
- **Interactions:** income-credit, loan-interest, and three-way interactions
- **Grade/Subgrade Encoding:** derived numeric grade features
- **Statistical Aggregations:** mean, max, min, std, range, coefficient of variation
- **Categorical Combos:** `gender_marital`, `education_employment`, `purpose_grade`
- **Anomaly Flags:** extreme DTI, low income, risky combinations
- **Polynomial/Ratios:** squared/cubed DTI, efficiency ratios

---

## Modeling & Ensembling

### Base Models

- **LightGBM** (GBDT, deep trees, strong regularization)
- **XGBoost** (hist, deeper trees, stronger regularization)
- **CatBoost** (logloss, tuned depth, and bagging)

### Cross-Validation

- 5-fold stratified CV
- OOF predictions for model comparison

### Ensemble Strategies

- Simple average
- Weighted average by model performance
- Rank average
- Power mean ensemble
- Optimized weighted ensemble
- Blended ensemble (linear mix of top strategies)

The pipeline selects the best ensemble based on OOF AUC and uses it for final predictions.

---

## Dataset Information

- **Files:** `train.csv`, `test.csv`, `sample_submission.csv`
- **Rows:** ~254,569 test rows (per sample submission)
- **Columns:** 27 total columns
- **Size:** 81.3 MB
- **License:** CC BY 4.0
- **Source:** Synthetic data generated from a Loan Prediction dataset

---

## Evaluation Metric

The competition uses **ROC-AUC**, which measures ranking quality of predicted probabilities:

$$
	ext{AUC} = \int_0^1 \text{TPR}(\text{FPR})\, d\text{FPR}
$$

Higher is better. A score of 0.5 is random, 1.0 is perfect classification.

---

## Timeline

| Event | Date |
|-------|------|
| Competition Start | November 1, 2025 |
| Entry Deadline | November 30, 2025 |
| Team Merger Deadline | November 30, 2025 |
| Final Submission Deadline | November 30, 2025 |

---

## Prizes

| Place | Prize |
|-------|-------|
| 1st | Choice of Kaggle merchandise |
| 2nd | Choice of Kaggle merchandise |
| 3rd | Choice of Kaggle merchandise |

Kaggle merchandise is awarded once per person in this series.

---

## Resources & References

- **Competition Overview:** https://www.kaggle.com/competitions/playground-series-s5e11/overview
- **Dataset:** https://www.kaggle.com/competitions/playground-series-s5e11/data
- **Public Notebook:** https://www.kaggle.com/code/adilshamim8/predicting-loan-payback-101

**Citation**

```
Yao Yan, Walter Reade, Elizabeth Park. Predicting Loan Payback.
https://kaggle.com/competitions/playground-series-s5e11, 2025. Kaggle.
```

---

**Public Notebook:** https://www.kaggle.com/code/adilshamim8/predicting-loan-payback-101
