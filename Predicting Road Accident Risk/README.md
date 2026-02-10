
# Predicting Road Accident Risk - Playground Series S5E10

![Competition](https://img.shields.io/badge/Competition-Playground%20Series%20S5E10-blue)
![Score](https://img.shields.io/badge/Private%20LB%20Score-29-brightgreen)
![Ranking](https://img.shields.io/badge/Ranking-Top%201%25-success)
![Status](https://img.shields.io/badge/Status-Late%20Submission-orange)
![Metric](https://img.shields.io/badge/Metric-RMSE-informational)

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
- [Contact & Links](#contact--links)

---

## Overview

This Kaggle Tabular Playground Series competition (Season 5, Episode 10) focuses on predicting road accident risk on different road types. The dataset is synthetic but derived from a simulated accident dataset, enabling quick iteration on EDA, feature engineering, and regression modeling.

This challenge also connects to the Stack Overflow Challenge, encouraging participants to build a web app and earn a “Code Scientist” badge.

---

## Personal Results

| Metric | Value |
|--------|-------|
| **Private Leaderboard Score** | 29 |
| **Ranking** | Top 1% |
| **Submission Type** | Late Submission |
| **Evaluation Metric** | RMSE |
| **Notebook** | [View Public Notebook](https://www.kaggle.com/code/adilshamim8/predicting-road-accident-risk-101) |

---

## Competition Details

- **Series:** Playground Series - Season 5, Episode 10
- **Goal:** Predict continuous-valued `accident_risk`
- **Metric:** RMSE (Root Mean Squared Error)
- **Start Date:** October 1, 2025
- **Final Submission Deadline:** October 31, 2025
- **Dataset Type:** Synthetic data derived from the Simulated Roads Accident dataset

---

## Problem Description

Predict the continuous target `accident_risk` in the range $[0, 1]$ for each test row. Submissions are evaluated using RMSE between predicted risk and observed target values.

**Submission format**:

```csv
id,accident_risk
517754,0.352
517755,0.992
517756,0.021
```

---

## Solution Approach

The notebook implements an end-to-end ML pipeline:

1. **Imports & Setup** - Libraries, plotting styles, and preprocessing tools.
2. **Data Loading** - Train/test ingestion and sample submission checks.
3. **EDA** - Target distribution, categorical/boolean/numeric analysis.
4. **Feature Interactions** - Risk patterns by road type, weather, lighting, and time.
5. **Feature Engineering** - One-hot encoding + interaction features.
6. **Model Development** - Baselines, boosted models, and tuning.
7. **Blending** - Weighted blends of top models.
8. **Submission** - Predictions, file export, and distribution checks.

---

## Notebook Summary

The notebook is structured into clear sections:

1. **Introduction** - Problem framing and RMSE evaluation.
2. **EDA** - Target distribution and feature-level analysis.
3. **Feature Interactions** - Heatmaps and interaction plots.
4. **Feature Engineering** - One-hot encoding + interaction features.
5. **Modeling** - Baseline models, tuned XGBoost, and blending.
6. **Submission** - Final prediction file creation and validation.
7. **Conclusion** - Summary and next steps.

---

## Feature Engineering Highlights

- **Categorical Encoding:** One-hot encoding for all categorical fields.
- **Interaction Features:**
	- `speed_curvature`
	- `lanes_speed`
	- `accident_speed_risk`
	- `curvature_squared`
- **Risk Flags:** Weather + lighting combinations (e.g., `high_risk_Rain_Dark_No_Lights`).

---

## Modeling & Ensembling

**Baseline models:**
- Linear Regression, Ridge
- Random Forest
- Gradient Boosting
- XGBoost, LightGBM, CatBoost

**Advanced training:**
- RandomizedSearchCV for hyperparameter tuning (XGBoost)
- Feature importance analysis

**Blending:**
- Weighted average of XGBoost, LightGBM, and CatBoost predictions

---

## Dataset Information

- **Files:** `train.csv`, `test.csv`, `sample_submission.csv`
- **Columns:** 29
- **Size:** 52.02 MB
- **License:** CC BY 4.0
- **Source:** Synthetic data generated from the Simulated Roads Accident dataset

---

## Evaluation Metric

The competition uses **RMSE**:

$$
	ext{RMSE} = \sqrt{\frac{1}{n} \sum_{i=1}^n (y_i - \hat{y}_i)^2}
$$

Lower is better. RMSE is measured in accident risk units.

---

## Timeline

| Event | Date |
|-------|------|
| Competition Start | October 1, 2025 |
| Entry Deadline | October 31, 2025 |
| Team Merger Deadline | October 31, 2025 |
| Final Submission Deadline | October 31, 2025 |

---

## Prizes

| Place | Prize |
|-------|-------|
| 1st | Choice of Kaggle merchandise |
| 2nd | Choice of Kaggle merchandise |
| 3rd | Choice of Kaggle merchandise |

Merchandise is awarded once per person in this series.

---

## Resources & References

- **Competition Overview:** https://www.kaggle.com/competitions/playground-series-s5e10/overview
- **Dataset:** https://www.kaggle.com/competitions/playground-series-s5e10/data
- **Public Notebook:** https://www.kaggle.com/code/adilshamim8/predicting-road-accident-risk-101

**Citation**

```
Walter Reade and Elizabeth Park. Predicting Road Accident Risk.
https://kaggle.com/competitions/playground-series-s5e10, 2025. Kaggle.
```

---

## Contact & Links

[![GitHub](https://img.shields.io/badge/GitHub-181717?style=for-the-badge&logo=github&logoColor=white)](https://github.com/AdilShamim8)  
[![LinkedIn](https://img.shields.io/badge/LinkedIn-0077B5?style=for-the-badge&logo=linkedin&logoColor=white)](https://www.linkedin.com/in/adilshamim8)  
[![Twitter](https://img.shields.io/badge/Twitter-1DA1F2?style=for-the-badge&logo=twitter&logoColor=white)](https://x.com/adil_shamim8)  
[![Kaggle](https://img.shields.io/badge/Kaggle-20BEFF?style=for-the-badge&logo=kaggle&logoColor=white)](https://www.kaggle.com/adilshamim8)

---

**Public Notebook:** https://www.kaggle.com/code/adilshamim8/predicting-road-accident-risk-101
