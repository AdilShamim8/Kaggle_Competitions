# Binary Classification with a Bank Dataset - Playground Series S5E8

![Competition](https://img.shields.io/badge/Competition-Playground%20Series%20S5E8-blue)
![Score](https://img.shields.io/badge/Private%20LB%20Score-151-brightgreen)
![Ranking](https://img.shields.io/badge/Ranking-Top%205%25-success)
![Status](https://img.shields.io/badge/Status-Late%20Submission-orange)
![Metric](https://img.shields.io/badge/Metric-ROC--AUC-informational)

## Table of Contents

- [Overview](#overview)
- [Personal Results](#personal-results)
- [Competition Details](#competition-details)
- [Problem Description](#problem-description)
- [Solution Approach](#solution-approach)
- [Notebook Summary](#notebook-summary)
- [EDA Highlights](#eda-highlights)
- [Modeling](#modeling)
- [Dataset Information](#dataset-information)
- [Evaluation Metric](#evaluation-metric)
- [Timeline](#timeline)
- [Prizes](#prizes)
- [Resources & References](#resources--references)
---

## Overview

This Kaggle Tabular Playground Series competition (Season 5, Episode 8) focuses on predicting whether a client will subscribe to a bank term deposit. The dataset is synthetic but modeled after the Bank Marketing Dataset, enabling fast iteration on EDA, feature exploration, and model validation.

---

## Personal Results

| Metric | Value |
|--------|-------|
| **Private Leaderboard Score** | 151 |
| **Ranking** | Top 5% |
| **Submission Type** | Late Submission |
| **Evaluation Metric** | ROC-AUC |
| **Notebook** | [View Public Notebook](https://www.kaggle.com/code/adilshamim8/bank-term-deposit-prediction) |

---

## Competition Details

- **Series:** Playground Series - Season 5, Episode 8
- **Goal:** Predict whether a client subscribes to a term deposit
- **Metric:** ROC-AUC
- **Start Date:** August 1, 2025
- **Final Submission Deadline:** August 31, 2025
- **Dataset Type:** Synthetic data derived from the Bank Marketing Dataset

---

## Problem Description

Predict the probability of the binary target `y` for each test row. Submissions are evaluated using ROC-AUC between predicted probabilities and the observed target.

**Submission format**:

```csv
id,y
750000,0.5
750001,0.5
750002,0.5
```

---

## Solution Approach

The notebook follows a full end-to-end workflow:

1. **Introduction & Setup** - Imports, styling, and configuration.
2. **Data Loading** - Load competition data and the original bank marketing dataset.
3. **EDA** - Target analysis and detailed numeric and categorical feature exploration.
4. **Feature Insights** - Distribution plots, box/violin plots, and heatmaps for category interactions.
5. **Model Training** - CatBoost classifier with stratified K-fold CV.
6. **Submission** - Predict probabilities and export `submission.csv`.

---

## Notebook Summary

The notebook provides a structured tutorial with these sections:

1. **Introduction and Setup**
	- Competition overview and ROC-AUC metric
	- Library imports and visual styling
2. **Exploratory Data Analysis**
	- Custom numeric describe with skewness and kurtosis
	- Target distribution plots (count and pie)
	- Numeric feature analysis with box, violin, and KDE
	- Categorical plots and interaction heatmaps
3. **Model Training**
	- CatBoost classifier on GPU
	- 10-fold stratified CV with early stopping
	- OOF ROC-AUC reporting
4. **Submission**
	- Probability predictions for test rows
	- Save `submission.csv`
5. **Conclusion**
	- Summary of EDA findings and modeling insights

---

## EDA Highlights

- **Target imbalance**: minority class visible in count and pie plots.
- **Numeric features**: distributions analyzed with boxplots and KDE.
- **Categorical features**: count/pie charts for top categories.
- **Cross-feature patterns**: heatmaps for job vs education, marital vs education, and poutcome vs contact.

---

## Modeling

**Model:** CatBoostClassifier

- Categorical features handled natively by CatBoost
- 10-fold StratifiedKFold CV
- GPU training enabled
- Early stopping to prevent overfitting

**Workflow:**
1. Train on combined competition + original dataset
2. Cross-validate with ROC-AUC
3. Average test predictions across folds

---

## Dataset Information

- **Files:** `train.csv`, `test.csv`, `sample_submission.csv`
- **Columns:** 37
- **Size:** 89.88 MB
- **License:** Apache 2.0
- **Source:** Synthetic data generated from the Bank Marketing Dataset

---

## Evaluation Metric

The competition uses **ROC-AUC**, which measures ranking quality of predicted probabilities:

$$
	ext{AUC} = \int_0^1 \text{TPR}(\text{FPR})\, d\text{FPR}
$$

Higher is better. A score of 0.5 is random and 1.0 is perfect.

---

## Timeline

| Event | Date |
|-------|------|
| Competition Start | August 1, 2025 |
| Entry Deadline | August 31, 2025 |
| Team Merger Deadline | August 31, 2025 |
| Final Submission Deadline | August 31, 2025 |

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

- **Competition Overview:** https://www.kaggle.com/competitions/playground-series-s5e8/overview
- **Dataset:** https://www.kaggle.com/competitions/playground-series-s5e8/data
- **Public Notebook:** https://www.kaggle.com/code/adilshamim8/bank-term-deposit-prediction

**Citation**

```
Walter Reade and Elizabeth Park. Binary Classification with a Bank Dataset.
https://kaggle.com/competitions/playground-series-s5e8, 2025. Kaggle.
```

---
