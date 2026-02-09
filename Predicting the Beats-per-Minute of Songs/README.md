# Predicting the Beats-per-Minute of Songs - Playground Series S5E9

![Competition](https://img.shields.io/badge/Competition-Playground%20Series%20S5E9-blue)
![Score](https://img.shields.io/badge/Private%20LB%20Score-38-brightgreen)
![Ranking](https://img.shields.io/badge/Ranking-Top%202%25-success)
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

This Kaggle Tabular Playground Series competition (Season 5, Episode 9) focuses on predicting a song’s Beats Per Minute (BPM). The dataset is synthetic but modeled after a BPM prediction dataset, enabling fast iteration on EDA, feature engineering, and ensemble modeling.

---

## Personal Results

| Metric | Value |
|--------|-------|
| **Private Leaderboard Score** | 38 |
| **Ranking** | Top 2% |
| **Submission Type** | Late Submission |
| **Evaluation Metric** | RMSE |
| **Notebook** | [View Public Notebook](https://www.kaggle.com/code/adilshamim8/predicting-the-beats-per-minute-of-songs-101) |

---

## Competition Details

- **Series:** Playground Series - Season 5, Episode 9
- **Goal:** Predict continuous-valued BeatsPerMinute
- **Metric:** RMSE (Root Mean Squared Error)
- **Start Date:** September 1, 2025
- **Final Submission Deadline:** September 30, 2025
- **Dataset Type:** Synthetic data derived from the BPM Prediction Challenge dataset

---

## Problem Description

Predict the continuous target `BeatsPerMinute` for each test row. Submissions are evaluated using RMSE between predicted BPM and observed BPM.

**Submission format**:

```csv
ID,BeatsPerMinute
524164,119.5
524165,127.42
524166,111.11
```

---

## Solution Approach

The notebook follows a full ML pipeline:

1. **Environment Setup** - Imports, plotting style, and reproducibility settings.
2. **Data Loading** - Train/test ingestion and sample submission checks.
3. **EDA** - Target distribution, feature distributions, and correlation analysis.
4. **Feature Engineering** - Rhythm, duration, intensity, and interaction features.
5. **Preprocessing** - Robust scaling, CV folds, and target split.
6. **Model Training** - LightGBM, XGBoost, CatBoost, and Random Forest.
7. **Ensembling** - Weighted blending and stacking with Ridge meta-model.
8. **Submission** - Column validation, clipping, and export.

---

## Notebook Summary

The notebook is a tutorial-style walkthrough covering:

1. **Introduction** - BPM definition and competition objective.
2. **Data Exploration** - Feature distributions, missing values, and correlation matrix.
3. **Feature Engineering** - Creation of rhythm-energy, duration, and musical character features.
4. **Feature Importance** - LightGBM-based importance ranking.
5. **Preprocessing** - Robust scaling and 5-fold CV.
6. **Model Training** - LGBM, XGB, CatBoost, and RF blending.
7. **Performance Visualization** - Distribution, scatter, and residual plots.
8. **Stacking Ensemble** - Ridge meta-model vs blended baseline.
9. **Submission Validation** - Strict column checks and final verification.
10. **Conclusion** - Key insights and improvement ideas.

---

## Feature Engineering Highlights

The pipeline creates music-aware features, including:

- **Rhythm + Energy interactions:** `Rhythm_Energy`, `Rhythm_Loudness`
- **Duration transforms:** `Duration_Minutes`, `Duration_Energy_Ratio`, `Log_Duration`
- **Non-linear features:** squared terms for rhythm and energy
- **Musical character ratios:** `Acoustic_Instrumental_Ratio`, `Energy_Loudness_Ratio`
- **Composite metrics:** `Audio_Intensity`, `Performance_Character`
- **Mood and performance blends:** `Live_Energy`, `Mood_Rhythm`

---

## Modeling & Ensembling

**Base models:**
- LightGBM Regressor
- XGBoost Regressor
- CatBoost Regressor
- Random Forest Regressor

**Ensembling strategy:**
- Weighted average blend across folds
- Advanced stacking with Ridge meta-model
- Automatic selection between stacking and blending based on CV RMSE

---

## Dataset Information

- **Files:** `train.csv`, `test.csv`, `sample_submission.csv`
- **Columns:** 23
- **Size:** 83.41 MB
- **License:** CC BY 4.0
- **Source:** Synthetic data generated from the BPM Prediction Challenge dataset

---

## Evaluation Metric

The competition uses **RMSE**:

$$
	ext{RMSE} = \sqrt{\frac{1}{n} \sum_{i=1}^n (y_i - \hat{y}_i)^2}
$$

Lower is better. RMSE is in BPM units.

---

## Timeline

| Event | Date |
|-------|------|
| Competition Start | September 1, 2025 |
| Entry Deadline | September 30, 2025 |
| Team Merger Deadline | September 30, 2025 |
| Final Submission Deadline | September 30, 2025 |

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

- **Competition Overview:** https://www.kaggle.com/competitions/playground-series-s5e9/overview
- **Dataset:** https://www.kaggle.com/competitions/playground-series-s5e9/data
- **Public Notebook:** https://www.kaggle.com/code/adilshamim8/predicting-the-beats-per-minute-of-songs-101
