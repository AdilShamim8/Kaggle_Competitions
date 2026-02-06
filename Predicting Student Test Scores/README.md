# Predicting Student Test Scores - Kaggle Playground Series S6E1

![Competition](https://img.shields.io/badge/Competition-Playground%20Series%20S6E1-blue)
![Score](https://img.shields.io/badge/Private%20LB%20Score-478-brightgreen)
![Ranking](https://img.shields.io/badge/Ranking-Top%2012%25-success)
![Status](https://img.shields.io/badge/Status-Late%20Submission-orange)
![Metric](https://img.shields.io/badge/Metric-RMSE-informational)

## Table of Contents

- [Overview](#overview)
- [Personal Results](#personal-results)
- [Competition Details](#competition-details)
- [Problem Description](#problem-description)
- [Solution Approach](#solution-approach)
- [Notebook Summary](#notebook-summary)
- [Feature Engineering](#feature-engineering)
- [Model Selection & Ensemble](#model-selection--ensemble)
- [Dataset Information](#dataset-information)
- [Evaluation Metrics](#evaluation-metrics)
- [Timeline](#timeline)
- [Prizes](#prizes)
- [Resources & References](#resources--references)

---

## Overview

The **Tabular Playground Series - Season 6 Episode 1** challenges participants to predict students' exam scores using machine learning regression techniques. This competition is part of Kaggle's monthly series designed to provide approachable datasets for practicing ML skills with synthetic data generated from deep learning models.

**Goal:** Predict students' `exam_score` using Root Mean Squared Error (RMSE) as the evaluation metric.

---

## Personal Results

| Metric | Value |
|--------|-------|
| **Private Leaderboard Score** | 478 |
| **Ranking** | Top 12% |
| **Submission Type** | Late Submission |
| **Evaluation Metric** | RMSE (Root Mean Squared Error) |
| **Notebook** | [View Public Notebook](https://www.kaggle.com/code/adilshamim8/predicting-student-test-scores-101) |

---

## Competition Details

### About Tabular Playground Series

The **Tabular Playground Series** provides the Kaggle community with lightweight challenges to learn and sharpen skills in machine learning and data science. Each competition typically lasts a few weeks and focuses on different aspects of ML problem-solving.

**Key Competition Information:**
- **Host:** Kaggle
- **Series:** Playground Series - Season 6 Episode 1
- **Start Date:** January 1, 2026
- **Submission Deadline:** January 31, 2026
- **Duration:** 1 month
- **Difficulty:** Beginner-Friendly

### Synthetically-Generated Dataset

The dataset for this competition was generated synthetically from a deep learning model trained on the [Exam Score Prediction Dataset](https://www.kaggle.com/datasets). The synthetic generation creates a beginner-friendly learning environment while maintaining realistic feature distributions and relationships.

**Note:** Participants are encouraged to use the original dataset to:
- Explore distribution differences
- Augment training data
- Improve model performance

---

## Problem Description

### The Challenge

Given a set of student-related features, predict the **exam_score** (continuous target variable) for students in the test set.

### Prediction Task

- **Type:** Regression
- **Target Variable:** `exam_score` (continuous numerical value)
- **Metric:** Root Mean Squared Error (RMSE)
- **Attempts:** 2 predictions per student ID required

### Key Objectives

1. **Understand Data Patterns** - Explore relationships between features and exam scores
2. **Feature Engineering** - Create meaningful features capturing student performance indicators
3. **Model Selection** - Evaluate multiple regression algorithms
4. **Hyperparameter Optimization** - Fine-tune models for best performance
5. **Ensemble Methods** - Combine models to reduce prediction variance

---

## Solution Approach

### Comprehensive 10-Step Pipeline

The solution implements a complete machine learning pipeline with the following stages:

#### 1. **Setup & Imports**
- Core libraries: NumPy, Pandas
- Visualization: Matplotlib, Seaborn
- ML algorithms: scikit-learn, XGBoost, LightGBM, CatBoost
- Optimization: Optuna for hyperparameter tuning

#### 2. **Load Data**
- Training set: Features + target (`exam_score`)
- Test set: Features only
- Sample submission: Format reference

#### 3. **Exploratory Data Analysis (EDA)**

**Comprehensive data exploration including:**

- **Target Distribution Analysis**
  - Histogram, box plot, and KDE visualization
  - Mean, median, and distribution shape analysis
  - Skewness and normality checks

- **Missing Values Analysis**
  - Identification of missing data patterns
  - Percentage calculation for train and test sets
  - Visualization of missing value distributions

- **Numerical Features Distribution**
  - Multi-panel histograms for all numerical features
  - Distribution shape analysis

- **Categorical Features Analysis**
  - Unique value counts
  - Category frequency distributions

- **Correlation Analysis**
  - Feature-target correlation ranking
  - Multicollinearity detection
  - Heatmap visualization

- **Outlier Detection**
  - IQR (Interquartile Range) method
  - Box plot visualization
  - Outlier percentage calculation

- **Train vs Test Comparison**
  - Distribution alignment verification
  - Domain shift detection

#### 4. **Data Preprocessing**

**Separation:**
- Features (X) and target (y)
- ID columns stored for submission

**Missing Value Handling:**
- Numerical features: Median imputation (robust to outliers)
- Categorical features: Mode imputation (most frequent value)

**Encoding:**
- Label encoding for categorical variables
- Combined train+test encoding to handle unseen categories

**Scaling:**
- StandardScaler for feature normalization
- Both scaled and unscaled versions maintained
  - Scaled: For linear models, SVM, KNN
  - Unscaled: For tree-based models (XGBoost, LightGBM, CatBoost)

#### 5. **Feature Engineering**

Advanced feature creation to capture hidden patterns:

**Statistical Features:**
- `mean_features`: Row-wise mean across numerical features
- `std_features`: Row-wise standard deviation
- `min_features`, `max_features`: Minimum and maximum values
- `range_features`: Max - Min
- `median_features`: Row-wise median

**Polynomial Features:**
- Squared features: `feature_squared`
- Square root: `feature_sqrt`
- Logarithmic: `feature_log`

**Interaction Features:**
- Multiplication: `feature1_x_feature2`
- Division: `feature1_div_feature2` (with epsilon for stability)

**Binning Features:**
- Quantile-based binning using `pd.qcut()`
- Converts continuous to categorical representations

**Runtime Optimization:**
- `FAST_MODE` flag for reduced feature set in testing
- Configurable feature count limits

#### 6. **Model Building**

**Train-Validation Split:**
- 80% training, 20% validation
- Random state for reproducibility

**Model Portfolio:**

| Category | Models |
|----------|--------|
| **Linear Models** | Linear Regression, Ridge, Lasso, ElasticNet |
| **Tree-Based** | Decision Tree, Random Forest, Extra Trees |
| **Gradient Boosting** | XGBoost, LightGBM, CatBoost  |
| **Other** | KNN, SVR |

**Evaluation:**
- RMSE (Root Mean Squared Error)
- MAE (Mean Absolute Error)
- R² Score
- Performance ranking and visualization

#### 7. **Cross-Validation**

**Strategy:**
- K-Fold Cross-Validation (3-5 folds)
- Top models selected for CV evaluation
- Mean and standard deviation RMSE calculation
- Fold-level score tracking

**Benefits:**
- More reliable performance estimates
- Reduced overfitting risk
- Model stability assessment

#### 8. **Hyperparameter Tuning**

**Optuna Optimization Framework:**

**XGBoost Tuning:**
- Parameters: `n_estimators`, `max_depth`, `learning_rate`, `subsample`, `colsample_bytree`, `min_child_weight`, `reg_alpha`, `reg_lambda`
- Search space: Log-scale for learning rates, integer ranges for tree parameters
- Objective: Minimize CV RMSE

**LightGBM Tuning:**
- Parameters: `n_estimators`, `max_depth`, `learning_rate`, `subsample`, `colsample_bytree`, `min_child_samples`, `reg_alpha`, `reg_lambda`, `num_leaves`
- Additional: Leaf-based tree control parameters
- Advanced pruning strategies

**CatBoost Tuning:**
- Parameters: `n_estimators`, `max_depth`, `learning_rate`, `subsample`, `colsample_bylevel`, `min_child_samples`, `l2_leaf_reg`
- L2 regularization for leaf values
- Categorical feature handling

**Tuning Configuration:**
- TPESampler for efficient search
- 10-50 trials (FAST_MODE dependent)
- Cross-validation within Optuna objectives

#### 9. **Ensemble Methods**

**Individual Model Training:**
- Train tuned XGBoost, LightGBM, CatBoost on full dataset
- Generate predictions for test set

**Ensemble Strategies:**

1. **Simple Averaging:**
   ```python
   ensemble_avg = (pred_xgb + pred_lgbm + pred_catboost) / 3
   ```

2. **Weighted Averaging:**
   ```python
   ensemble_weighted = (0.35 * pred_xgb + 
                        0.35 * pred_lgbm + 
                        0.30 * pred_catboost)
   ```
   - Weights based on CV performance
   - Adjustable for optimization

**Prediction Analysis:**
- Distribution visualization
- Min/max/mean statistics
- Outlier detection

#### 10. **Final Predictions & Submission**

**Submission Files Created:**
- `submission_xgb.csv` - XGBoost only
- `submission_lgbm.csv` - LightGBM only
- `submission_catboost.csv` - CatBoost only
- `submission_ensemble.csv` - Weighted ensemble (Recommended ⭐)

**Format Validation:**
- Column name verification
- ID count matching
- NaN and infinite value checks
- Sample submission format compliance

---

## Notebook Summary

### Comprehensive Solution Tutorial

The [public notebook](https://www.kaggle.com/code/adilshamim8/predicting-student-test-scores-101) provides a complete walkthrough with **10 major sections**:

#### Section Breakdown

**1. Setup & Imports** (Cells 1-4)
- Library imports and configuration
- `FAST_MODE` runtime controls
- Reproducibility settings (`RANDOM_STATE = 42`)

**2. Load Data** (Cells 5-6)
- Data loading from competition files
- Shape and structure verification

**3. Exploratory Data Analysis** (Cells 7-22)
- 10 comprehensive sub-sections
- 15+ visualizations saved as PNG files
- Statistical analysis and insights

**4. Data Preprocessing** (Cells 23-27)
- Feature-target separation
- Missing value imputation
- Categorical encoding
- Feature scaling

**5. Feature Engineering** (Cells 28-29)
- `create_features()` function
- 30+ new features created
- Infinity and NaN handling

**6. Model Building** (Cells 30-34)
- Train-validation split
- 11 models defined
- Performance evaluation and ranking
- Horizontal bar chart comparison

**7. Cross-Validation** (Cells 35-36)
- K-Fold CV on top models
- Mean RMSE with std deviation
- Error bar visualization

**8. Hyperparameter Tuning** (Cells 37-42)
- Optuna optimization for XGBoost
- Optuna optimization for LightGBM
- Optuna optimization for CatBoost
- Best parameter extraction
- Fast-mode fallback defaults

**9. Ensemble Methods** (Cells 43-48)
- Model training on full dataset
- Individual predictions
- Simple and weighted averaging
- Distribution visualization (2×2 grid)

**10. Final Predictions & Submission** (Cells 49-52)
- Submission dataframe creation
- Format validation
- File export (4 CSV files)

**Summary & Next Steps** (Cells 53-54)
- Pipeline recap
- Improvement suggestions
- Contact information

### Key Code Highlights

**Feature Engineering Function:**
```python
def create_features(df):
    # Statistical aggregations
    df['mean_features'] = df[num_cols].mean(axis=1)
    df['std_features'] = df[num_cols].std(axis=1)
    
    # Polynomial features
    df[f'{col}_squared'] = df[col] ** 2
    df[f'{col}_sqrt'] = np.sqrt(np.abs(df[col]))
    
    # Interaction features
    df[f'{col1}_x_{col2}'] = df[col1] * df[col2]
    
    # Binning
    df[f'{col}_binned'] = pd.qcut(df[col], q=10, labels=False)
    
    return df
```

**Optuna Objective Function:**
```python
def objective_xgb(trial):
    params = {
        'n_estimators': trial.suggest_int('n_estimators', 100, 600),
        'max_depth': trial.suggest_int('max_depth', 3, 12),
        'learning_rate': trial.suggest_float('learning_rate', 0.01, 0.2, log=True),
        # ... more parameters
    }
    model = XGBRegressor(**params)
    cv_scores = cross_val_score(model, X_fe, y, cv=3, 
                                scoring='neg_root_mean_squared_error')
    return -cv_scores.mean()
```

**Weighted Ensemble:**
```python
ensemble_weighted = (0.35 * pred_xgb + 
                     0.35 * pred_lgbm + 
                     0.30 * pred_catboost)
```

---

## Feature Engineering

### Categories of Features Created

#### 1. Statistical Aggregation Features
- **Row-wise statistics** across numerical columns
- Mean, standard deviation, min, max, range, median
- Captures overall magnitude and variability

#### 2. Polynomial Features
- **Power transformations** to capture non-linear relationships
- Squared, square root, logarithmic transformations
- First 3-5 columns (FAST_MODE dependent)

#### 3. Interaction Features
- **Pairwise combinations** of features
- Multiplication and division operations
- Captures feature dependencies and ratios

#### 4. Binning Features
- **Discretization** of continuous features
- Quantile-based binning (6-10 bins)
- Converts numerical to ordinal categorical

### Feature Count Expansion

| Stage | Feature Count |
|-------|---------------|
| **Original Features** | ~27 columns |
| **After Feature Engineering** | 60+ columns |
| **New Features Created** | 30+ features |

---

## Model Selection & Ensemble

### Model Performance Hierarchy

Based on cross-validation results, the typical performance ranking:

**Tier 1 - Gradient Boosting (Best Performers):**
- 🥇 **LightGBM** - Fast training, handles large datasets
- 🥈 **XGBoost** - Robust, widely used in competitions
- 🥉 **CatBoost** - Excellent with categorical features

**Tier 2 - Tree Ensembles:**
- Random Forest - Good baseline
- Extra Trees - High variance reduction

**Tier 3 - Linear Models:**
- Ridge, Lasso, ElasticNet - Good for feature selection
- Linear Regression - Baseline reference

**Tier 4 - Other:**
- KNN, SVR - Computationally expensive

### Why Gradient Boosting Wins

1. **Sequential Learning** - Each tree corrects previous errors
2. **Regularization** - Built-in overfitting prevention
3. **Feature Interactions** - Automatically captures complex patterns
4. **Handling Missing Data** - Native support
5. **Competition Proven** - Dominates Kaggle leaderboards

### Ensemble Rationale

**Why Ensemble:**
- **Variance Reduction** - Different models make different errors
- **Bias-Variance Trade-off** - Balances model diversity
- **Robustness** - More stable predictions
- **Competition Edge** - 0.5-2% improvement typical

**Weighted vs Simple Average:**
- Weights reflect individual model strength
- Based on cross-validation RMSE
- Fine-tunable for optimal performance

---

## Dataset Information

### Files

| File | Rows | Columns | Size | Description |
|------|------|---------|------|-------------|
| `train.csv` | 630,000 | 27 | ~64 MB | Training data with target |
| `test.csv` | 270,000 | 26 | ~27 MB | Test data without target |
| `sample_submission.csv` | 270,000 | 2 | 2.43 MB | Submission format |

### Dataset Characteristics

**Training Set:**
- **Samples:** 630,000 students
- **Features:** 26 (excluding `id` and `exam_score`)
- **Target:** `exam_score` (continuous variable)

**Test Set:**
- **Samples:** 270,000 students
- **Features:** 26 (excluding `id`)
- **Prediction:** Required for all test IDs

**Feature Types:**
- Numerical features
- Categorical features (if any)
- Mixed data types

### Original Dataset

The synthetic data was generated from the **Exam Score Prediction Dataset**. Key differences:
- Feature distributions are close but not identical
- Synthetic data may contain artifacts
- Original data can be used for training augmentation

### Data Exploration Findings

**Target Distribution:**
- Shape: Approximately normal
- Range: Varies by dataset
- Mean/Median: Calculated in EDA

**Missing Values:**
- Pattern: Analyzed per column
- Strategy: Median for numerical, mode for categorical

**Correlations:**
- Feature-target correlations ranked
- Multicollinearity detected via heatmap

**Train-Test Alignment:**
- Distribution comparison visualized
- Minimal domain shift expected

---

## Evaluation Metrics

### Primary Metric: RMSE

**Root Mean Squared Error (RMSE):**

$$
\text{RMSE} = \sqrt{\frac{1}{n}\sum_{i=1}^{n}(y_i - \hat{y}_i)^2}
$$

Where:
- $n$ = number of predictions
- $y_i$ = actual exam score
- $\hat{y}_i$ = predicted exam score

**Interpretation:**
- **Lower is better** - Smaller RMSE indicates better predictions
- **Same units as target** - RMSE in exam score points
- **Penalizes large errors** - Squared term emphasizes outliers

### Submission Format

```csv
id,exam_score
630000,97.5
630001,89.2
630002,85.5
...
```

**Requirements:**
- Header row: `id,exam_score`
- One prediction per test ID
- Numerical predictions (continuous)
- No missing values

---

## Timeline

| Event | Date |
|-------|------|
| Competition Start | January 1, 2026 |
| Entry Deadline | January 31, 2026 |
| Team Merger Deadline | January 31, 2026 |
| Final Submission Deadline | January 31, 2026 |

**Note:** All deadlines are at 11:59 PM UTC

---

## Prizes

### Kaggle Merchandise Awards

| Place | Prize |
|-------|-------|
| 🥇 1st Place | Choice of Kaggle merchandise |
| 🥈 2nd Place | Choice of Kaggle merchandise |
| 🥉 3rd Place | Choice of Kaggle merchandise |

**Important Note:**
- Merchandise awarded **once per person** in the series
- Encourages participation from beginners
- If previous winner, prize goes to next team

---

## Resources & References

### Competition Resources

- **Main Competition:** [Playground Series S6E1](https://www.kaggle.com/competitions/playground-series-s6e1)
- **Dataset:** [Competition Data](https://www.kaggle.com/competitions/playground-series-s6e1/data)
- **Original Dataset:** [Exam Score Prediction Dataset](https://www.kaggle.com/datasets)
- **Discussion Forum:** [Community Discussions](https://www.kaggle.com/competitions/playground-series-s6e1/discussion)

### Key Libraries & Documentation

- **scikit-learn:** [https://scikit-learn.org](https://scikit-learn.org)
- **XGBoost:** [https://xgboost.readthedocs.io](https://xgboost.readthedocs.io)
- **LightGBM:** [https://lightgbm.readthedocs.io](https://lightgbm.readthedocs.io)
- **CatBoost:** [https://catboost.ai](https://catboost.ai)
- **Optuna:** [https://optuna.org](https://optuna.org)

### Learning Resources

- **Kaggle Learn:** [Machine Learning Courses](https://www.kaggle.com/learn)
- **Feature Engineering Guide:** [Kaggle Feature Engineering](https://www.kaggle.com/learn/feature-engineering)
- **Gradient Boosting:** [XGBoost Tutorial](https://www.kaggle.com/learn/intro-to-machine-learning)

---

## Your Public Notebook

**Comprehensive Solution Tutorial:**  
[Predicting Student Test Scores 101](https://www.kaggle.com/code/adilshamim8/predicting-student-test-scores-101)

This notebook demonstrates:
- ✅ Complete 10-step ML pipeline
- ✅ 15+ EDA visualizations
- ✅ Advanced feature engineering
- ✅ Multiple model comparison
- ✅ Optuna hyperparameter tuning
- ✅ Ensemble methods
- ✅ Production-ready code with FAST_MODE
- ✅ Detailed comments and explanations

**Notebook Highlights:**
- **Educational Focus** - Beginner-friendly with extensive comments
- **Modular Design** - Reusable functions and clear structure
- **Runtime Optimized** - FAST_MODE for quick experimentation
- **Visual Insights** - High-quality plots saved as PNG
- **Best Practices** - Cross-validation, proper train-test split, reproducibility

---

## Summary & Key Takeaways

### Competition Pipeline Achievement

✅ **Complete ML Pipeline** - 10-stage end-to-end solution  
✅ **Top 12% Ranking** - RMSE score of 478  
✅ **Advanced Techniques** - Feature engineering, hyperparameter tuning, ensembling  
✅ **Multiple Models** - 11+ algorithms evaluated  
✅ **Optimized Solution** - Optuna for efficient hyperparameter search  
✅ **Ensemble Methods** - Weighted averaging of top performers  
✅ **Production Quality** - FAST_MODE, error handling, validation checks  

### Key Success Factors

1. **Comprehensive EDA** - 30-40% time spent understanding data deeply
2. **Feature Engineering** - 30+ new features capturing hidden patterns
3. **Gradient Boosting Focus** - XGBoost, LightGBM, CatBoost as core models
4. **Cross-Validation** - Reliable performance estimation
5. **Hyperparameter Tuning** - Optuna for smart optimization
6. **Ensemble Strategy** - Weighted averaging for variance reduction

### Next Steps for Improvement

**Data Augmentation:**
- Incorporate original exam score dataset
- Synthetic-original distribution alignment

**Advanced Feature Engineering:**
- Domain-specific features
- Target encoding for categorical features
- Feature selection (RFE, permutation importance)

**Model Improvements:**
- Stacking with meta-learner
- Blending multiple ensemble strategies
- Neural network regression models

**Hyperparameter Optimization:**
- Increase Optuna trials (50-100+)
- Bayesian optimization alternatives
- Grid search for final fine-tuning

**Post-Processing:**
- Prediction clipping to valid ranges
- Outlier handling in predictions
- Calibration techniques

**Experiment Tracking:**
- MLflow or Weights & Biases integration
- Systematic A/B testing
- Version control for models

### Competition Learnings

**Playground Series Benefits:**
- **Beginner-Friendly** - Approachable datasets, clear metrics
- **Monthly Practice** - Regular skill-building opportunities
- **Community Learning** - Discussion forums, shared notebooks
- **Synthetic Data** - Safe experimentation environment
- **Quick Iterations** - 1-month timeframe encourages rapid experimentation

**Gradient Boosting Dominance:**
- Consistently outperforms other algorithms
- Handles mixed data types well
- Native missing value support
- Competition-winning track record

**Ensemble Power:**
- 0.5-2% performance boost typical
- Reduces prediction variance
- Combines diverse model strengths
- Industry-standard approach

---

**Thank you for exploring this solution! Feel free to fork, adapt, and improve upon this approach. Happy Kaggling!**
