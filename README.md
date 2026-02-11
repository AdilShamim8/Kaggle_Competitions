<div align="center">
    
# Kaggle Competitions

![Kaggle](https://img.shields.io/badge/Kaggle-Competitions-20BEFF?logo=kaggle&logoColor=white)
![Python](https://img.shields.io/badge/Python-3.x-3776AB?logo=python&logoColor=white)
![Machine Learning](https://img.shields.io/badge/Machine%20Learning-Deep%20Learning-FF6F00?logo=tensorflow&logoColor=white)
![Status](https://img.shields.io/badge/Status-Active-success)

**A collection of my Kaggle competition solutions, notebooks, and learnings**

[My Kaggle Profile](https://www.kaggle.com/adilshamim8)

</div>

---

##  Table of Contents

- [About](#-about)
- [Competitions](#-competitions)
- [Repository Structure](#-repository-structure)
- [License](#-license)

---

##  About

This repository contains my solutions and approaches to various Kaggle competitions. Each competition folder includes:

-  Jupyter notebooks with complete solution code
-  Detailed README explaining the approach
-  Model architectures and training configurations
-  Results and performance metrics

---

## Competitions

| Competition | Description | Result | Approach |
|-------------|-------------|--------|----------|
| [**NeurIPS 2025 - Google Code Golf Championship**](./NeurIPS%202025%20-%20Google%20Code%20Golf%20Championship/) | Shortest code wins! Google Code Golf at NeurIPS 2025 - Write the most concise programs to solve challenging tasks. | **Top 24%** (Score: 264) | Deep pattern analysis, hypothesis-driven solution development, code golf optimization with batch compression & verification optimizer for 400 ARC-AGI tasks |
| [**MABe Challenge - Social Action Recognition in Mice**](./MABe%20Challenge%20-%20Social%20Action%20Recognition%20in%20Mice/) | Detect unique behaviors from pose estimates of mice across 400+ hours of footage | **Top 11%** (Score: 145) | LightGBM + XGBoost + CatBoost Ensemble with separate GBDT models per `body_parts_tracked` config, extensive feature engineering (distance, velocity, angle, wavelet features), and GroupKFold CV |
| [**Fake or Real: The Impostor Hunt in Texts**](./Fake%20or%20Real%20The%20Impostor%20Hunt%20in%20Texts/) | ESA's "Secure Your AI" series - Identify real vs. fake LLM-generated documents | **Top 12%** (Score: 114) | 5-Head DeBERTa-v3-Large Ensemble (Mean Pool, Max Pool, CLS Token, Attention Pool, Concat Pool) with Optuna-optimized voting, achieving ~0.925 CV accuracy |
| [**ARC Prize 2025**](./ARC%20Prize%202025/) | Advanced Reasoning Challenge - Solve ARC tasks with ML/AI | **Top 29%** (Score: 413) | Multi-stage pipeline combining DSL-based symbolic program search (DFS), CNN-based neural-guided program synthesis (SketchPredictor), and adaptive task classification with fallback predictions |
| [**Predicting Student Test Scores**](./Predicting%20Student%20Test%20Scores/) | Predicting standardized test outcomes for students | **Top 12%** (Score: 478) | Regression with feature engineering, model selection & ensemble strategy, evaluated on RMSE (Playground Series S6E1) |
| [**Predicting Loan Payback**](./Predicting%20Loan%20Payback/) | Loan default risk prediction | **Top 11%** (Score: 404) | 12-stage feature engineering pipeline (risk, affordability, ratios, bins, interactions), multi-model ensemble with configurable weights and folds, ROC-AUC optimized |
| [**Predicting Road Accident Risk**](./Predicting%20Road%20Accident%20Risk/) | Predict road accident risk on different road types | **Top 1%** (Score: 29) | XGBoost + LightGBM + CatBoost weighted blending with interaction features (speed_curvature, lanes_speed, curvature_squared), weather+lighting risk flags, and RandomizedSearchCV tuning |
| [**Binary Classification with a Bank Dataset**](./Binary%20Classification%20with%20a%20Bank%20Dataset/) | Predict bank customer responses for targeted marketing | **Top 5%** (Score: 151) | CatBoost classifier with native categorical handling, 10-fold Stratified CV, GPU training, early stopping, trained on combined competition + original Bank Marketing dataset |
| [**Predicting the Beats-per-Minute of Songs**](./Predicting%20the%20Beats-per-Minute%20of%20Songs/) | Predict the tempo (BPM) of songs using audio/music features | **Top 2%** (Score: 38) | LightGBM + XGBoost + CatBoost + Random Forest ensemble with weighted blending & Ridge meta-model stacking, rhythm/duration/intensity feature engineering |

---

## Repository Structure

```
Kaggle_Competitions/
├── README.md                                           # This file
├── LICENSE                                             # MIT License
│
├── NeurIPS 2025 - Google Code Golf Championship/       # NeurIPS 2025 Code Golf
│   ├── README.md                                       # Detailed solution writeup
│   └── google-code-golf-championship-101.ipynb         # Main notebook
│
├── MABe Challenge - Social Action Recognition in Mice/ # Mouse Behavior Competition
│   ├── README.md                                       # Detailed solution writeup
│   └── mabe-challenge.ipynb                            # Main notebook
│
├── Fake or Real The Impostor Hunt in Texts/            # ESA NLP Competition
│   ├── README.md                                       # Detailed solution writeup
│   └── fake-or-real-the-impostor-hunt-in-texts.ipynb   # Main notebook
│
├── ARC Prize 2025/                                     # ARC Prize 2025
│   ├── README.md                                       # Detailed solution writeup
│   └── arc-prize-2025-comprehensive-solution.ipynb     # Main notebook
│
├── Predicting Student Test Scores/                     # Student Score Prediction
│   ├── README.md                                       # Detailed solution writeup
│   └── student_test_scores.ipynb                       # Main notebook
│
├── Predicting Loan Payback/                            # Loan Payback Prediction
│   ├── README.md                                       # Detailed solution writeup
│   └── predicting-loan-payback.ipynb                   # Main notebook
│
├── Predicting Road Accident Risk/                      # Road Accident Risk Prediction
│   ├── README.md                                       # Detailed solution writeup
│   └── predicting-road-accident-risk-101.ipynb         # Main notebook
│
├── Binary Classification with a Bank Dataset/          # Bank Term Deposit Prediction
│   ├── README.md                                       # Detailed solution writeup
│   └── bank-term-deposit-prediction.ipynb              # Main notebook
│
└── Predicting the Beats-per-Minute of Songs/           # BPM Prediction
    ├── README.md                                       # Detailed solution writeup
    └── predicting-the-beats-per-minute-of-songs-101.ipynb   # Main notebook
```

##  Contributing

Feel free to:
-  Star this repository if you find it helpful
-  Open issues for questions or suggestions
-  Fork and submit PRs for improvements

---

## License

This project is licensed under the MIT License - see the [LICENSE](LICENSE) file for details.

---

## Contact & Links

Feel free to connect and follow my work:

[![GitHub](https://img.shields.io/badge/GitHub-181717?style=for-the-badge&logo=github&logoColor=white)](https://github.com/AdilShamim8)  
[![LinkedIn](https://img.shields.io/badge/LinkedIn-0077B5?style=for-the-badge&logo=linkedin&logoColor=white)](https://www.linkedin.com/in/adilshamim8)  
[![Twitter](https://img.shields.io/badge/Twitter-1DA1F2?style=for-the-badge&logo=twitter&logoColor=white)](https://x.com/adil_shamim8)  
[![Kaggle](https://img.shields.io/badge/Kaggle-20BEFF?style=for-the-badge&logo=kaggle&logoColor=white)](https://www.kaggle.com/adilshamim8)

---

<div align="center">

**Happy Kaggling!**

*"The best way to learn is by competing."*

<!-- <img src="https://media.giphy.com/media/3o7btPCcdNniyf0ArS/giphy.gif" width="200" alt="Detective">  -->

</div>
