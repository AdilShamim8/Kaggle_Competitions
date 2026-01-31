# Fake or Real: The Impostor Hunt in Texts

<div align="center">

![Kaggle Competition](https://img.shields.io/badge/Kaggle-Competition-blue?logo=kaggle)
![ESA](https://img.shields.io/badge/ESA-European%20Space%20Agency-navy)
![Private LB](https://img.shields.io/badge/Private%20LB%20Score-114-green)
![Top 12%](https://img.shields.io/badge/Ranking-Top%2012%25-gold)
![Python](https://img.shields.io/badge/Python-3.x-blue?logo=python)
![PyTorch](https://img.shields.io/badge/PyTorch-2.x-orange?logo=pytorch)
![HuggingFace](https://img.shields.io/badge/🤗-Transformers-yellow)

**Part of the "Secure Your AI" series of competitions by the European Space Agency**

[🏆 Competition Link](https://www.kaggle.com/competitions/fake-or-real-the-impostor-hunt) | [📊 Dataset](https://www.kaggle.com/competitions/fake-or-real-the-impostor-hunt/data)

</div>

---

## Table of Contents

- [Competition Overview](#-competition-overview)
- [My Results](#-my-results)
- [Solution Approach](#-solution-approach)
- [Dataset Description](#-dataset-description)
- [Model Architecture](#-model-architecture)
- [Ensemble Methods](#-ensemble-methods)
- [Training Configuration](#-training-configuration)
- [File Structure](#-file-structure)
- [Usage](#-usage)
- [Timeline](#-timeline)
- [References](#-references)

---

## Competition Overview

### Background

ESA's European Space Operations Centre initiated a project to outline a clear direction for data strategy and AI implementation in mission operations. This initiative led to the creation of the **DataX** strategy, developed to improve the scalability of advanced analytics applications, including AI systems. This competition addresses two real-life AI security threats: **data poisoning** and **overreliance on text data**.

### Story

> A company uses LLM(s) to perform operations on official documents (e.g., summarization). Different models are used, and the history of which model performed which operation is not strictly tracked. It was discovered that sometimes models malfunction and may provide harmful output (hallucination, hidden triggers, changing facts). Your task as the company's data scientist is to distinguish between real and fake documents.

### Task

Given pairs of documents where one is **Real** (optimal for the recipient, as close as possible to the hidden original text) and one is **Fake** (more distant from the hidden original text), identify which document in each pair is the real one.

The documents cover space-related research, scientific devices, research outcomes, space workshops, and notable figures in engineering, science, and astronauts.

### Evaluation Metric

Submissions are evaluated using **Pairwise Accuracy** - how well models align with human preferences in choosing the correct text from each pair.

---

## My Results

| Metric | Score |
|--------|-------|
| **Private Leaderboard Score** | **114** |
| **Ranking** | **Top 12%** |
| **CV Accuracy** | ~0.92531 |

---

## Solution Approach

### 5-Head DeBERTa Ensemble

My solution implements a powerful ensemble approach using **5 different DeBERTa-v3-Large models**, each with a unique pooling head strategy. The ensemble combines predictions using three different methods to achieve optimal performance.

### Why This Approach Works

1. **Diverse Representations**: Different pooling strategies capture different aspects of the text
2. **Ensemble Power**: Combining multiple models reduces variance and improves robustness
3. **Optimized Weights**: Optuna finds the optimal contribution of each model
4. **Large Pre-trained Model**: DeBERTa-v3-Large provides strong language understanding

---

## Dataset Description

### Overview

The data comes from **The Messenger** journal articles. Both texts in each sample (Real and Fake) have been significantly modified using LLMs.

### Structure

```
data/
├── train/                    # 95 article directories
│   ├── article_0001/
│   │   ├── file_1.txt       # Text 1 (Real or Fake)
│   │   └── file_2.txt       # Text 2 (Fake or Real)
│   └── ...
├── test/                     # 1068 article directories
│   └── ...
└── train.csv                 # Ground truth labels
```

### Dataset Statistics

| Split | Articles | Files | Size |
|-------|----------|-------|------|
| Train | 95 | 190 | - |
| Test | 1068 | 2136 | - |
| **Total** | **1163** | **2327** | **5.15 MB** |

### train.csv Format

| Column | Description |
|--------|-------------|
| `id` | Article identifier |
| `real_text_id` | Which file (1 or 2) contains the real text |

---

## Model Architecture

### DeBERTa-v3-Large Backbone

- **Model**: `microsoft/deberta-v3-large`
- **Parameters**: 304M
- **Architecture**: Decoding-enhanced BERT with Disentangled Attention

### 5 Pooling Strategies

| Pooling Type | Description | Strengths |
|-------------|-------------|-----------|
| **Mean Pooling** | Average of all token embeddings | Captures overall semantic meaning |
| **Max Pooling** | Maximum value across all tokens | Highlights most salient features |
| **CLS Token** | Uses the [CLS] token representation | Standard BERT-style approach |
| **Attention Pooling** | Learnable attention weights over tokens | Focuses on important words |
| **Concat Pooling** | Concatenates CLS + Mean + Max | Combines multiple representations |

### Architecture Diagram

```
Input Text
    │
    ▼
┌─────────────────────────┐
│  DeBERTa-v3-Large       │
│  Encoder (304M params)  │
└─────────────────────────┘
    │
    ▼ (Last Hidden State)
    │
    ├──► Mean Pool ──► Classifier ──► Logits
    ├──► Max Pool  ──► Classifier ──► Logits
    ├──► CLS Token ──► Classifier ──► Logits
    ├──► Attention ──► Classifier ──► Logits
    └──► Concat    ──► Classifier ──► Logits
    │
    ▼
┌─────────────────────────┐
│   Ensemble Aggregation  │
│ (Average/Vote/Optuna)   │
└─────────────────────────┘
    │
    ▼
Final Prediction
```

---

## Ensemble Methods

### 1. Simple Averaging
Average probabilities from all 5 models:
```
P_final = (P_mean + P_max + P_cls + P_attention + P_concat) / 5
```

### 2. Majority Voting
Each model votes (probability > 0.5 = Real), majority wins.

### 3. Optuna Optimized Weights
Learns optimal weights for each model using Optuna hyperparameter optimization:
```
P_final = w1*P_mean + w2*P_max + w3*P_cls + w4*P_attention + w5*P_concat
```
where w1 + w2 + w3 + w4 + w5 = 1

### Expected Performance

| Model Type | Expected CV Accuracy |
|-----------|---------------------|
| Mean Pool | ~0.920 - 0.925 |
| Max Pool | ~0.915 - 0.920 |
| CLS Token | ~0.918 - 0.923 |
| Attention Pool | ~0.922 - 0.928 |
| Concat Pool | ~0.925 - 0.930 |
| **Simple Average** | **~0.928 - 0.932** |
| **Majority Voting** | **~0.925 - 0.930** |
| **Optuna Optimized** | **~0.930 - 0.935** |

---

## Training Configuration

### Hyperparameters

| Parameter | Value |
|-----------|-------|
| Model | `microsoft/deberta-v3-large` |
| Max Sequence Length | 512 |
| Batch Size | 4 |
| Gradient Accumulation | 4 |
| Effective Batch Size | 16 |
| Epochs | 3 |
| Learning Rate | 2e-5 |
| LR Scheduler | Cosine |
| Warmup Ratio | 0.06 |
| Weight Decay | 0.01 |
| Label Smoothing | 0.05 |
| Cross-Validation | 5-Fold Stratified |
| Optuna Trials | 200 |

### Training Features

- ✅ Gradient Checkpointing (memory optimization)
- ✅ Mixed Precision Training (FP16/BF16)
- ✅ Class Weighting (balanced)
- ✅ In-Memory Best Model Tracking
- ✅ Label Smoothing (0.05)

---

## File Structure

```
Fake or Real The Impostor Hunt in Texts/
├── README.md                                 # This file
├── fake-or-real-the-impostor-hunt-in-texts.ipynb  # Main notebook
└── hf_run_deberta_ensemble/                  # Output directory
    ├── oof_mean_pool.csv                     # OOF predictions
    ├── oof_max_pool.csv
    ├── oof_cls_token.csv
    ├── oof_attention_pool.csv
    ├── oof_concat_pool.csv
    ├── oof_ensemble_average.csv
    ├── oof_ensemble_voting.csv
    ├── oof_ensemble_optuna.csv
    ├── submission_average.csv
    ├── submission_voting.csv
    └── submission_optuna.csv                 # ← Best submission
```

---

## Usage

### Prerequisites

```bash
pip install transformers datasets evaluate optuna torch scikit-learn pandas numpy
```

### Running on Kaggle

1. Create a new Kaggle notebook
2. Add the competition dataset
3. Copy the code from the notebook
4. Enable GPU accelerator
5. Run all cells

### Submission Format

```csv
id,real_text_id
1501,1
1502,1
1503,2
1504,1
...
```

---

## Timeline

| Event | Date |
|-------|------|
| Competition Start | June 23, 2025 |
| Entry Deadline | September 23, 2025 |
| Team Merger Deadline | September 23, 2025 |
| Final Submission Deadline | September 23, 2025 |
| Private Leaderboard Release | September 30, 2025 |
| ESA AI STAR Conference | December 3-5, 2025 |
| IEEE SaTML Conference | March 23-25, 2026 |

*All deadlines at 12:00 PM CEST*

---

## Prizes

| Place | Prize |
|-------|-------|
|  1st | $500 USD |
|  2nd | $250 USD |
|  3rd | $150 USD |

Winners will be invited as co-authors of a joint paper summarizing the competition.

---

## Organizers

The competition was organized on behalf of **ESA** (Evridiki Ntagiou) by:

- **MI Space Team** from Warsaw University of Technology
  - Przemysław Biecek
  - Artur Janicki
  - Agata Kaczmarek
  - Dawid Płudowski
- **KPLabs**
  - Krzysztof Kotowski
  - Ramez Shendy

---

## References

1. **DataX Strategy**: E. Ntagiou, J. Eggleston, K. Cichecka, and P. Collins, "DataX: a state of the art data strategy for mission operations", IAC 2024. [Link](https://iafastro.directory/iac/paper/id/89097/summary/)

2. **Secure AI for Space**: K. Kotowski et al., "Towards Explainable and Secure AI for Space Mission Operations", SpaceOps-2025. [Link](https://star.spaceops.org/2025/user_manudownload.php?doc=150__traivhwa.pdf)

3. **Resistance Against Manipulative AI**: P. Wilczyński et al., ECAI 2024. [arXiv](https://arxiv.org/abs/2404.14230)

4. **Dark Patterns in LLMs**: W. Mieleszczenko-Kowszewicz et al. [arXiv](https://arxiv.org/abs/2411.06008)

5. **DeBERTa**: P. He et al., "DeBERTa: Decoding-enhanced BERT with Disentangled Attention". [arXiv](https://arxiv.org/abs/2006.03654)

6. **DeBERTa V3**: P. He et al., "DeBERTaV3: Improving DeBERTa using ELECTRA-Style Pre-Training". [arXiv](https://arxiv.org/abs/2111.09543)

7. **Optuna**: T. Akiba et al., "Optuna: A Hyperparameter Optimization Framework". [Link](https://optuna.org/)

---

## Citation

```bibtex
@misc{fake-or-real-2025,
  author = {Kaczmarek, Agata and Płudowski, Dawid and Kotowski, Krzysztof 
            and Shendy, Ramez and Janicki, Artur and Biecek, Przemysław 
            and Ntagiou, Evridiki},
  title = {Fake or Real: The Impostor Hunt in Texts},
  year = {2025},
  publisher = {Kaggle},
  url = {https://kaggle.com/competitions/fake-or-real-the-impostor-hunt}
}
```
