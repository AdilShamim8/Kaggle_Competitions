# 🐭 MABe Challenge - Social Action Recognition in Mice

<div align="center">

![Kaggle Competition](https://img.shields.io/badge/Kaggle-Competition-blue?logo=kaggle)
![Prize Pool](https://img.shields.io/badge/Prize%20Pool-$50,000-green)
![Private LB](https://img.shields.io/badge/Private%20LB%20Score-145-brightgreen)
![Top 11%](https://img.shields.io/badge/Ranking-Top%2011%25-gold)
![Python](https://img.shields.io/badge/Python-3.x-blue?logo=python)
![LightGBM](https://img.shields.io/badge/LightGBM-Ensemble-orange)
![XGBoost](https://img.shields.io/badge/XGBoost-Ensemble-red)
![CatBoost](https://img.shields.io/badge/CatBoost-Ensemble-yellow)

**Detect unique behaviors from pose estimates of mice**

[🏆 Competition Link](https://www.kaggle.com/competitions/MABe-mouse-behavior-detection) | [📊 Dataset](https://www.kaggle.com/competitions/MABe-mouse-behavior-detection/data) | [📓 My Kaggle Notebook](https://www.kaggle.com/code/adilshamim8/mabe-challenge/notebook)

</div>

---

## Table of Contents

- [Competition Overview](#-competition-overview)
- [My Results](#-my-results)
- [Solution Approach](#-solution-approach)
- [Dataset Description](#-dataset-description)
- [Feature Engineering](#-feature-engineering)
- [Model Architecture](#-model-architecture)
- [Prediction Pipeline](#-prediction-pipeline)
- [File Structure](#-file-structure)
- [Timeline](#-timeline)
- [References](#-references)

---

## Competition Overview

### Background

Animal social behavior is complex. Species from ants to wolves to mice form social groups where they build nests, raise their young, care for their groupmates, and defend their territory. Studying these behaviors teaches us about the brain and the evolution of behavior, but the work has usually required subjective, time-consuming documentation of animals' actions. ML advancements now let us automate this process, supporting large-scale behavioral studies in the wild and in the lab.

### Challenge

This competition challenges you to build models to identify **over 30 different social and non-social behaviors** in pairs and groups of co-housed mice, based on markerless motion capture of their movements in top-down video recordings. The dataset includes:

- **400+ hours** of footage
- **20+ behavioral recording systems**
- **Frame-by-frame expert annotations**

### Impact

Your work will help scientists automate behavior analysis and better understand animal social structures. These models may be deployed across numerous labs in:
- 🧠 Neuroscience
- 🧬 Computational Biology
- 🐾 Ethology
- 🌍 Ecology

---

## My Results

| Metric | Score |
|--------|-------|
| **Private Leaderboard Score** | **145** |
| **Ranking** | **Top 11%** |
| **Evaluation Metric** | F-Score (averaged across labs/videos) |

---

## Solution Approach

### Core Strategy

My solution tackles the MABe challenge by building **separate GBDT ensemble models for each unique `body_parts_tracked` configuration**:

1. **Grouped by Tracker**: Loop through each unique `body_parts_tracked` string
2. **FPS-Aware Features**: Generate advanced temporal and spatial features with window sizes scaled by video's FPS for time-invariant features
3. **Stratified Subsampling**: Use `StratifiedSubsetClassifier` wrapper to train GBDTs on stratified subsamples to manage memory and time
4. **Ensemble Model**: For each behavior, train an ensemble of LightGBM, XGBoost, and CatBoost models
5. **Adaptive Prediction**: Use temporal smoothing, adaptive per-action probability thresholds, and minimum duration filtering to generate final event segments

### Key Innovations

-  **Wavelet Transform Features** - Multi-scale frequency analysis
-  **Physics-Informed Features** - Jerk, angular velocity, kinetic energy
-  **Transition Detection Features** - Behavior onset/offset detection
-  **Enhanced Pair Interaction Features** - Bearing, time-to-collision
-  **Relative Physics Features** - Relative velocity, acceleration

---

## Dataset Description

### Overview

The dataset contains pose tracking data and behavior annotations for mice from multiple labs using different recording setups.

### Files Structure

```
data/
├── train.csv                    # Metadata about mice and recording setups
├── test.csv                     # Test metadata
├── train_tracking/              # Pose tracking data (parquet files)
│   └── {lab_id}/{video_id}.parquet
├── train_annotation/            # Behavior annotations (parquet files)
│   └── {lab_id}/{video_id}.parquet
├── test_tracking/               # Test pose tracking data
└── sample_submission.csv        # Submission format
```

### Dataset Statistics

| Metric | Value |
|--------|-------|
| Total Files | 9,640 |
| Size | 2.84 GB |
| Format | Parquet, CSV |
| License | CC BY 4.0 |
| Test Videos | ~200 (hidden) |

### Metadata Columns

| Column | Description |
|--------|-------------|
| `lab_id` | Lab pseudonym (includes CalMS21, CRIM13, MABe22) |
| `video_id` | Unique video identifier |
| `mouse[1-4]_*` | Mouse info (strain, color, sex, age, condition) |
| `frames_per_second` | Video FPS |
| `pix_per_cm_approx` | Pixels per centimeter |
| `arena_width/height` | Arena dimensions (cm) |
| `body_parts_tracked` | List of tracked body parts |
| `behaviors_labeled` | Annotated behaviors for the video |

### Tracking Data Format

| Column | Description |
|--------|-------------|
| `video_frame` | Frame number |
| `mouse_id` | Mouse identifier |
| `bodypart` | Tracked body part name |
| `x`, `y` | Position in pixels |

---

## Feature Engineering

### Single Mouse Features

#### Base Distance Features
- Pairwise distances between all tracked body parts
- Speed-like features (ear-left, ear-right, tail-base movements)
- Body elongation ratio

#### Temporal Features
```python
# Rolling statistics at multiple windows (scaled by FPS)
for w in [5, 15, 30, 60]:
    - center_x/y mean, std
    - x/y range
    - displacement
    - activity (movement variance)
```

#### Advanced Features

| Feature Category | Description |
|-----------------|-------------|
| **Curvature** | Trajectory curvature, turn rate |
| **Multi-scale** | Speed statistics at 10, 40, 160 frame windows |
| **State Transitions** | Speed state changes (stationary/slow/medium/fast) |
| **Long-range** | Extended window statistics (120, 240 frames) |
| **Cumulative Distance** | Path length over 180-frame horizon |
| **Grooming Micro-features** | Head-body decoupling, nose radius std |

#### NEW: Physics-Informed Features

```python
# Jerk (rate of acceleration change)
jerk = d³position/dt³  # Key for detecting behavior transitions

# Angular velocity (rotation detection)
angular_vel = dθ/dt

# Kinetic energy proxy
KE = speed²

# Centripetal acceleration (circling behavior)
centripetal = speed × angular_velocity
```

#### NEW: Wavelet Features

```python
# Multi-scale frequency decomposition
coeffs = pywt.wavedec(speed, 'db4', level=4)

# Features extracted:
- wavelet_energy_L{0-4}     # Energy at each decomposition level
- wavelet_low_freq_mean     # Slow movement patterns
- wavelet_high_freq_mean    # Fast movement patterns
- wavelet_freq_ratio        # Behavior type indicator
```

#### NEW: Transition Detection

```python
# Z-score of speed vs local context
speed_zscore = (speed - local_mean) / local_std

# Change point detection
speed_change = future_mean - past_mean

# Autocorrelation (rhythmic behaviors like grooming)
speed_autocorr_lag_N
```

### Pair Interaction Features

#### Inter-Mouse Distances
- All pairwise body part distances between mice
- Relative orientation (heading alignment)
- Approach rate

#### Distance Bins
```python
- very_close: distance < 5 cm
- close: 5-15 cm
- medium: 15-30 cm
- far: > 30 cm
```

#### Temporal Interaction Features
- Distance statistics at multiple windows
- Velocity alignment
- Movement coordination (correlation)
- Chase detection (approach × following direction)

#### NEW: Relative Physics Features

```python
# Relative velocity between mice
rel_speed = |v_A - v_B|

# Relative acceleration
rel_acc = |a_A - a_B|

# Time to collision
ttc = distance / closing_speed

# Bearing angle (A's view of B)
bearing = angle_to_B - A_heading
facing_B = |bearing| < π/4
```

---

## Model Architecture

### Ensemble Components

| Model | Configuration | Purpose |
|-------|--------------|---------|
| **LightGBM (225 est)** | lr=0.07, leaves=31, subsample=0.8 | Main predictor |
| **LightGBM (150 est)** | lr=0.1, leaves=63, depth=8, reg=0.1 | Regularized variant |
| **LightGBM (100 est)** | lr=0.05, leaves=127, depth=10 | Deep variant |
| **XGBoost (180 est)** | lr=0.08, depth=6, gamma=1.0 | Gradient boosting |
| **CatBoost (120 est)** | lr=0.1, depth=6 | Categorical handling |

#### GPU-Enhanced Models (when available)

| Model | Configuration |
|-------|--------------|
| **XGBoost (2000 est)** | lr=0.05, max_leaves=255, early stopping |
| **XGBoost (1400 est)** | lr=0.06, depth=7, early stopping |
| **CatBoost (4000 est)** | lr=0.03, depth=8, Bayesian bootstrap |

### Training Strategy

```
┌─────────────────────────────────────────┐
│   For each body_parts_tracked config   │
├─────────────────────────────────────────┤
│                                         │
│   ┌───────────────────────────────┐    │
│   │   Single Mouse Pipeline       │    │
│   │   - transform_single()        │    │
│   │   - Train ensemble per action │    │
│   └───────────────────────────────┘    │
│                                         │
│   ┌───────────────────────────────┐    │
│   │   Pair Mouse Pipeline         │    │
│   │   - transform_pair()          │    │
│   │   - Train ensemble per action │    │
│   └───────────────────────────────┘    │
│                                         │
└─────────────────────────────────────────┘
```

### Stratified Subset Classifier

Handles memory constraints by training on stratified subsamples:

```python
class StratifiedSubsetClassifierWEval:
    - n_samples: Training sample limit
    - valid_size: Validation split (10%)
    - Early stopping with AUCPR metric
    - Auto class weight balancing
```

---

##  Prediction Pipeline

### Adaptive Thresholding

```python
def predict_multiclass_adaptive():
    1. Apply temporal smoothing (window=5)
    2. Get argmax predictions
    3. Apply per-action thresholds (default=0.27)
    4. Filter short events (< 3 frames)
    5. Generate action segments
```

### Post-Processing: Robustify

```python
def robustify(submission):
    1. Remove overlapping predictions
    2. Fill missing video predictions
    3. Ensure valid start/stop frames
```

---

## File Structure

```
MABe Challenge - Social Action Recognition in Mice/
├── README.md                     # This file
├── mabe-challenge.ipynb          # Main solution notebook
└── submission.csv                # Generated predictions
```

---

## Timeline

| Event | Date |
|-------|------|
| Competition Start | September 18, 2025 |
| Entry Deadline | December 8, 2025 |
| Team Merger Deadline | December 8, 2025 |
| Final Submission Deadline | December 15, 2025 |

*All deadlines at 11:59 PM UTC*

---

## Prizes

| Place | Prize |
|-------|-------|
|  1st | $20,000 |
|  2nd | $10,000 |
|  3rd | $8,000 |
| 4th | $7,000 |
| 5th | $5,000 |

**Total Prize Pool: $50,000**

---

## Previous Efforts & Benchmarks

This competition builds on previous Multi-Agent Behavior (MABe) Workshops:

| Competition | Venue | Paper |
|-------------|-------|-------|
| CalMS21 | NeurIPS 2021 | [arXiv:2104.02710](https://arxiv.org/pdf/2104.02710.pdf) |
| MABe22 | ICML 2023 | [arXiv:2207.10553](https://arxiv.org/pdf/2207.10553.pdf) |
| CRIM13 | CVPR 2012 | doi: 10.1109/CVPR.2012.6247817 |

All pose and annotation files from CalMS21, MABe22, and CRIM13 are provided as additional training data.

---

## Code Requirements

- CPU Notebook: ≤ 9 hours
- GPU Notebook: ≤ 9 hours
- Internet: Disabled
- External Data: Publicly available allowed (pre-trained models included)
- Output: `submission.csv`

---

## Dependencies

```python
# Core
pandas, numpy, polars

# ML Models
lightgbm, xgboost, catboost
scikit-learn

# Signal Processing
scipy
pywt (PyWavelets)  # Optional: wavelet features

# Optional
sklearn-crfsuite  # CRF post-processing
fastdtw           # Template matching
```

---

## Citation

```bibtex
@misc{mabe-challenge-2025,
  author = {Sun, Jennifer J. and Marks, Markus and Golden, Sam and Pereira, Talmo 
            and Kennedy, Ann and Dane, Sohier and Howard, Addison and Chow, Ashley},
  title = {MABe Challenge - Social Action Recognition in Mice},
  year = {2025},
  publisher = {Kaggle},
  url = {https://kaggle.com/competitions/MABe-mouse-behavior-detection}
}
```

---

<div align="center">

**🐭 Advancing Behavioral Science Through Machine Learning 🧠**

</div>
