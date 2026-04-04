<h1 align="center">🧬 Stanford RNA 3D Folding Part 2</h1>

<p align="center">
  <b>Solving RNA 3D structure prediction — one of biology's remaining grand challenges</b>
</p>

<p align="center">
  <a href="https://www.kaggle.com/competitions/stanford-rna-3d-folding-2/overview">
    <img src="https://img.shields.io/badge/Kaggle-Competition-blue?style=for-the-badge&logo=kaggle" alt="Kaggle Competition"/>
  </a>
  <a href="https://www.kaggle.com/code/adilshamim8/stanford-rna-3d-folding-part-2">
    <img src="https://img.shields.io/badge/Notebook-View%20on%20Kaggle-20BEFF?style=for-the-badge&logo=kaggle" alt="Kaggle Notebook"/>
  </a>
  <img src="https://img.shields.io/badge/Rank-220%20%2F%201867-gold?style=for-the-badge" alt="Rank Badge"/>
</p>

---

## 🏆 Competition Result

<div align="center">

| 🏅 Metric        | 📊 Value              |
|------------------|-----------------------|
| Final Rank       | **220 / 1867**        |
| Percentile       | **Top 12%**           |
| Competition Type | RNA Structure Prediction |
| Host             | Stanford × Kaggle     |

</div>

---

## 📌 Overview

RNA molecules fold into complex **three-dimensional structures** that determine their biological function. Unlike DNA, RNA's single-stranded nature allows it to fold back on itself, forming intricate shapes critical to cellular processes — including gene regulation, catalysis, and viral replication.

This competition challenges participants to **accurately predict the 3D atomic coordinates** of RNA sequences from scratch — a problem that has stumped computational biologists for decades and sits at the frontier of modern structural biology.

---

## 🎯 Objective

Given a raw RNA nucleotide sequence (composed of A, U, G, C bases), predict the precise **3D spatial coordinates** of each atom in the folded structure, evaluated by structural similarity metrics against experimentally determined ground-truth conformations.

---

## 🔬 Approach

### 1. 🔍 Exploratory Data Analysis
- Analyzed RNA sequence length distributions, nucleotide composition biases, and structural diversity across training samples
- Visualized 3D coordinate distributions and identified common folding motifs

### 2. 🧱 Feature Engineering
- Encoded nucleotide sequences using positional and base-pairing features
- Extracted secondary structure predictions as auxiliary inputs
- Engineered distance-based and angular features to capture spatial geometry

### 3. 🤖 Modeling
- Applied deep learning architectures suited for sequential and structural biological data
- Leveraged transformer-based encoders to capture long-range nucleotide dependencies
- Experimented with graph neural network (GNN) layers to model base-pairing interactions

### 4. ⚙️ Optimization
- Performed cross-validation to ensure model generalization across diverse RNA families
- Tuned hyperparameters including learning rate schedules, dropout, and architecture depth
- Applied post-processing coordinate refinement to improve predicted structure quality

### 5. 📤 Submission
- Generated final 3D coordinate predictions and validated against expected submission format
- Selected best-performing checkpoint based on validation structural similarity scores

---

## 📁 Project Structure

```
Stanford RNA 3D Folding Part 2/
│
├── notebook.ipynb        # Full solution notebook (EDA → Model → Submission)
└── README.md             # Project documentation
```

---

## 📊 Key Results

| Metric              | Value           |
|---------------------|-----------------|
| 🏅 Final Rank       | 220 / 1867      |
| 📈 Top Percentile   | Top ~12%        |
| 🧬 Problem Type     | 3D Structure Prediction |
| 🔧 Primary Tools    | Python, Deep Learning, Kaggle Notebooks |

---

## 🛠️ Tech Stack

![Python](https://img.shields.io/badge/Python-3776AB?style=flat&logo=python&logoColor=white)
![PyTorch](https://img.shields.io/badge/PyTorch-EE4C2C?style=flat&logo=pytorch&logoColor=white)
![NumPy](https://img.shields.io/badge/NumPy-013243?style=flat&logo=numpy&logoColor=white)
![Pandas](https://img.shields.io/badge/Pandas-150458?style=flat&logo=pandas&logoColor=white)
![Kaggle](https://img.shields.io/badge/Kaggle-20BEFF?style=flat&logo=kaggle&logoColor=white)

---

## 🔗 Links

| Resource            | Link |
|---------------------|------|
| 🏁 Competition Page | [Stanford RNA 3D Folding Part 2](https://www.kaggle.com/competitions/stanford-rna-3d-folding-2/overview) |
| 📓 My Notebook      | [View on Kaggle](https://www.kaggle.com/code/adilshamim8/stanford-rna-3d-folding-part-2) |
| 👤 My Kaggle Profile | [adilshamim8](https://www.kaggle.com/adilshamim8) |

---

## 📬 Connect & Contribute

If you found this work helpful or interesting, feel free to ⭐ star the repository, fork it, or open an issue. Feedback and collaboration are always welcome!

---

<p align="center">
  Made with ❤️ by <a href="https://www.kaggle.com/adilshamim8"><b>Adil Shamim</b></a>
</p>
