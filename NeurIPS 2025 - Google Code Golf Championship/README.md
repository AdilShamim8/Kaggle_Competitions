<div align="center">

# NeurIPS 2025 - Google Code Golf Championship

![Kaggle Competition](https://img.shields.io/badge/Kaggle-Competition-blue?logo=kaggle)
![NeurIPS](https://img.shields.io/badge/NeurIPS-2025-purple)
![Private LB](https://img.shields.io/badge/Private%20LB%20Score-264-brightgreen)
![Top 24%](https://img.shields.io/badge/Ranking-Top%2024%25-gold)
![Python](https://img.shields.io/badge/Python-3.x-blue?logo=python)

**Implement a variety of programs using the fewest number of characters!**

[Competition Link](https://www.kaggle.com/competitions/google-code-golf-2025) | [Dataset](https://www.kaggle.com/competitions/google-code-golf-2025/data) | [My Kaggle Notebook](https://www.kaggle.com/code/adilshamim8/google-code-golf-championship-101/notebook)

</div>

---

## Table of Contents

- [Competition Overview](#-competition-overview)
- [My Results](#-my-results)
- [Notebook Summary](#-notebook-summary)
- [Problem Setup](#-problem-setup)
- [Constraints](#-constraints)
- [Evaluation](#-evaluation)
- [Dataset Description](#-dataset-description)
- [Submission Format](#-submission-format)
- [Timeline](#-timeline)
- [Prizes](#-prizes)
- [NeurIPS 2025 Track](#-neurips-2025-track)
- [Resources](#-resources)
- [Citation](#-citation)

---

## Competition Overview

This competition challenges you to solve **400 ARC-AGI tasks** using Python 3 programs that are **both correct and as short as possible**. Each task is represented by pairs of input/output grids, and your code must generalize across training, test, and ARC-GEN examples.

The concise implementations produced by top teams are expected to serve as canonical reference solutions for the ARC-AGI benchmark suite.

ARC-AGI example task:
https://arcprize.org/play?task=543a7ed5

---

## My Results

| Metric | Value |
|--------|-------|
| **Private Leaderboard Score** | **264** |
| **Ranking** | **Top 24%** |

---

## Notebook Summary

This repository includes a full, step-by-step notebook that demonstrates:

- Environment setup and dataset loading
- Deep pattern analysis for a target task (Task 253)
- Hypothesis-driven solution development
- Code golf optimization techniques
- Verification and submission packaging
- A comprehensive optimizer for mass task compression

Notebook: [google-code-golf-championship-101.ipynb](google-code-golf-championship-101.ipynb)

### Key Notebook Sections

1. **Environment Setup and Data Loading**
2. **Deep Pattern Analysis**
3. **Pattern Identification and Solution Development**
4. **Implementation of Working Solution**
5. **Code Golf Optimization**
6. **Verification and Submission**
7. **Optimization Techniques and Strategies**
8. **Competition Strategy for All 400 Tasks**
9. **Advanced Optimizer (Batch Compression + Verification)**

---

## Problem Setup

You are given 400 tasks. Each task includes a set of training, test, and ARC-GEN input/output grid pairs. Your program must:

1. Determine the output grid size.
2. Fill each cell with a number from 0 to 9.
3. Match all expected outputs exactly.
4. Use as few characters (bytes) as possible.

---

## Constraints

- Each solution file must be **self-contained**.
- **Only Python Standard Library imports** are allowed.
- Submission files **must not import from other task files**.
- Security constraints are enforced for prize eligibility.

---

## Evaluation

For each task, your score is:

$$
	ext{score} = \max(1, 2500 - \text{length})
$$

- **Correct program**: $\max(1, 2500 - \text{bytes})$
- **Incorrect program**: $0.001$ points

Total score is the sum across all 400 tasks.

---

## Dataset Description

Each task file is a JSON file containing:

- `train`: ARC-AGI-1 training pairs
- `test`: ARC-AGI-1 test pairs
- `arc-gen`: ARC-GEN-100K pairs

Each pair includes:

- `input`: grid (list of lists)
- `output`: grid (list of lists)

Grids range from **1×1 to 30×30**, with values from **0 to 9**.

**Dataset Stats**

| Item | Value |
|------|-------|
| Files | 401 |
| Size | 97.14 MB |
| Type | json, py |
| License | Apache 2.0 |

---

## Submission Format

Submit a single `submission.zip` containing at most one Python file per task:

```
task001.py
task002.py
...
task400.py
```

Example minimal program (hypothetical):

```
def p(g):
 for r, row in enumerate(g):
	for c, color in enumerate(row):
	 if r and c and color==5 and g[r-1][c-1] not in [0,5]: g[r][c]=0
 return g
```

---

## Timeline

| Event | Date |
|------|------|
| Start Date | July 31, 2025 |
| Entry Deadline | October 23, 2025 |
| Team Merger Deadline | October 23, 2025 |
| Final Submission Deadline | October 30, 2025 |

All deadlines are at **11:59 PM UTC**.

---

## Prizes

**Total Prize Pool: $100,000**

| Place | Prize |
|------|-------|
| 1st | $30,000 |
| 2nd | $20,000 |
| 3rd | $10,000 |
| 4th | $5,000 |
| 5th | $5,000 |
| 6th | $5,000 |
| 7th | $5,000 |
| 8th | $5,000 |
| 9th | $5,000 |
| 10th | $5,000 |
| Longest Leader | $5,000 |

---

## NeurIPS 2025 Track

This contest is part of the **NeurIPS 2025 Competition Track**. Top submissions will be invited to give talks in a special session during the conference in **San Diego, California**.

Members of the top three winning teams will also be invited to collaborate with the organizers on a contest retrospective submitted to **PMLR**.

---

## Resources

- ARC Prize Foundation: https://arcprize.org
- ARC Task Playground: https://arcprize.org/play
- Code Golf Practice: https://code.golf

---

## Citation

```bibtex
@misc{google-code-golf-2025,
	author = {Moffitt, Michael D. and Thakkar, Divy and Burnell, Ryan and Firat, Orhan and Reade, Walter and Dane, Sohier and Howard, Addison},
	title = {NeurIPS 2025 - Google Code Golf Championship},
	year = {2025},
	publisher = {Kaggle},
	url = {https://kaggle.com/competitions/google-code-golf-2025}
}
```

---

<div align="center">

**Shorter code, stronger generalization.**

</div>



