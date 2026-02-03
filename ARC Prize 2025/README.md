# ARC Prize 2025: Comprehensive Solution with Visualizations

![Competition](https://img.shields.io/badge/Competition-ARC%20Prize%202025-blue)
![Score](https://img.shields.io/badge/Private%20LB%20Score-413-brightgreen)
![Ranking](https://img.shields.io/badge/Ranking-Top%2029%25-success)
![Status](https://img.shields.io/badge/Status-Late%20Submission-orange)

## Table of Contents

- [Overview](#overview)
- [Personal Results](#personal-results)
- [Competition Details](#competition-details)
- [Problem Description](#problem-description)
- [Solution Approach](#solution-approach)
- [Notebook Summary](#notebook-summary)
- [Dataset Information](#dataset-information)
- [Evaluation Metrics](#evaluation-metrics)
- [Submission Requirements](#submission-requirements)
- [Timeline](#timeline)
- [Prize Structure](#prize-structure)
- [Resources & References](#resources--references)
- [Contact & Links](#contact--links)

---

## Overview

The **ARC Prize 2025** is a competition focused on developing AI systems capable of **novel reasoning** and abstract problem-solving. Unlike traditional ML competitions that rely on extensive training data, this competition tests the ability of AI systems to generalize to completely new problems they have never encountered before.

The **Abstraction and Reasoning Corpus for Artificial General Intelligence (ARC-AGI-2)** benchmark measures an AI system's capacity to efficiently learn new skills. While humans have collectively achieved 100% accuracy on ARC tasks, the best AI systems to date only score 4%. This competition encourages researchers to explore novel AI approaches beyond Large Language Models (LLMs), which struggle with tasks outside their training distribution.

---

## Personal Results

| Metric | Value |
|--------|-------|
| **Private Leaderboard Score** | 413 |
| **Ranking** | Top 29% |
| **Submission Type** | Late Submission |
| **Notebook** | [View Public Notebook](https://www.kaggle.com/code/adilshamim8/arc-prize-2025-comprehensive-solution/notebook) |

---

## Competition Details

### About ARC Prize 2025

The competition builds upon the **ARC Prize 2024** with an **updated dataset of human-calibrated problems** and **increased compute resources** for participants. It's designed to advance progress toward **Artificial General Intelligence (AGI)** by encouraging novel reasoning approaches.

**Key Competition Information:**
- **Host:** Kaggle & ARCPrize.org
- **Start Date:** March 26, 2025
- **Submission Deadline:** November 3, 2025
- **Paper Award Deadline:** November 9, 2025
- **Total Prize Pool:** $1,000,000+

### Competition Format

This competition focuses on:
1. **Novel Reasoning:** Solving tasks the AI has never encountered before
2. **Efficient Learning:** Learning from minimal demonstrations (typically 3 input-output pairs)
3. **Exact Predictions:** Making 2 attempted predictions for each test case
4. **Open-Source Solutions:** Winners must open-source their solutions

---

## Problem Description

### The Core Challenge

Participants must develop algorithms capable of solving **abstract reasoning tasks** that require:

- **Pattern Recognition:** Identifying underlying patterns from demonstration pairs
- **Symbolic Reasoning:** Understanding abstract rules and transformations
- **Generalization:** Applying learned patterns to novel test cases
- **Precise Prediction:** Constructing exact output grids matching ground truth

### Task Format

Each task consists of:
- **Train Set:** 2-4 demonstration input-output pairs showing the reasoning pattern
- **Test Set:** 1-2 test inputs requiring predicted outputs
- **Grid Representation:** Rectangular matrices (1×1 to 30×30) with integer values 0-9 (colors)

### Evaluation Model

- **2 Attempts per Task:** Participants make 2 predictions (attempt_1, attempt_2) per test output
- **Binary Scoring:** A task scores 1 if either prediction matches ground truth exactly
- **Final Score:** Percentage of correct task test outputs

---

## Solution Approach

### Methodology

The solution implements a **multi-stage pipeline** combining symbolic and neural approaches:

#### 1. **Data Exploration & Visualization**
- Load and analyze the ARC-AGI-2 dataset
- Create visualization tools for grid-based tasks
- Analyze task complexity metrics (grid sizes, color diversity, transformations)

#### 2. **Domain-Specific Language (DSL)**
Implement core transformation operations:
- **Geometric:** Rotate, flip, crop, pad
- **Color-based:** Replace colors, flood fill
- **Structural:** Extract bounding boxes, identify patterns

```python
# Example DSL operations
- rotate(grid, k): Rotate k×90 degrees
- flip(grid, axis): Flip vertically or horizontally
- color_replace(grid, from, to): Replace colors
- flood_fill(grid, x, y, color): Flood fill from position
- crop(grid, x1, y1, x2, y2): Extract subgrid
- pad(grid, top, bottom, left, right): Add padding
```

#### 3. **Symbolic Program Search**
- Use depth-first search (DFS) with time limits
- Search for sequences of DSL operations matching training examples
- Test candidate programs on all train pairs
- Apply verified programs to test inputs

#### 4. **Neural-Guided Program Synthesis**
- **SketchPredictor Model:** CNN-based network predicting likely DSL operations
- **Input Processing:** One-hot encode input/output grids
- **Feature Extraction:** Learn patterns from grid transformations
- **Operation Ranking:** Prioritize promising DSL operations

#### 5. **Task Classification**
- **TaskClassifier:** Categorize tasks by pattern type
- **Feature Extraction:** Grid sizes, color counts, transformation ratios
- **Strategy Selection:** Adapt search depth based on task complexity

```
Task Categories:
- Pattern Expansion: Output > 4× input size
- Pattern Contraction: Output < 0.5× input size
- Pattern Transformation: Output ≈ input size
- General Approach: Default multi-depth search
```

#### 6. **Fallback Predictions**
- Generate multiple attempts using different strategies
- Combine rotation and color transformation variants
- Ensure both attempt_1 and attempt_2 are provided

---

## Notebook Summary

### Key Sections

The comprehensive notebook implements the complete solution pipeline:

**1. Introduction and Setup**
- Load required libraries (PyTorch, NumPy, Matplotlib, sklearn, etc.)
- Configure visualization environment

**2. Data Loading and Exploration**
- Load training, evaluation, and test datasets
- Analyze task complexity metrics
- Visualize sample tasks and transformations

**3. Visualization Functions**
- `show_grid()`: Render colored grids with proper formatting
- `visualize_task()`: Display complete task with train/test pairs
- `demonstrate_dsl_operations()`: Show all transformation operations

**4. Domain-Specific Language (DSL)**
Implement 9+ core transformation operations with testing

**5. Symbolic Program Search**
- `search_program()`: DFS-based search for operation sequences
- `program_matches()`: Verify candidate programs on training data
- `solve_task_symbolic()`: Complete symbolic solving pipeline

**6. Neural-Guided Program Synthesis**
- `SketchPredictor`: CNN predicting likely DSL operations
- `prepare_grid_features()`: One-hot encode and process grids
- Training framework for neural guidance

**7. Task Classification System**
- `TaskClassifier`: Neural network categorizing task types
- `extract_task_features()`: Extract complexity metrics
- Strategy selection based on task characteristics

**8. Final Solution Pipeline**
- `solve_arc_task()`: Complete end-to-end solver
- Integration of classification, search, and neural components

**9. Submission Generation**
- `generate_submission()`: Create submission.json for all test tasks
- Output format: Dual predictions per test case

---

## Dataset Information

### Files

| File | Size | Description |
|------|------|-------------|
| `arc-agi_training_challenges.json` | Multiple MB | Training tasks with demonstrations |
| `arc-agi_training_solutions.json` | - | Ground truth solutions for training tasks |
| `arc-agi_evaluation_challenges.json` | 984.68 KB | Evaluation/validation tasks |
| `arc-agi_evaluation_solutions.json` | - | Ground truth for evaluation tasks |
| `arc-agi_test_challenges.json` | - | 240 hidden test tasks (for scoring) |
| `sample_submission.json` | - | Submission format example |

### Dataset Statistics

- **Training Tasks:** Full ARC-AGI-2 training set
- **Evaluation Tasks:** Validation set (12+ tasks visible)
- **Test Tasks:** 240 held-out tasks for leaderboard scoring
- **Task Structure:** 2-4 train pairs + 1-2 test inputs per task
- **Grid Range:** 1×1 to 30×30 cells
- **Color Range:** 0-9 (10 colors)

### Data Format

```json
{
  "task_id": {
    "train": [
      {
        "input": [[0, 1, 2], [3, 4, 5]],
        "output": [[0, 0, 0], [0, 0, 0]]
      },
      ...
    ],
    "test": [
      {
        "input": [[1, 2, 3], [4, 5, 6]]
      },
      ...
    ]
  }
}
```

---

## Evaluation Metrics

### Scoring System

**Primary Metric:** Percentage of Correct Predictions

- **Per Test Output:** 1 point if either attempt_1 OR attempt_2 matches ground truth exactly
- **Final Score:** (Total matches) / (Total test outputs) × 100%

### Submission Requirements

**JSON Format:**
```json
{
  "task_id_1": [
    {
      "attempt_1": [[0, 1], [2, 3]],
      "attempt_2": [[0, 0], [0, 0]]
    }
  ],
  "task_id_2": [
    {
      "attempt_1": [[4, 5], [6, 7]],
      "attempt_2": [[8, 9], [1, 2]]
    },
    {
      "attempt_1": [[0, 0]],
      "attempt_2": [[1, 1]]
    }
  ]
}
```

**Requirements:**
- All task_ids from test challenges must be included
- Both "attempt_1" and "attempt_2" must be present
- Each prediction is a list of lists (grid format)
- Exact cell-by-cell matching required for correct score

---

## Submission Requirements

### Code Competition Rules

- **Runtime:** CPU or GPU notebooks ≤ 12 hours
- **Internet:** No internet access enabled
- **External Data:** Publicly available datasets and pre-trained models allowed
- **Submission:** Must be named `submission.json`
- **Notebook:** Submissions via Kaggle Notebooks only

### Notebook Configuration

- No external internet access
- CPU/GPU runtime limited to 12 hours
- Rerun submissions scored on hidden 240-task test set
- Platform: Kaggle Notebooks required

### GPU Options

- **Upgraded L4x4 Machines:** 96GB GPU memory available
  - Double GPU quota usage rate
  - Only for this competition
  - Internet disabled required

---

## Timeline

| Event | Date |
|-------|------|
| Competition Start | March 26, 2025 |
| Entry Deadline | October 27, 2025 |
| Team Merger Deadline | October 27, 2025 |
| Final Submission Deadline | November 3, 2025 |
| Paper Award Deadline | November 9, 2025 |

**Note:** All deadlines are 11:59 PM UTC

---

## Prize Structure

### Total Prize Pool: $1,000,000+

#### 2025 Progress Prizes: $125,000

**Top-Ranking Teams:**
| Place | Prize |
|-------|-------|
| 1st | $25,000 |
| 2nd | $10,000 |
| 3rd | $5,000 |
| 4th | $5,000 |
| 5th | $5,000 |

**Paper Award Prizes: $75,000**
| Place | Prize |
|-------|-------|
| Winner | $50,000 |
| Runner-up 1 | $20,000 |
| Runner-up 2 | $5,000 |

#### Grand Prize: $700,000

**Unlocked at 85% Accuracy**

| Place | Prize |
|-------|-------|
| 1st | $350,000 |
| 2nd | $150,000 |
| 3rd | $70,000 |
| 4th | $70,000 |
| 5th | $60,000 |

*Divided among qualifying teams if fewer than 5 reach 85% accuracy*

#### To Be Announced: $175,000

Additional prizes announced on ARCprize.org

### Paper Award Evaluation Criteria

Papers are scored 0-5 in each category:

| Category | Description |
|----------|-------------|
| **Accuracy** | Leaderboard performance of the submission |
| **Universality** | Generalizability beyond this competition |
| **Progress** | Contribution toward 85% accuracy milestone |
| **Theory** | Explanation of why the approach works |
| **Completeness** | Thoroughness of documentation |
| **Novelty** | Innovation relative to existing research |

**Paper Requirements:**
- Submit within 6 days of competition end
- Must be public/open-sourced
- Reference your submission ID
- Format: Kaggle Notebook, PDF, arXiv, or text

---

## Resources & References

### Competition Resources

- **Main Competition:** [ARC Prize 2025 on Kaggle](https://www.kaggle.com/competitions/arc-prize-2025)
- **Dataset:** [ARC Prize 2025 Data](https://www.kaggle.com/competitions/arc-prize-2025/data)
- **Interactive App:** [ARCPrize.org](https://www.arcprize.org)
- **Paper Award Info:** [ARC Prize Paper Awards](https://www.kaggle.com/competitions/arc-prize-2025/discussion)

### Key Papers & Research

- **ARC-AGI Benchmark:** Chollet, F. (2019). "The Measure of Intelligence"
- **Previous ARC Competition:** [ARC Prize 2024](https://www.kaggle.com/competitions/arc-prize-2024)

### Learning Resources

- **Code Golf Practice:** [code.golf](https://code.golf)
- **PyTorch Documentation:** [pytorch.org](https://pytorch.org)
- **NumPy Visualizations:** [matplotlib.org](https://matplotlib.org)
- **scikit-learn:** [scikit-learn.org](https://scikit-learn.org)

---

## Your Public Notebook

**Comprehensive Solution Notebook:**  
[View on Kaggle](https://www.kaggle.com/code/adilshamim8/arc-prize-2025-comprehensive-solution/notebook)

This notebook implements:
- Complete data exploration and visualization
- Domain-specific language (DSL) for grid transformations
- Symbolic program search with time limits
- Neural-guided program synthesis
- Task classification system
- End-to-end solution pipeline
- Submission generation

---

## Contact & Links

Feel free to connect and follow my work:

[![GitHub](https://img.shields.io/badge/GitHub-181717?style=for-the-badge&logo=github&logoColor=white)](https://github.com/AdilShamim8)  
[![LinkedIn](https://img.shields.io/badge/LinkedIn-0077B5?style=for-the-badge&logo=linkedin&logoColor=white)](https://www.linkedin.com/in/adilshamim8)  
[![Twitter](https://img.shields.io/badge/Twitter-1DA1F2?style=for-the-badge&logo=twitter&logoColor=white)](https://x.com/adil_shamim8)  
[![Kaggle](https://img.shields.io/badge/Kaggle-20BEFF?style=for-the-badge&logo=kaggle&logoColor=white)](https://www.kaggle.com/adilshamim8)

---

## Summary

The ARC Prize 2025 represents one of the most challenging competitions in AI, pushing the boundaries of what modern AI systems can accomplish. By combining symbolic reasoning, neural guidance, and task classification, this solution demonstrates how hybrid approaches can tackle novel reasoning problems. The goal of achieving 85% accuracy and contributing to AGI development remains ambitious yet inspiring.

**Key Takeaways:**
- ✅ Achieved top 29% ranking with score of 413
- ✅ Implemented comprehensive DSL for grid transformations
- ✅ Combined symbolic search with neural guidance
- ✅ Developed adaptive task classification system
- ✅ Created complete end-to-end solution pipeline

The competition continues to drive innovation in AI reasoning and problem-solving capabilities.
