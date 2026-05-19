# Adversarially Robust Intrusion Detection System

**Course:** ICCN-INE2 | INPT  
**Supervisor:** Pr. Tarik Fissaa  
**Author:** Zakariya El Mansouri

---

## Overview

This project builds a deep learning-based Intrusion Detection System (IDS) on the **UNSW-NB15** dataset, then systematically attacks it with adversarial examples and evaluates multiple defenses.

The core question: *can an attacker subtly manipulate network traffic features to evade detection — and can we defend against it?*

---

## Dataset

**UNSW-NB15** — labeled network traffic (normal vs. attack), binary classification.

| Split | Samples | Attack rate |
|-------|---------|-------------|
| Train | 82,332  | 46.5%       |
| Test  | 175,341 | 45.2%       |

- 42 features after preprocessing (drop `id`, `attack_cat`)
- **14 manipulable features** (attacker-controllable): `dur`, `spkts`, `dpkts`, `sbytes`, `dbytes`, `sttl`, `sload`, `dload`, `sinpkt`, `dinpkt`, `sjit`, `djit`, `smean`, `dmean`

> Download `UNSW_NB15_training-set.csv` and `UNSW_NB15_testing-set.csv` into `data/`

---

## Model Architecture

```
Input (42) → Linear(128) → ReLU → Dropout(0.3)
           → Linear(64)  → ReLU → Dropout(0.3)
           → Linear(32)  → ReLU
           → Linear(1)   → Sigmoid → binary prediction
```

**Training:** Adam (lr=0.001), BCELoss, 30 epochs  
**Baseline:** F1 = 0.9617 on clean data

---

## Attacks

Attacks are applied only to the 14 manipulable features (mask = 1 on those features, 0 elsewhere), with ε = 0.1 (L∞ constraint).

| Attack | Description | Evasion Rate (on baseline) |
|--------|-------------|---------------------------|
| **FGSM** | One-step gradient sign attack: `x_adv = x + ε·sign(∇_x L)·mask` | ~20% |
| **PGD-40** | 40-step iterative FGSM with projection: `α = ε/steps` | 26.9% |

---

## Defenses

### Defense 1 — Adversarial Training (3 variants, all failed)

| Variant | Evasion under PGD-40 | Notes |
|---------|----------------------|-------|
| FGSM-AT | 36.7% | Worse than baseline |
| PGD-AT (100% adversarial) | 40.2% | Catastrophic forgetting |
| PGD-AT (mixed 50/50) | 41.5% | Gradient masking |

**Why it fails:** Sigmoid saturation causes near-zero gradients → false robustness illusion (gradient masking). Robustness-accuracy trade-off collapses with tabular data.

### Defense 2 — Feature Squeezing (4 bits)

Quantize each feature to 4 bits to erase small adversarial perturbations:

```
x_sq = round(x / x_max × 16) / 16 × x_max
```

| | F1 | Evasion PGD-40 |
|---|---|---|
| Clean data | 0.9004 | — |
| Under PGD-40 | 0.8449 | **24.5%** |

### Defense 3 — TRADES + PGD-7

TRADES replaces standard adversarial training with a KL-divergence regularization:

```
L = BCE(f(x), y) + β · KL( f(x) || f(x_adv) )
```

With β = 6.0 and PGD-7 inner adversary:

| | F1 | Evasion PGD-40 |
|---|---|---|
| Clean data | 0.9192 | — |
| Under PGD-40 | 0.8941 | **17.4%** |

### Defense 4 — TRADES + Feature Squeezing Pipeline *(best result)*

Orthogonal combination: FS acts on the **input**, TRADES acts on the **model**.

```
x  →  [Feature Squeezing 4-bit]  →  x_sq  →  [TRADES model]  →  ŷ
```

| | F1 | Evasion PGD-40 |
|---|---|---|
| Clean data | 0.9089 | 10.3% |
| Under PGD-40 | **0.9077** | **10.7%** |

The gap between clean and adversarial evasion is only 0.4 pt — the pipeline is stable.

---

## Full Comparison

| Defense | F1 (clean) | Evasion PGD-40 | Notes |
|---------|-----------|----------------|-------|
| Baseline MLP | 0.9617 | 26.9% | No defense |
| AT — FGSM | — | 36.7% | ❌ Worse |
| AT — PGD 100% | — | 40.2% | ❌ Catastrophic forgetting |
| AT — PGD Mix | — | 41.5% | ❌ Gradient masking |
| Feature Squeezing | 0.9004 | 24.5% | ⚠️ Moderate |
| TRADES + PGD-7 | 0.9192 | 17.4% | ✅ Good |
| **TRADES + FS pipeline** | **0.9089** | **10.7%** | ✅✅ Best |

---

## Perspectives

Without retraining, additional strategies can push evasion even lower:

| Strategy | Evasion PGD-40 | F1 (clean) |
|----------|----------------|-----------|
| TRADES — threshold 0.40 | 14.7% | 0.9245 |
| TRADES + FS — threshold 0.45 | **7.8%** | 0.9133 |

See `run_boost.py` for implementation.

---

## Results

All generated figures and CSVs are in `results/`:

| File | Description |
|------|-------------|
| `training_loss.png` | Baseline training curve |
| `confusion_baseline.png` | Confusion matrix (clean vs. PGD) |
| `attack_comparison.png` | F1 drop under FGSM / PGD-40 |
| `trades_comparison.png` | Defense comparison bar chart |
| `final_results.csv` | All defense metrics |
| `trades_results.csv` | TRADES + FS detailed metrics |

---

## Project Structure

```
adversarial-IDS/
├── data/                          ← CSVs (not versioned, see .gitignore)
│   └── .gitkeep
├── notebooks/
│   ├── 1_exploration.ipynb        ← dataset exploration
│   ├── 2_baseline.ipynb           ← MLP baseline training
│   ├── 3_attacks.ipynb            ← FGSM + PGD-40 attacks
│   ├── 4_defense.ipynb            ← Feature Squeezing + AT variants
│   ├── 5_results.ipynb            ← comparison and plots
│   ├── 6_pgd_adversarial_training.ipynb  ← PGD-AT deep dive
│   └── 7_pgd_at_corrected.ipynb   ← corrected PGD-AT + TRADES
├── results/
│   ├── *.png                      ← all plots
│   ├── *.csv                      ← metrics (models not versioned)
├── run_trades.py                  ← TRADES training script
├── run_boost.py                   ← threshold + FS pipeline evaluation
├── CONCEPTS.md                    ← technical concept explainer
├── requirements.txt
└── README.md
```

---

## Setup

```bash
pip install -r requirements.txt
```

Run notebooks in order: `1_exploration` → `2_baseline` → `3_attacks` → `4_defense` → `5_results`

Or run the standalone scripts:

```bash
python run_trades.py    # trains TRADES model
python run_boost.py     # evaluates threshold + pipeline strategies
```

---

## Tech Stack

- Python 3.10+, PyTorch, scikit-learn
- pandas, numpy, matplotlib, seaborn
- Dataset: [UNSW-NB15](https://research.unsw.edu.au/projects/unsw-nb15-dataset)
