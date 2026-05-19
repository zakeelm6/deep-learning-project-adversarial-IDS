# Adversarially Robust Intrusion Detection System

A deep learning-based IDS trained on **UNSW-NB15** network traffic data, tested against adversarial attacks (FGSM, PGD-40), with multiple defenses evaluated.

---

## Dataset

**UNSW-NB15** — labeled network traffic (normal vs. attack), binary classification.

| Split | Samples |
|-------|---------|
| Train | 82,332  |
| Test  | 175,341 |

- 42 features after preprocessing
- **14 manipulable features** (attacker-controllable): `dur`, `spkts`, `dpkts`, `sbytes`, `dbytes`, `sttl`, `sload`, `dload`, `sinpkt`, `dinpkt`, `sjit`, `djit`, `smean`, `dmean`

> Download `UNSW_NB15_training-set.csv` and `UNSW_NB15_testing-set.csv` into `data/`  
> Source: [UNSW-NB15 Dataset](https://research.unsw.edu.au/projects/unsw-nb15-dataset)

---

## Model Architecture

```
Input (42) → Linear(128) → ReLU → Dropout(0.3)
           → Linear(64)  → ReLU → Dropout(0.3)
           → Linear(32)  → ReLU
           → Linear(1)   → Sigmoid
```

**Training:** Adam (lr=0.001), BCELoss, 30 epochs — Baseline F1 = 0.9617

---

## Attacks

Perturbations applied only to the 14 manipulable features, with ε = 0.1 (L∞):

| Attack | Strategy | Evasion Rate (baseline) |
|--------|----------|------------------------|
| FGSM | One-step gradient sign: `x_adv = x + ε·sign(∇_x L)·mask` | ~20% |
| PGD-40 | 40-step iterative FGSM with L∞ projection | 26.9% |

---

## Defenses

### Defense 1 — Adversarial Training (3 variants, all failed)

| Variant | Evasion PGD-40 |
|---------|----------------|
| FGSM-AT | 36.7% ❌ |
| PGD-AT (100% adversarial) | 40.2% ❌ |
| PGD-AT (mixed 50/50) | 41.5% ❌ |

Sigmoid saturation → gradient masking → false robustness. Robustness-accuracy trade-off collapses on tabular data.

### Defense 2 — Feature Squeezing (4 bits)

```
x_sq = round(x / x_max × 16) / 16 × x_max
```

F1 (clean) = 0.9004 | Evasion PGD-40 = **24.5%**

### Defense 3 — TRADES + PGD-7

```
L = BCE(f(x), y) + β · KL( f(x) || f(x_adv) )    β = 6.0
```

F1 (clean) = 0.9192 | Evasion PGD-40 = **17.4%**

### Defense 4 — TRADES + Feature Squeezing Pipeline *(best)*

Orthogonal combination: FS acts on the input, TRADES acts on the model.

```
x  →  [Feature Squeezing 4-bit]  →  x_sq  →  [TRADES model]  →  ŷ
```

F1 (clean) = 0.9089 | Evasion PGD-40 = **10.7%**  
Gap clean vs. adversarial = 0.4 pt (stable pipeline)

---

## Full Results

| Defense | F1 (clean) | Evasion PGD-40 |
|---------|-----------|----------------|
| Baseline MLP | 0.9617 | 26.9% |
| AT — FGSM | — | 36.7% ❌ |
| AT — PGD 100% | — | 40.2% ❌ |
| AT — PGD Mix | — | 41.5% ❌ |
| Feature Squeezing | 0.9004 | 24.5% |
| TRADES + PGD-7 | 0.9192 | 17.4% |
| **TRADES + FS pipeline** | **0.9089** | **10.7%** ✅ |

---

## Perspectives

| Strategy | Evasion PGD-40 | F1 (clean) |
|----------|----------------|-----------|
| TRADES — threshold 0.40 | 14.7% | 0.9245 |
| TRADES + FS — threshold 0.45 | 7.8% | 0.9133 |

See `run_boost.py` for implementation.

---

## Project Structure

```
adversarial-IDS/
├── data/                          ← CSVs (not versioned)
│   └── .gitkeep
├── notebooks/
│   ├── 1_exploration.ipynb        ← dataset exploration
│   ├── 2_baseline.ipynb           ← MLP baseline
│   ├── 3_attacks.ipynb            ← FGSM + PGD-40
│   ├── 4_defense.ipynb            ← Feature Squeezing + AT
│   ├── 5_results.ipynb            ← comparison and plots
│   ├── 6_pgd_adversarial_training.ipynb
│   └── 7_pgd_at_corrected.ipynb   ← TRADES
├── results/
│   ├── *.png                      ← plots
│   └── *.csv                      ← metrics
├── run_trades.py                  ← TRADES training
├── run_boost.py                   ← threshold + pipeline evaluation
├── requirements.txt
└── README.md
```

---

## Setup

```bash
pip install -r requirements.txt
```

Run notebooks in order: `1_exploration` → `2_baseline` → `3_attacks` → `4_defense` → `5_results`

Or run the scripts directly:

```bash
python run_trades.py   # train TRADES model
python run_boost.py    # evaluate threshold + pipeline strategies
```

---

## Stack

Python 3.10+ · PyTorch · scikit-learn · pandas · numpy · matplotlib · seaborn
