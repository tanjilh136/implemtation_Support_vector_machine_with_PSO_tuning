# SVM Hyperparameter Tuning with Particle Swarm Optimization — MIT-BIH ECG Classification

## Overview

Academic coursework implementing **Particle Swarm Optimization (PSO)** from scratch to tune the hyperparameters of a **Support Vector Machine (SVM)** for cardiac arrhythmia classification using real clinical ECG data.

The core challenge with SVM is selecting the right values for two hyperparameters — **gamma** (kernel width) and **C** (regularization strength). Poor choices lead to significant performance loss. Grid search is computationally expensive. PSO offers a smarter, population-based search strategy inspired by the collective movement of birds and fish.

---

## Dataset

**MIT-BIH Arrhythmia Database** — a standard clinical benchmark dataset of ECG (electrocardiogram) recordings from real patients.

- **187 features** per sample — digitized ECG signal values per heartbeat
- **5 target classes:**

| Class | Description |
|-------|-------------|
| 0 | Normal Beat |
| 1 | Supraventricular Premature Beat |
| 2 | Premature Ventricular Contraction |
| 3 | Fusion of Ventricular and Normal Beat |
| 4 | Unclassifiable Beat |

- Train/validation/test split: 80/20 train-validation, separate held-out test set
- AUC (Area Under the ROC Curve) used as the evaluation metric — appropriate for the class imbalance present in cardiac data

---

## Algorithm — Particle Swarm Optimization

PSO is a metaheuristic optimization algorithm (Kennedy & Eberhart, 1995) that searches a parameter space using a swarm of candidate solutions (particles). Each particle moves through the search space guided by its own best known position and the global best position found by any particle.

**Constriction factor variant** (Clerc & Kennedy, 2002) is used for stability:

```
khi = 2K / |2 - c - sqrt(c² - 4c)|

where:
  K  = 0.3  (constriction factor)
  c1 = 2.05 (cognitive coefficient — weight toward personal best)
  c2 = 2.05 (social coefficient — weight toward global best)
  c  = c1 + c2
```

**Search space:** gamma ∈ [0.0001, 10], C ∈ [0.001, 10]

**PSO Configuration:**
- 100 particles
- 10 iterations
- Velocity bounds: [-10, 10] per dimension
- Fitness function: 3-fold cross-validation AUC on training data

**Velocity update rule:**
```
v_new = khi * (v + c1*r1*(pbest - x) + c2*r2*(gbest - x))
x_new = x + v_new
```

Where `r1`, `r2` are random values in [0, 1] per update.

---

## Results

After PSO finds the optimal (gamma, C) combination, a final SVM with RBF kernel is trained on the full training set and evaluated on the validation set using `classification_report`.

---

## Project Status

**Academic coursework — known issues:**

- Global best position (`gbest_position`) is not correctly persisted across iterations — the swarm tracks per-iteration bests but does not carry the all-time best forward. This means the PSO runs but does not fully converge as designed. Fix planned.
- Variable scoping issue: boundary values defined inside a loop are relied upon outside it — works in Python by accident due to loop variable leakage, but is not correct practice.
- Unused debug line (`fitness_cadidate_`) is a remnant from testing and has no effect on results.

These issues are documented here for transparency. The algorithmic concept and structure are correct — the implementation has targeted bugs that will be addressed in a future revision.

---

## Tech Stack

- **Language:** Python
- **ML:** scikit-learn (SVC, KFold, cross_val_score, classification_report)
- **Data:** pandas · NumPy
- **Visualization:** matplotlib
- **Dataset:** MIT-BIH Arrhythmia Database (mitbih_train.csv / mitbih_test.csv)

---

## References

- Kennedy, J., & Eberhart, R. (1995). Particle swarm optimization. *Proceedings of ICNN'95*
- Clerc, M., & Kennedy, J. (2002). The particle swarm — explosion, stability, and convergence in a multidimensional complex space. *IEEE Transactions on Evolutionary Computation*

---

## Author

**Tanjil Hasan**
github.com/tanjilh136 · linkedin.com/in/tanjil-hasan-650246169
