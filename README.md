# crucible-models

**Trained-model packs for the Converge Engine.**

`crucible-models` is the companion to
[`prism-analytics`](../prism-analytics) — while prism owns deterministic,
hand-authored inference, crucible owns models that must be *fit to data*.

> The crucible is where raw ore becomes refined metal under intense heat.
> Here, raw data becomes trained models via Burn.

## Project boundary

| prism-analytics | crucible-models |
|-----------------|-----------------|
| Mamdani / Sugeno / Tsukamoto FIS (expert rules) | ANFIS (learned MF parameters) |
| Gaussian Naive Bayes (given priors) | Random Forest (learned tree structure) |
| Logistic / linear regression (pre-fit weights) | SVM (kernel SVM via QP) |
| K-means (inference from centroids) | Gradient-boosted trees |
| All closed-form, no training pipeline | All require a Burn training loop |

## Planned packs

- **`trees`** — CART Decision Tree
- **`ensembles`** — Random Forest, gradient-boosted trees (XGBoost-style)
- **`svm`** — SVM with RBF, polynomial, and linear kernels
- **`neuro_fuzzy`** — ANFIS via Burn autodiff

## Stack

- **Burn 0.20** — training, autodiff, ndarray/GPU backends
- **Converge 3.8.1** — pack and suggestor contracts
- MSRV 1.94.0 · Edition 2024 · `unsafe_code = "forbid"`

## Commands

```bash
cargo build
cargo test
cargo clippy
```
