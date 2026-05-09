---
tags: [architecture, models]
source: llm
date: 2026-05-09
---
# Model Types

Inventory of planned models in `crucible-models`, with algorithm sketches and
training/inference input shapes.

---

## 1. Decision Tree (CART)

**Module:** `trees`

**Algorithm:** Recursive binary splitting on the feature/threshold pair that
maximises information gain (Gini or entropy for classification) or minimises
MSE (regression).  Stopping criteria: pure node, `max_depth`, or
`min_samples_split`.

**Training input shape:**
```json
{
  "records": [[f64, ...]],
  "labels":  [usize, ...],
  "max_depth": 10,
  "min_samples_split": 2,
  "criterion": "gini"
}
```

**Why Burn:** Burn's `Record` mechanism serialises the tree structure to a
portable artifact.  The feature-comparison pass can be vectorised over the
batch dimension via Burn tensors.

---

## 2. Random Forest

**Module:** `ensembles`

**Algorithm:** Bootstrap aggregation of `n_estimators` CART trees, each
trained on a random subsample with `max_features` candidate splits per node.
Prediction: majority vote (classification) or mean (regression).

**Training input shape:**
```json
{
  "records": [[f64, ...]],
  "labels":  [usize, ...],
  "n_estimators": 100,
  "max_features": "sqrt",
  "max_depth": null
}
```

**Why Burn:** Trees are independent → trivially data-parallel.  Burn's
backend can run multiple trees concurrently on GPU.

---

## 3. Gradient-Boosted Trees (XGBoost-style)

**Module:** `ensembles`

**Algorithm:** Sequential weak learner training.  Each tree fits the negative
gradient (pseudo-residuals) of the previous ensemble.  Regularisation via L2
leaf-weight penalty `lambda` and minimum gain to split `gamma`.

**Training input shape:**
```json
{
  "records": [[f64, ...]],
  "labels":  [usize, ...],
  "n_estimators": 200,
  "learning_rate": 0.1,
  "max_depth": 6,
  "subsample": 0.8,
  "lambda": 1.0,
  "gamma": 0.0
}
```

**Why Burn:** Differentiable loss functions; Burn's autodiff computes the
gradient explicitly.  The second-order Taylor approximation of the loss
becomes a Burn tensor operation.

---

## 4. Support Vector Machine

**Module:** `svm`

**Algorithm:** Maximum-margin hyperplane.  Dual QP solved via Sequential
Minimal Optimisation (SMO) or Burn subgradient descent.  Kernel functions:
linear, RBF, polynomial, sigmoid.

**Training input shape:**
```json
{
  "records": [[f64, ...]],
  "labels":  [i32, ...],
  "kernel":  "rbf",
  "C": 1.0,
  "gamma": 0.1,
  "degree": 3,
  "coef0": 0.0
}
```

**Why Burn:** Kernel matrix computation is a batch matrix multiply —
`K = X @ X.T` for linear, pointwise functions for RBF.  Burn tensors make
this backend-agnostic.

---

## 5. ANFIS (Adaptive Neuro-Fuzzy Inference System)

**Module:** `neuro_fuzzy`

**Algorithm:** Five-layer Sugeno FIS whose Gaussian MF parameters (μ, σ) and
linear consequent coefficients are learned via backpropagation.

| Layer | Operation | Learned? |
|-------|-----------|---------|
| 1 — Fuzzification | Gaussian MF evaluation | yes (μ, σ) |
| 2 — Rule firing | Product t-norm | no |
| 3 — Normalisation | strength / Σ strengths | no |
| 4 — Consequent | Linear Sugeno: `p·x + q·y + r` | yes (p, q, r) |
| 5 — Output | Weighted sum | no |

**Training input shape:**
```json
{
  "records":       [[f64, ...]],
  "targets":       [f64, ...],
  "n_rules":       5,
  "epochs":        100,
  "learning_rate": 0.01,
  "batch_size":    32
}
```

**Why Burn:** Backprop through layers 1 and 4 is the defining feature of
ANFIS.  Burn's autodiff backend differentiates through the Gaussian MF
evaluation automatically.

**Difference from prism-analytics Sugeno:** prism-analytics Sugeno takes
expert-authored MF parameters and rules at call time — no training.  ANFIS
learns those parameters from data.

---

## Roadmap order (tentative)

1. Decision Tree — simplest learned model, validates Burn artifact pipeline
2. Random Forest — parallelism, ensemble infrastructure
3. Gradient Boosted Trees — sequential learning, requires differentiable loss
4. SVM — kernel algebra, QP solver
5. ANFIS — most complex; requires neuro-fuzzy layer construction in Burn
