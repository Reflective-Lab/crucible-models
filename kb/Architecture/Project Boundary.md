---
tags: [architecture, strategy, boundary]
source: human
date: 2026-05-09
---
# Project Boundary — prism-analytics vs. crucible-models

## The decision

prism-analytics is a **pure inference library** — no training, no learned
weights, no gradient descent.  `crucible-models` is the companion project
that owns trained models.

## What belongs in prism-analytics

- Hand-authored rules (Mamdani, Sugeno, Tsukamoto FIS)
- Closed-form statistics (z-score, descriptive stats, SES, OLS)
- Inference from pre-fit parameters (logistic classifier, linear regression,
  k-means, cosine similarity, ranking, trend detection, Gaussian Naive Bayes)
- Anything where the "model" is fully described by its inputs at call time

The defining property: **deterministic, explainable, no training pipeline**.

## What belongs in crucible-models (here)

- Random Forests and gradient boosted trees (XGBoost-style ensembles)
- ANFIS — Adaptive Neuro-Fuzzy Inference System (learned Sugeno via Burn)
- SVM with kernel functions (dual QP training)
- Decision Trees (CART, learned splits from data)
- Any model whose parameters come from fitting to data

Training stack: **Burn 0.20** (native Rust, GPU-capable via WGPU).

## The fuzzy boundary

Mamdani and Sugeno with **expert-authored rules** → prism-analytics.

ANFIS → crucible-models.  ANFIS is Sugeno + backprop on Gaussian MF
parameters, which requires a training pipeline and loss function — exactly
what Burn is for.

## Why the split

- Keeps prism-analytics dependency-light and audit-friendly (no training deps,
  no large binary artifacts).
- Training pipelines have different release cadences, data governance concerns,
  and compute requirements than inference libraries.
- Explainability and determinism are easier to guarantee when there are no
  learned weights in the loop.

Canonical source: also documented in
`prism-analytics/kb/Architecture/Project Boundary.md`.
