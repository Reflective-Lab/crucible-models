---
tags: [architecture, ai-history, concepts]
source: llm
date: 2026-05-09
---
# AI Paradigms

Conceptual map of major AI paradigms — how they differ, where they overlap, and
where each model type in `crucible-models` fits.

Copied from `prism-analytics/kb/Architecture/AI Paradigms.md` for context.
`crucible-models` owns the paradigms that require learned parameters (rows
marked **here** in the table below).

## Summary

| Paradigm | Core Idea | Where |
|---|---|---|
| Symbolic AI | Intelligence via explicit rules and logic | prism-analytics |
| Expert Systems | Symbolic AI specialized for domain expertise | prism-analytics |
| Fuzzy Logic (expert rules) | Reasoning with partial truth | prism-analytics |
| Gaussian Naive Bayes | Closed-form probabilistic classification | prism-analytics |
| **Decision Trees** | Learned symbolic rules from data | **crucible-models** |
| **Random Forests** | Ensemble of learned trees | **crucible-models** |
| **Gradient Boosting** | Sequential weak learner training | **crucible-models** |
| **SVM** | Maximum-margin geometric classification | **crucible-models** |
| **ANFIS** | Fuzzy + neural learning (Sugeno + backprop) | **crucible-models** |
| Generative AI | Learn distributions that create new content | out of scope |

---

## 1. Symbolic AI

Also called: GOFAI ("Good Old-Fashioned AI"), rule-based AI, knowledge-based AI.

**Core philosophy:** represent intelligence explicitly using symbols, logic,
rules, and facts.

```text
IF fever AND cough THEN flu
```

Symbolic AI remains useful where explainability, deterministic behavior, safety,
or limited data are constraints (medical rules, finance compliance, industrial
automation, configuration systems).

---

## 2. Decision Trees

Decision trees sit between symbolic AI and machine learning. They produce
symbolic-looking logic:

```text
IF income > 50k
  IF age < 30
    approve
```

Unlike expert systems, the rules are *statistically learned* from data, not
handcrafted. Trees are interpretable, symbolic, and explainable — but
data-driven.  CART (Classification and Regression Trees) is the canonical
algorithm.

**→ Planned in `crucible-models/trees`.**

---

## 3. Ensemble Methods

| Model | Strategy |
|---|---|
| Random Forest | Bagging of independent trees with random feature subsets |
| Gradient Boosted Trees | Sequential boosting: each tree fits residuals |
| XGBoost-style | Regularised gradient boosting with L2 leaf penalties |

For tabular/business data (finance, fraud, insurance, analytics), gradient
boosted trees often outperform deep learning.

**→ Planned in `crucible-models/ensembles`.**

---

## 4. Support Vector Machines

SVMs find the maximum-margin hyperplane.  Non-linear kernels (RBF, polynomial,
sigmoid) map features into higher-dimensional spaces where linear separation
is possible.

**→ Planned in `crucible-models/svm`.**

---

## 5. ANFIS — Fuzzy + Neural Networks

ANFIS combines fuzzy rules with neural learning: humans define the rule
structure, backpropagation tunes the Gaussian MF parameters and linear
consequent coefficients.

ANFIS is Sugeno FIS + backprop on MF parameters.  It requires a Burn training
pipeline and loss function — which is why it belongs in `crucible-models`
rather than `prism-analytics`.

**→ Planned in `crucible-models/neuro_fuzzy`.**

---

## 6. Big Picture Evolution

```text
Symbolic AI
    ↓
Expert Systems
    ↓
Fuzzy Logic (expert rules)   ←→   prism-analytics
    ↓
Statistical ML
Decision Trees / Random Forests / SVMs / ANFIS   ←→   crucible-models
    ↓
Deep Learning
    ↓
Generative AI / Neuro-symbolic hybrids
```

These are not replacements — modern AI stacks often combine multiple paradigms.

See also: [Project Boundary](Project%20Boundary.md), [Model Types](Model%20Types.md)
