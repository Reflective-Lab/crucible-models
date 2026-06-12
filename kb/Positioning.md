---
tags: [positioning, pitch, ml, training]
source: llm
date: 2026-06-12
---
# Positioning

Why Crucible exists, why it plays well with LLMs, and the full model
catalog. Companion pitches live in the Ferrox, Arbiter, Soter, and Prism
knowledge bases; this note is the Crucible chapter of the same story.

## Elevator Pitch

Crucible is the **forge of the Converge platform**: the one place where raw
data becomes trained models. The name is the thesis — a crucible is where ore
becomes refined metal under heat; here, datasets become fitted models by
gradient descent in **Burn** (for differentiable architectures like ANFIS)
and classical optimisation in **linfa** (trees, ensembles, SVMs).

The boundary with Prism is a single sentence: **Prism never fits; Crucible
never owns expert rules.** Prism's packs are closed-form arithmetic with
hand-authored parameters; every Crucible model requires a real training
loop. And training itself is first-class: the pipeline is ten
Suggestor-shaped agents — dataset, validation, feature engineering,
hyperparameter search, training, evaluation, registry, monitoring,
deployment, sample inference — runnable from a CLI today, liftable into a
Formation when a real retrain trigger pulls.

It matters because trained models are where rigor usually goes to die:
unversioned artifacts, irreproducible fits, silent drift. Crucible answers
with deterministic-under-seed training, bincode-serialized artifacts, typed
payloads, and predictions that carry `ProvenanceSource::Crucible` into the
governed promotion path — opinions, clearly labeled as opinions.

## Why It Plays Well With LLMs

An LLM cannot learn from your data — its knowledge is frozen at training
time and general-purpose by construction. Crucible is where the platform
grows **domain-specific judgment** the LLM can call:

- The LLM frames the question ("will this loan default?"); a Crucible
  classifier answers from *your* distribution, with class probabilities
  (`predict_proba`), not vibes from pretraining.
- Typed contracts make trained models tool-shaped:
  `ClassificationFeaturesPayload` in, `ClassPredictionPayload` out, with
  tracing spans and provenance — an agent can cite which model artifact,
  trained when, said what.
- The pipeline agents give an agentic system a governed path to *retrain
  itself*: an LLM can propose a retrain, but evaluation gates, the registry,
  and Converge promotion decide what ships.
- Interpretability is a deliberate axis: a single CART tree when the LLM
  (or a human) needs to read the model's reasoning; a forest when accuracy
  outranks explanation.

The LLM generalizes; Crucible specializes. Prism computes what is already
known; Crucible learns what nobody wrote down.

## What It Solves Better Than Anything Else

Crucible's niche is **trained inference inside the governed loop, in pure
Rust**. No Python sidecar, no model server, no pickle files: training and
inference are `unsafe`-forbidden Rust in-process with the Convergence
Engine, deterministic under fixed seed, with artifacts proven end-to-end by
the integration harness (synthetic data → trained random forest → typed
prediction inside a real Engine). For the platform, that means learned
judgment with the same supply-chain, reproducibility, and provenance
discipline as every other extension — which is exactly what off-the-shelf ML
serving stacks don't give you.

## Model Catalog

### Shipping

| Model / capability | Surface | Tagline |
|---|---|---|
| Random Forest (bagging of CART) | `ensembles::RandomForestModel` via linfa-trees | Many weak trees, one strong vote — deterministic under seed. |
| Decision tree (CART) | `trees::DecisionTreeClassifier` | When reading the model's mind matters more than variance reduction. |
| Generic classifier inference | `ClassifierSuggestor<M>` (+ RF and tree aliases) | Any `ClassifierModel`, one typed Suggestor contract. |
| `ClassifierModel` trait | `train` / `predict` / `predict_proba` / `save` / `load` | The narrow contract every fact-emitting classifier signs. |
| Training pipeline | ten agents in `crucible::training` | Dataset to deployment as Suggestor-shaped, auditable steps. |
| Data plumbing | `crucible::ingest` (Polars) | CSV, TSV, Parquet, and Excel into training-ready frames. |
| Loan-default exemplar | `train_loan_default` CLI + harness tests | The worked proof: synthetic data to typed prediction in a real Engine. |

### Planned (pull-driven)

| Model | Surface | Tagline |
|---|---|---|
| Gradient-boosted trees | `ensembles::GradientBoostedClassifier` | Each tree learns from the last one's mistakes. |
| ANFIS | `neuro_fuzzy::AnfisModel` (Burn) | Fuzzy rules that learn their own membership shapes by gradient descent. |
| Kernel SVM | `svm::SvmClassifier` / `svm::SvmRegressor` | Maximum-margin separation, lifted into kernel space. |
| Regression packs | `RegressorModel` trait | A continuous score when a class label is not enough. |
| Clustering packs | K-means, DBSCAN, GMM via linfa-clustering | Learned segment structure when a showcase pulls. |

### Backend stack

| Library | Role | Tagline |
|---|---|---|
| Burn 0.20 | Differentiable packs (ANFIS) | Autodiff and training, ndarray or GPU backends. |
| linfa 0.8 + linfa-trees | Trees, ensembles, SVMs | Classical ML, pure Rust, no Python anywhere. |
| Polars 0.51 | Ingestion | Columnar speed from file to feature matrix. |

## Boundaries (One-Line Reminders)

- Crucible answers: *what does a model fitted to our data predict?*
  (trained opinion with provenance — never promotion authority)
- Prism answers: *what does the data say, closed-form and auditable?*
  (`Observed` / `Argued`)
- Arbiter answers: *should this concrete request be allowed now?* (`Decided`)
- Ferrox answers: *what is the best feasible plan?* (`Searched`, optimization)
- Soter answers: *can any modeled request violate this invariant?*
  (`Searched`, symbolic)
- Prism never fits; Crucible never owns expert rules — see
  [[Architecture/Project Boundary]] and [[Architecture/Backend Library Choices]].
