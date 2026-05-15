# crucible-models

**Trained-model packs and training pipeline for the Converge Engine.**

`crucible-models` is the companion to
[`prism-analytics`](../prism-analytics) — while prism owns deterministic,
hand-authored inference, crucible owns models that must be *fit to data*.

> The crucible is where raw ore becomes refined metal under intense heat.
> Here, raw data becomes trained models — by gradient descent in **Burn**
> for differentiable architectures (ANFIS), and by classical optimisation
> in **linfa** for everything else (Random Forest, CART, kernel SVMs).
> See [`kb/Architecture/Backend Library Choices.md`](kb/Architecture/Backend%20Library%20Choices.md)
> for the per-pack rationale.

## Project boundary

| prism-analytics | crucible-models |
|-----------------|-----------------|
| Mamdani / Sugeno / Tsukamoto FIS (expert rules) | ANFIS (learned MF parameters via Burn autodiff) |
| Gaussian Naive Bayes (given priors) | Random Forest (learned tree structure via linfa-trees) |
| Logistic / linear regression (pre-fit weights) | Kernel SVM (linfa-svm, planned) |
| K-means inference (centroids provided) | Gradient-boosted trees (planned) |
| All closed-form, no training pipeline | All require a real training loop |

The boundary was restored on 2026-05-15 by lifting the training pipeline
back out of prism into crucible. Prism never fits; crucible never owns
expert rules.

## Shipping today

- **Training pipeline** as Suggestor-shaped agents in `crucible::training`:
  `DatasetAgent`, `DataValidationAgent`, `FeatureEngineeringAgent`,
  `HyperparameterSearchAgent`, `ModelTrainingAgent`,
  `ModelEvaluationAgent`, `ModelRegistryAgent`, `MonitoringAgent`,
  `DeploymentAgent`, `SampleInferenceAgent`. Runs from a CLI today; lifts
  into a Formation when a real retrain trigger pulls.
- **Data plumbing** in `crucible::ingest` (CSV / TSV / Parquet / Excel via
  Polars) and `crucible::storage` (Polars ⇄ `converge-storage::ObjectStore`
  bridge for `gs://`, `s3://`, `file://`, MinIO — behind the `storage`
  feature).
- **`ClassifierModel` trait** — narrow contract (`train`, `n_classes`,
  `predict`, `predict_proba`, `save`, `load`) implemented by every
  fact-emitting classification pack.
- **`crucible::ensembles::RandomForestModel`** — real bagging-of-CART on
  top of `linfa_trees::DecisionTree`. Deterministic under fixed seed.
  Bincode-serializable.
- **`crucible::trees::DecisionTreeClassifier`** — single CART tree, also
  via linfa-trees. Useful when interpretability matters more than variance
  reduction.
- **`crucible::ClassifierSuggestor<M>`** — generic inference Suggestor
  with type aliases `RandomForestClassifierSuggestor` and
  `DecisionTreeClassifierSuggestor`. Reads typed
  `ClassificationFeaturesPayload`s from a configurable input
  `ContextKey`; emits typed `ClassPredictionPayload`s under a
  configurable output key with `ProvenanceSource::Crucible` and the
  `crucible.suggestor.execute` tracing span.
- **Typed payloads** in `crucible::types`:
  `ClassificationFeaturesPayload` (`crucible.classification.features` v1)
  and `ClassPredictionPayload` (`crucible.classification.prediction` v1).
- **Synthetic loan-default fixture** + training CLI:
  ```sh
  cargo run --bin train_loan_default
  ```
  Generates a deterministic 1,000-sample dataset, trains a 50-tree RF,
  prints validation accuracy + confusion matrix, and writes the artifact
  to `crucible-models/artifacts/`.
- **End-to-end Engine integration** verified by
  [`mosaic-integration-harness/tests/crucible_loan_classifier.rs`](../integration-harness/tests/crucible_loan_classifier.rs)
  — three tests prove the full pipeline from synthetic data through
  trained artifact to typed prediction inside a real Convergence Engine.

## Planned packs

Pull-driven; each lands when an app pulls. See
[`kb/Planning/Capability Roadmap.md`](kb/Planning/Capability%20Roadmap.md)
for the order and rationale.

- **`ensembles::GradientBoostedClassifier`** — same `ClassifierModel`
  trait, sequential boosted residuals.
- **`neuro_fuzzy::AnfisModel`** — the only Burn pack. ANFIS is a fuzzy
  inference system whose membership-function parameters are learned by
  gradient descent.
- **`svm::SvmClassifier`** and **`svm::SvmRegressor`** — kernel methods.
- **`RegressorModel` trait + regression packs** — when an app needs a
  continuous score rather than a class probability.
- **`ClusteringModel` trait + clustering packs** (K-means, DBSCAN, GMM
  via linfa-clustering) — when a segmentation showcase pulls.

## Stack

- **Burn 0.20** — autodiff, training, ndarray / GPU backends. Used for
  ANFIS only.
- **linfa 0.8** + **linfa-trees 0.8** — tree ensembles, CART, SVMs. Pins
  ndarray 0.16; crucible uses ndarray 0.17 elsewhere and bridges at the
  linfa boundary via a renamed `ndarray-linfa` dep.
- **Polars 0.51** — tabular ingestion and parquet I/O.
- **converge-storage 3.8.1** — `ObjectStore` abstraction (feature-gated
  `storage`).
- **Converge 3.8.1** — pack and suggestor contracts.
- MSRV 1.94.0 · Edition 2024 · `unsafe_code = "forbid"`

## Commands

```bash
cargo build                                 # default build
cargo build --features storage              # with ObjectStore bridge
cargo test --all-features                   # full test suite
cargo run --bin train_loan_default          # train and persist an RF
cargo clippy --workspace --all-targets -- -D warnings
```
