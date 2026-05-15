# Changelog

All notable changes to crucible-models will be documented in this file.

The format is based on [Keep a Changelog](https://keepachangelog.com/en/1.0.0/),
and this project adheres to [Semantic Versioning](https://semver.org/spec/v2.0.0.html).

## [Unreleased]

### Added

- `crucible::model::ClassifierModel` trait — narrow classifier surface
  (`train`, `n_classes`, `predict`, `predict_proba`, `save`, `load`)
  shared by every fact-emitting pack in the crate. A companion
  `RegressorModel` will land when a continuous-target app pulls.
- `crucible::ensembles::random_forest::RandomForestModel` and
  `RandomForestConfig` — real bagging-of-CART implementation on top of
  `linfa_trees::DecisionTree`. The training loop fits `n_trees`
  decision trees, each on a bootstrap sample of the training rows;
  `predict` returns the majority vote and `predict_proba` returns the
  per-class vote fraction. Deterministic under a fixed
  `random_seed`. Bincode-serializable artifacts via linfa-trees'
  `serde` feature. 7 unit tests cover: two-cluster separability
  (>= 95/100 correct on linearly-separable data), probability rows sum
  to 1.0, identical-config-identical-data determinism, save/load
  round-trip preserving predictions, and three input-validation error
  paths. Feature subsampling (the "random" in Random Forest's
  split-time feature sampling) is deferred to a follow-up slice when
  an app pulls on the additional variance reduction; the trait shape
  is unchanged either way.
- Workspace deps: `linfa = "0.8"`, `linfa-trees = { version = "0.8",
  features = ["serde"] }`, and a renamed `ndarray-linfa = "0.16"` that
  bridges linfa's ndarray 0.16 against the workspace's ndarray 0.17.
  See `kb/Architecture/Backend Library Choices.md` for the per-pack
  rationale.
- Lifted the training pipeline and supporting data plumbing from
  `prism-analytics` into crucible, restoring the prism / crucible boundary
  (prism = closed-form inference with hand-authored rules; crucible = trained
  models with a Burn-driven training pipeline). The new modules are:
  - `crucible::ingest` — multi-format CSV / TSV / Parquet / Excel readers via
    Polars, with `IngestFormat`, `read_file`, `read_csv`, `read_tsv`,
    `read_parquet`, `read_excel`, `summarize`, and `IngestSummary`.
  - `crucible::storage` (behind the `storage` feature) — Polars ⇄
    `converge-storage::ObjectStore` bridge for `fetch_parquet`,
    `write_parquet_to_store`, and `scan_local_parquet`.
  - `crucible::training` — full training pipeline as Suggestor-shaped agents:
    `DatasetAgent`, `DataValidationAgent`, `FeatureEngineeringAgent`,
    `HyperparameterSearchAgent`, `ModelTrainingAgent`, `ModelEvaluationAgent`,
    `ModelRegistryAgent`, `MonitoringAgent`, `DeploymentAgent`,
    `SampleInferenceAgent`, plus `TrainingPlan`, `DatasetSplit`,
    `HyperparameterSearchPlan`, `EvaluationReport`, `ModelRegistryRecord`,
    `DeploymentDecision`, `FeatureSpec`, `BaselineModel`, and friends.
- `crucible::provenance` with typed `ProvenanceSource::Crucible`, the
  `CRUCIBLE_PROVENANCE` constant, and the `suggestor_span` helper emitting
  `crucible.suggestor.execute` spans. Mirrors the workspace Suggestor Contract.
- Cargo features: `storage` (enables `converge-storage`) and `excel` (enables
  `calamine`). Both off by default to keep the lean build cheap.

### Notes

- Public agent and type names are preserved across the lift, so the only
  downstream change is the import path: `prism::training::DatasetAgent`
  becomes `crucible::training::DatasetAgent`.
- The hard-coded California-housing dataset URL that lived in prism's
  `training.rs` is retained for now but flagged for removal in favour of an
  explicit fixture before the loan-application showcase wiring.
