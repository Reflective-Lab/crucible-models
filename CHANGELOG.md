# Changelog

All notable changes to crucible-models will be documented in this file.

The format is based on [Keep a Changelog](https://keepachangelog.com/en/1.0.0/),
and this project adheres to [Semantic Versioning](https://semver.org/spec/v2.0.0.html).

## [Unreleased]

## [0.2.2] - 2026-05-17

### Changed
- `crucible::storage` is now a deprecation shim re-exporting
  `converge_storage::polars_bridge::{fetch_parquet, fetch_to_cache,
  write_parquet_to_store, scan_local_parquet}`. The implementation
  moved to `converge-storage 3.9.1` so any extension that needs
  Parquet from a remote `ObjectStore` (training pipelines, KB
  persistence, app-level dataset ingest) shares a single source of
  truth. The crucible-side `pub use` items carry
  `#[deprecated(since = "0.2.2", ...)]` pointing at the new location.
- `converge-pack` and `converge-optimization` pins bumped to `3.9.1`
  to align with the platform release that introduced
  `converge-storage::polars_bridge`.

## [0.2.0] - 2026-05-15

### Changed (BREAKING)

- `crucible::types::ClassPredictionPayload` bumped from family-version
  `1` to `2`. New required field `execution_identity:
  converge_pack::ExecutionIdentity` records producer crate, backend
  (`linfa-trees-v0.8` for both RF and DT today), and serialized
  runtime config (the model's hyperparameters) so downstream audit
  and replay can answer "which library version, with which
  hyperparameters, produced this prediction?" without re-opening the
  artifact. Mirrors the existing audit pattern in Ferrox and Soter.
  v1 had no external consumers; only in-tree call sites need to pass
  the new argument to `ClassPredictionPayload::new`.

### Added

- `crucible::model::ClassifierModel::execution_identity(&self) ->
  ExecutionIdentity` — required trait method that every classifier
  pack implements. `RandomForestModel` and `DecisionTreeClassifier`
  both return a non-native identity anchored to the workspace's
  linfa-trees pin and the serialized model config.
- `ClassifierSuggestor::execute` now fills
  `ClassPredictionPayload.execution_identity` from
  `self.model.execution_identity()` so every emitted prediction is
  audit-trail complete. Verified end-to-end by the integration
  harness test (asserts producer name + backend string).
- `crucible::model::ClassifierModel` trait — narrow classifier surface
  (`train`, `n_classes`, `predict`, `predict_proba`, `save`, `load`)
  shared by every fact-emitting pack in the crate. A companion
  `RegressorModel` will land when a continuous-target app pulls.
- `crucible::trees::DecisionTreeClassifier` and `DecisionTreeConfig`
  — single-tree CART classifier on top of `linfa_trees::DecisionTree`,
  implementing `ClassifierModel`. Useful for interpretable single-tree
  inference; bincode-serializable; 4 unit tests cover separability,
  one-hot `predict_proba`, save/load round-trip, and mismatched-length
  rejection.
- `crucible::types::ClassificationFeaturesPayload` and
  `ClassPredictionPayload` — typed `FactPayload` implementations in
  the `crucible.classification.*` family. Suggestor input/output
  contract for classifier inference. 3 unit tests cover round-trip and
  stable family/version strings.
- `crucible::suggestor::ClassifierSuggestor<M: ClassifierModel>` —
  generic inference Suggestor that reads
  `ClassificationFeaturesPayload`s from a configurable input
  `ContextKey`, runs `predict_proba`, and emits
  `ClassPredictionPayload`s under a configurable output key with
  `ProvenanceSource::Crucible` and a `crucible.suggestor.execute`
  tracing span.
- Type aliases at the crate root:
  `DecisionTreeClassifierSuggestor = ClassifierSuggestor<DecisionTreeClassifier>`
  and
  `RandomForestClassifierSuggestor = ClassifierSuggestor<RandomForestModel>`.
  Integration tests of the Suggestors against a real `Context` live
  in the loan-application showcase (atelier-showcase, slice 2e).
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
- Added the repository security-audit gate and `cargo-deny` policy, with
  explicit advisory ignores for the currently accepted ML/data dependency
  chain.

### Notes

- Public agent and type names are preserved across the lift, so the only
  downstream change is the import path: `prism::training::DatasetAgent`
  becomes `crucible::training::DatasetAgent`.
- The hard-coded California-housing dataset URL that lived in prism's
  `training.rs` is retained for now but flagged for removal in favour of an
  explicit fixture before the loan-application showcase wiring.
