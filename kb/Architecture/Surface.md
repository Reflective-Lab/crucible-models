---
tags: [architecture, surface]
source: llm
date: 2026-05-15
---
# Public Surface — crucible-models

## Crate name

- Cargo package: `converge-crucible-models`
- Rust library name: `crucible`
- crates.io: `converge-crucible-models`

## Top-level module layout

```
crucible/
├── ensembles/      bagging-of-CART (Random Forest); boosted trees (planned)
├── trees/          single CART decision-tree classifier
├── svm/            kernel SVMs (planned)
├── neuro_fuzzy/    ANFIS via Burn (planned — only Burn pack)
├── training/       Suggestor-shaped training pipeline agents
├── ingest/         multi-format dataset readers (CSV / TSV / Parquet / Excel)
├── storage/        Polars ⇄ ObjectStore bridge (feature `storage`)
├── fixtures/       deterministic synthetic datasets for tests and CLIs
├── model/          `ClassifierModel` trait
├── suggestor/      generic `ClassifierSuggestor<M>` inference Suggestor
├── types/          typed FactPayload payloads (features / prediction)
├── provenance/     `ProvenanceSource::Crucible` and span helpers
└── bin/
    └── train_loan_default.rs   loan-default training CLI
```

## Stable trait

```rust
pub trait ClassifierModel: Sized {
    type Config: Send + Sync;
    fn train(config: &Self::Config, features: &Array2<f64>, labels: &Array1<usize>) -> Result<Self>;
    fn n_classes(&self) -> usize;
    fn predict(&self, features: &Array2<f64>) -> Result<Array1<usize>>;
    fn predict_proba(&self, features: &Array2<f64>) -> Result<Array2<f64>>;
    fn save(&self, path: &Path) -> Result<()>;
    fn load(path: &Path) -> Result<Self>;
}
```

Companion `RegressorModel` and `ClusteringModel` traits follow when
pulled. See [Capability Roadmap](../Planning/Capability%20Roadmap.md).

## Models implementing `ClassifierModel`

- `ensembles::RandomForestModel` (+ `RandomForestConfig`) — bagging of
  CART trees via `linfa_trees::DecisionTree<f64, usize>`.
- `trees::DecisionTreeClassifier` (+ `DecisionTreeConfig`) — single
  CART tree.

## Suggestor surface

```rust
pub struct ClassifierSuggestor<M: ClassifierModel> { /* ... */ }

pub type DecisionTreeClassifierSuggestor =
    ClassifierSuggestor<DecisionTreeClassifier>;

pub type RandomForestClassifierSuggestor =
    ClassifierSuggestor<RandomForestModel>;
```

Each Suggestor:

- Reads `ClassificationFeaturesPayload`s from its configured input
  `ContextKey`.
- Runs `model.predict_proba` and computes argmax.
- Emits one `ClassPredictionPayload` per input under its configured
  output key, with `ProvenanceSource::Crucible`.
- Wraps `execute` in a `crucible.suggestor.execute` tracing span
  carrying provenance, suggestor name, context keys, and input count.

## Typed payloads

- `types::ClassificationFeaturesPayload`
  - family `crucible.classification.features`, version `1`
  - input contract for any classifier Suggestor
- `types::ClassPredictionPayload`
  - family `crucible.classification.prediction`, version `1`
  - output contract carrying `predicted_class` + `class_probabilities`

## Training pipeline

`crucible::training` exposes the pipeline as Suggestor-shaped agents.
Today they run from a CLI; they lift into a Formation when a real
retrain trigger pulls.

- `DatasetAgent` — ingest labelled training data via the storage layer
- `DataValidationAgent` — quality checks
- `FeatureEngineeringAgent` — feature-spec application
- `HyperparameterSearchAgent` — hyperparameter sweep
- `ModelTrainingAgent` — fit
- `ModelEvaluationAgent` — held-out metrics
- `ModelRegistryAgent` — artifact registration
- `MonitoringAgent` — drift detection
- `DeploymentAgent` — promotion to active inference
- `SampleInferenceAgent` — sample inference for validation

Plus types: `TrainingPlan`, `DatasetSplit`, `HyperparameterSearchPlan`,
`HyperparameterSearchResult`, `EvaluationReport`, `ModelMetadata`,
`ModelRegistryRecord`, `MonitoringReport`, `DeploymentDecision`,
`FeatureSpec`, `BaselineModel`, `InferenceSample`, `DataQualityReport`,
`FeatureInteraction`.

## Provenance and tracing

- `provenance::CRUCIBLE_PROVENANCE: ProvenanceSource` — constant
  pointing at `ProvenanceSource::Crucible`.
- `provenance::suggestor_span(name, input_key, output_key, count)`
  emits the `crucible.suggestor.execute` span used across the crate.
- Aligns with the workspace
  [Suggestor Contract](../../../kb/Standards/Suggestor%20Contract.md).

## Feature flags

- `default` — minimal surface, no optional native deps.
- `storage` — enables the `crucible::storage` module and pulls
  `converge-storage`.
- `excel` — enables Excel ingestion via `calamine`.

## CLI binaries

- `train_loan_default` — generates a deterministic synthetic
  loan-default dataset, fits a 50-tree Random Forest, prints
  validation accuracy + per-class confusion-matrix counts, and writes
  the artifact to `crucible-models/artifacts/`. Bypasses the
  Suggestor / Convergence-Engine protocol on purpose; the
  Suggestor-driven retrain path lifts later when a real trigger pulls.

## Integration tests (cross-extension)

End-to-end Engine wiring lives in
`mosaic-extensions/integration-harness/tests/crucible_loan_classifier.rs`:

- high-risk loan applicant classified as default
- low-risk loan applicant classified as non-default
- non-features payload (e.g. `TextPayload`) is correctly ignored

These prove the full pipeline from synthetic dataset → trained
artifact → typed prediction inside a real Convergence Engine.

## Status

Public-surface stability:

- `ClassifierModel`, `ClassifierSuggestor`, the two payload types, the
  two type aliases (`DecisionTreeClassifierSuggestor`,
  `RandomForestClassifierSuggestor`), and the training pipeline agents
  are the stable surface as of 2026-05-15.
- `svm::*`, `neuro_fuzzy::*`, and gradient-boosted variants under
  `ensembles::*` remain planned and have no committed surface.
