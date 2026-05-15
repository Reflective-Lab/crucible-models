---
tags: [planning, roadmap]
source: mixed
date: 2026-05-15
---
# Capability Roadmap

What learned-model capabilities `crucible-models` should cover, in what
order, with which backend. Pull-driven: every capability lands only when
an app pulls.

See also: [Architecture/Backend Library Choices](../Architecture/Backend%20Library%20Choices.md),
[Architecture/Model Types](../Architecture/Model%20Types.md),
[Architecture/Project Boundary](../Architecture/Project%20Boundary.md).

## What ships today

| Capability | Pack | Backend | Trait | Suggestor |
|---|---|---|---|---|
| Single CART | `trees::DecisionTreeClassifier` | linfa-trees | `ClassifierModel` | `DecisionTreeClassifierSuggestor` |
| Bagging ensemble | `ensembles::RandomForestModel` | linfa-trees + own bagging | `ClassifierModel` | `RandomForestClassifierSuggestor` |
| Training pipeline | `training::{DatasetAgent, … , DeploymentAgent}` | Polars + linfa | n/a | runs from CLI today |
| Typed payloads | `types::{ClassificationFeaturesPayload, ClassPredictionPayload}` | converge-pack `FactPayload` | n/a | consumed by classifier Suggestors |

## Next capabilities, in order

Order is set by which scenario in `atelier-showcase` is most likely to
pull. Each step adds one new trait or one new pack — never both in a
single slice.

### 1. Gradient-boosted classifier (same trait)

`ensembles::GradientBoostedClassifier` implementing `ClassifierModel`.

- **Backend**: linfa-trees (sequential boosted-residual trees) or
  smartcore if linfa's coverage is thin; bind early and decide.
- **Trait**: no change. Same `ClassifierModel` shape as
  `RandomForestModel`. New `GradientBoostedClassifierSuggestor` is one
  type-alias line.
- **Pull**: same loan-application pull, once we want to compare RF and
  GBT on the validation metric.
- **Why next**: zero new abstractions; proves the trait scales.

### 2. Regression (new `RegressorModel` trait)

A companion trait for continuous-target prediction:

```rust
pub trait RegressorModel: Sized {
    type Config: Send + Sync;
    fn train(config: &Self::Config, features: &Array2<f64>, targets: &Array1<f64>) -> Result<Self>;
    fn predict(&self, features: &Array2<f64>) -> Result<Array1<f64>>;
    fn save(&self, path: &Path) -> Result<()>;
    fn load(path: &Path) -> Result<Self>;
}
```

Plus matching typed payloads (`RegressionFeaturesPayload`,
`RegressionPredictionPayload` with point estimate + optional
prediction-interval bounds) and a generic `RegressorSuggestor<R>`
mirroring `ClassifierSuggestor`.

- **Backends**: linfa-elasticnet (linear), linfa-trees (regression
  trees / random-forest regressor — same crate, regression variant),
  Burn (deep regression if a deep model is pulled).
- **Pull**: an app that needs a continuous score (loan default
  *probability* is already covered by `predict_proba`; this is for
  *amount-predicted-loss*, *time-to-default-days*, *expected-claim-cost*).
- **Status**: deferred. Open the trait when an app names the target.

### 3. Clustering (new `ClusteringModel` trait)

A different protocol — no labels at training time:

```rust
pub trait ClusteringModel: Sized {
    type Config: Send + Sync;
    fn train(config: &Self::Config, features: &Array2<f64>) -> Result<Self>;
    fn assign(&self, features: &Array2<f64>) -> Result<Array1<usize>>;
    fn cluster_centroids(&self) -> Option<&Array2<f64>>;
    fn save(&self, path: &Path) -> Result<()>;
    fn load(path: &Path) -> Result<Self>;
}
```

Typed payloads: `ClusteringFeaturesPayload` (input) and
`ClusterAssignmentPayload` (output, with cluster index + distance to
centroid). A `ClusteringSuggestor<C>` follows the same pattern.

- **Backends**: linfa-clustering (K-means, DBSCAN, Gaussian-mixture).
- **Pull**: vendor-segmentation or customer-segmentation showcases.
- **Notable**: clustering is unsupervised, so the existing training
  pipeline's `DataValidationAgent` / `ModelEvaluationAgent` need
  alternative metrics (silhouette score, Davies–Bouldin index) rather
  than accuracy / AUC. Open as a sub-slice.
- **Status**: deferred until a real segmentation pull.

### 4. ANFIS — the only Burn pack

`neuro_fuzzy::AnfisModel`. The original "via Burn" framing applies
here: ANFIS is a fuzzy inference system whose membership-function
parameters are learned by gradient descent. Burn provides the
autodiff. Crucible inherits the Sugeno rule shape from
`prism-analytics/src/fuzzy/` at the data-format level — `prism`
exposes the rules and consequents, `crucible` learns the parameters.

- **Trait**: either `RegressorModel` or a new
  `NeuroFuzzyModel` trait if rule activations need to be returned as
  part of the prediction (for explainability).
- **Pull**: an app where ANFIS's interpretability matters and a
  Random Forest is too opaque. Niche but real.
- **Status**: deferred.

### 5. Kernel SVMs

`svm::SvmClassifier` and `svm::SvmRegressor` against the existing
`ClassifierModel` / `RegressorModel` traits.

- **Backend**: linfa-svm (or smartcore for richer kernel choice).
- **Pull**: small-feature, sharp-boundary problems — vendor-risk
  scoring is the textbook example.
- **Status**: deferred.

## What deliberately does not belong here

The boundary in [Architecture/Project Boundary](../Architecture/Project%20Boundary.md)
stays sharp:

- **Closed-form / expert-rule inference** stays in
  `prism-analytics`: Mamdani / Sugeno / Tsukamoto FIS with
  hand-authored rules, pre-fit logistic regression with frozen
  weights, Naive Bayes with hand-set priors, K-means inference from
  pre-computed centroids.
- **Anomaly detection on a fitted baseline** lives in
  `prism-analytics` when the baseline is closed-form (z-score,
  Mahalanobis); crucible owns it only if a learned model (e.g.
  isolation forest, autoencoder) is required.
- **Ranking** is the same call: closed-form rank fusion stays in
  prism; learned listwise / pairwise rankers (LambdaMART, RankNet)
  belong in crucible if an app pulls.

## Continuous-learning position (for reference)

The agents in `training.rs` (`MonitoringAgent`, `DeploymentAgent`,
`ModelRegistryAgent`) and the Suggestor-friendly inference path
(`ClassifierSuggestor`) are the substrate for online retraining.
Today they run from a CLI; an app-pull on drift-triggered or
schedule-triggered retraining converts them into a Formation. Full
closed-loop autonomy (Experience → drift signal → retrain → AB →
deploy with humans only on escalations) is a deliberate vision
target, not an early-release claim. Each capability above
participates in the loop; none of them require the loop to be
useful in early releases.
