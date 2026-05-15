//! `Model` trait for trained-artifact packs in `crucible-models`.
//!
//! Implementations are owned by per-pack modules (`ensembles`, `trees`,
//! `svm`, `neuro_fuzzy`). The training pipeline in [`crate::training`]
//! drives instances of [`ClassifierModel`] through the dataset, fitting,
//! evaluation, and registry stages.
//!
//! The trait is intentionally narrow: classifiers receive an integer
//! label vector at training time and emit both crisp class predictions
//! and per-class probability matrices at inference time. A future
//! `RegressorModel` companion trait will mirror this shape for
//! continuous targets when an app pulls.

use std::path::Path;

use anyhow::Result;
use ndarray::{Array1, Array2};

/// A classifier that fits on integer labels and predicts crisp classes
/// plus per-class probabilities.
pub trait ClassifierModel: Sized {
    /// Hyperparameters for this classifier family.
    type Config: Send + Sync;

    /// Fit the classifier to `(features, labels)` under `config`.
    ///
    /// `features` is shape `(n_samples, n_features)`. `labels` is shape
    /// `(n_samples,)` and must contain non-negative integers naming the
    /// class of each sample.
    fn train(config: &Self::Config, features: &Array2<f64>, labels: &Array1<usize>)
    -> Result<Self>;

    /// Number of classes the model was trained on.
    fn n_classes(&self) -> usize;

    /// Predict the most-likely class for each row.
    fn predict(&self, features: &Array2<f64>) -> Result<Array1<usize>>;

    /// Predict per-class probabilities for each row.
    ///
    /// Returned shape is `(n_samples, n_classes)`. Rows sum to 1.0
    /// within floating-point tolerance.
    fn predict_proba(&self, features: &Array2<f64>) -> Result<Array2<f64>>;

    /// Serialize the trained artifact to a path on disk.
    fn save(&self, path: &Path) -> Result<()>;

    /// Deserialize a trained artifact from a path on disk.
    fn load(path: &Path) -> Result<Self>;
}
