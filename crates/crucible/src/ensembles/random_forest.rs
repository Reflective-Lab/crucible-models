//! Random Forest classifier.
//!
//! This module ships a scaffolded [`RandomForestModel`] that implements
//! [`crate::model::ClassifierModel`] with a stub training body. The
//! trait shape, public surface, and bincode serialization are stable;
//! the actual bagging-of-CART-trees training lands in a follow-up
//! slice (see `crucible-models/kb/Planning/MILESTONES.md`).

use std::{fs, path::Path};

use anyhow::{Context as _, Result};
use ndarray::{Array1, Array2};
use serde::{Deserialize, Serialize};

use crate::model::ClassifierModel;

/// Hyperparameters for [`RandomForestModel`].
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct RandomForestConfig {
    pub n_trees: usize,
    pub max_depth: Option<usize>,
    pub min_samples_split: usize,
    pub random_seed: u64,
}

impl Default for RandomForestConfig {
    fn default() -> Self {
        Self {
            n_trees: 100,
            max_depth: None,
            min_samples_split: 2,
            random_seed: 0,
        }
    }
}

/// Random Forest classifier.
///
/// TODO(slice-2b): real bagging-of-CART-trees implementation. The
/// current body records the dominant class observed at training and
/// returns it as the prediction for every sample, with uniform
/// per-class probabilities. This is enough to round-trip the trait,
/// the artifact, and the training-pipeline shape; it is not a model
/// to ship into a decision.
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct RandomForestModel {
    config: RandomForestConfig,
    n_classes: usize,
    dominant_class: usize,
}

impl ClassifierModel for RandomForestModel {
    type Config = RandomForestConfig;

    fn train(
        config: &Self::Config,
        features: &Array2<f64>,
        labels: &Array1<usize>,
    ) -> Result<Self> {
        anyhow::ensure!(
            features.nrows() == labels.len(),
            "feature/label length mismatch: {} rows vs {} labels",
            features.nrows(),
            labels.len()
        );
        anyhow::ensure!(
            !labels.is_empty(),
            "RandomForest training requires at least one labelled sample"
        );

        let n_classes = labels.iter().copied().max().map_or(0, |m| m + 1);

        let dominant_class = {
            let mut counts = vec![0usize; n_classes];
            for &y in labels {
                counts[y] += 1;
            }
            counts
                .iter()
                .enumerate()
                .max_by_key(|&(_, &c)| c)
                .map_or(0, |(idx, _)| idx)
        };

        Ok(Self {
            config: config.clone(),
            n_classes,
            dominant_class,
        })
    }

    fn n_classes(&self) -> usize {
        self.n_classes
    }

    fn predict(&self, features: &Array2<f64>) -> Result<Array1<usize>> {
        Ok(Array1::from_elem(features.nrows(), self.dominant_class))
    }

    fn predict_proba(&self, features: &Array2<f64>) -> Result<Array2<f64>> {
        let uniform = 1.0_f64 / self.n_classes.max(1) as f64;
        Ok(Array2::from_elem(
            (features.nrows(), self.n_classes),
            uniform,
        ))
    }

    fn save(&self, path: &Path) -> Result<()> {
        let bytes = bincode::serialize(self).context("serializing RandomForestModel artifact")?;
        fs::write(path, bytes).with_context(|| format!("writing {}", path.display()))?;
        Ok(())
    }

    fn load(path: &Path) -> Result<Self> {
        let bytes = fs::read(path).with_context(|| format!("reading {}", path.display()))?;
        let model = bincode::deserialize::<Self>(&bytes)
            .context("deserializing RandomForestModel artifact")?;
        Ok(model)
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use ndarray::{Axis, array};

    #[test]
    fn trains_and_predicts_dominant_class_stub() {
        let features = array![[0.0, 0.0], [1.0, 1.0], [2.0, 2.0], [3.0, 3.0]];
        let labels = array![1, 1, 0, 1];
        let model =
            RandomForestModel::train(&RandomForestConfig::default(), &features, &labels).unwrap();

        assert_eq!(model.n_classes(), 2);
        let pred = model.predict(&features).unwrap();
        assert!(pred.iter().all(|&c| c == 1));
    }

    #[test]
    fn predict_proba_rows_sum_to_one() {
        let features = array![[0.0, 0.0], [1.0, 1.0]];
        let labels = array![0, 1];
        let model =
            RandomForestModel::train(&RandomForestConfig::default(), &features, &labels).unwrap();

        let proba = model.predict_proba(&features).unwrap();
        for row in proba.axis_iter(Axis(0)) {
            let s: f64 = row.iter().sum();
            assert!((s - 1.0).abs() < 1e-9, "row sum {s} != 1.0");
        }
    }

    #[test]
    fn save_load_roundtrip() {
        let features = array![[0.0, 1.0], [2.0, 3.0], [4.0, 5.0]];
        let labels = array![0, 1, 1];
        let model =
            RandomForestModel::train(&RandomForestConfig::default(), &features, &labels).unwrap();

        let tmp = tempfile::NamedTempFile::new().unwrap();
        model.save(tmp.path()).unwrap();
        let loaded = RandomForestModel::load(tmp.path()).unwrap();

        assert_eq!(loaded.n_classes(), model.n_classes());
        assert_eq!(loaded.dominant_class, model.dominant_class);
    }

    #[test]
    fn train_rejects_mismatched_lengths() {
        let features = array![[0.0]];
        let labels = array![0, 1];
        let err = RandomForestModel::train(&RandomForestConfig::default(), &features, &labels)
            .unwrap_err();
        assert!(err.to_string().contains("feature/label length mismatch"));
    }

    #[test]
    fn train_rejects_empty_labels() {
        let features = Array2::<f64>::zeros((0, 1));
        let labels = Array1::<usize>::zeros(0);
        let err = RandomForestModel::train(&RandomForestConfig::default(), &features, &labels)
            .unwrap_err();
        assert!(err.to_string().contains("at least one labelled sample"));
    }
}
