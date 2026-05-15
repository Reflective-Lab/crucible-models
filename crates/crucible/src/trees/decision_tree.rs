//! Single CART decision-tree classifier.
//!
//! Wraps `linfa_trees::DecisionTree<f64, usize>` to implement
//! [`crate::model::ClassifierModel`]. Unlike
//! [`crate::ensembles::RandomForestModel`] this is a single tree with
//! no bagging — useful when interpretability matters more than
//! variance reduction, or as the unit element from which Random
//! Forests are built.
//!
//! Per the backend-library decision in
//! `kb/Architecture/Backend Library Choices.md`, the backend is
//! `linfa-trees`, not Burn. Decision trees are non-differentiable.

use std::{fs, path::Path};

use anyhow::{Context as _, Result};
use converge_pack::ExecutionIdentity;
use linfa::Dataset;
use linfa::prelude::*;
use linfa_trees::DecisionTree;
use ndarray::{Array1, Array2};
use ndarray_linfa as ndl;
use serde::{Deserialize, Serialize};

use crate::model::ClassifierModel;

/// Backend identifier embedded in every `DecisionTreeClassifier`'s
/// `ExecutionIdentity`. Tracks the workspace's linfa-trees pin.
const DT_BACKEND: &str = "linfa-trees-v0.8";

/// Hyperparameters for [`DecisionTreeClassifier`].
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct DecisionTreeConfig {
    /// Maximum depth of the tree. `None` means unlimited.
    pub max_depth: Option<usize>,
    /// Minimum sum-of-sample-weights to allow a split. Linfa default
    /// is `2.0`.
    pub min_weight_split: f32,
}

impl Default for DecisionTreeConfig {
    fn default() -> Self {
        Self {
            max_depth: None,
            min_weight_split: 2.0,
        }
    }
}

/// Single CART decision-tree classifier.
///
/// Built on `linfa_trees::DecisionTree<f64, usize>`. `predict` returns
/// the crisp leaf class for each row; `predict_proba` returns 1.0 for
/// the predicted class and 0.0 for the others (a single tree has no
/// probabilistic output of its own — for graded confidences, use
/// [`crate::ensembles::RandomForestModel`]).
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct DecisionTreeClassifier {
    config: DecisionTreeConfig,
    n_classes: usize,
    tree: DecisionTree<f64, usize>,
}

fn to_linfa_features(arr: &Array2<f64>) -> ndl::Array2<f64> {
    let (rows, cols) = arr.dim();
    let data: Vec<f64> = arr.iter().copied().collect();
    ndl::Array2::from_shape_vec((rows, cols), data).expect("shape known correct from source array")
}

impl ClassifierModel for DecisionTreeClassifier {
    type Config = DecisionTreeConfig;

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
            "DecisionTree training requires at least one labelled sample"
        );

        let n_classes = labels.iter().copied().max().map_or(0, |m| m + 1);

        let feat_linfa = to_linfa_features(features);
        let lab_linfa = ndl::Array1::from_vec(labels.iter().copied().collect());
        let dataset = Dataset::new(feat_linfa, lab_linfa);

        let tree = DecisionTree::<f64, usize>::params()
            .max_depth(config.max_depth)
            .min_weight_split(config.min_weight_split)
            .fit(&dataset)
            .context("fitting DecisionTree")?;

        Ok(Self {
            config: config.clone(),
            n_classes,
            tree,
        })
    }

    fn n_classes(&self) -> usize {
        self.n_classes
    }

    fn predict(&self, features: &Array2<f64>) -> Result<Array1<usize>> {
        let features_linfa = to_linfa_features(features);
        let preds_linfa: ndl::Array1<usize> = self.tree.predict(&features_linfa);
        Ok(Array1::from_vec(preds_linfa.to_vec()))
    }

    fn predict_proba(&self, features: &Array2<f64>) -> Result<Array2<f64>> {
        // A single decision tree assigns a single class per leaf; the
        // "probability" is 1.0 for that class and 0.0 for the rest.
        let preds = self.predict(features)?;
        let n_rows = features.nrows();
        let n_classes = self.n_classes.max(1);
        let mut proba = Array2::<f64>::zeros((n_rows, n_classes));
        for (row, &cls) in preds.iter().enumerate() {
            if cls < self.n_classes {
                proba[(row, cls)] = 1.0;
            }
        }
        Ok(proba)
    }

    fn save(&self, path: &Path) -> Result<()> {
        let bytes =
            bincode::serialize(self).context("serializing DecisionTreeClassifier artifact")?;
        fs::write(path, bytes).with_context(|| format!("writing {}", path.display()))?;
        Ok(())
    }

    fn load(path: &Path) -> Result<Self> {
        let bytes = fs::read(path).with_context(|| format!("reading {}", path.display()))?;
        let model = bincode::deserialize::<Self>(&bytes)
            .context("deserializing DecisionTreeClassifier artifact")?;
        Ok(model)
    }

    fn execution_identity(&self) -> ExecutionIdentity {
        let runtime_config = serde_json::to_string(&self.config).unwrap_or_default();
        ExecutionIdentity::non_native(
            env!("CARGO_PKG_NAME"),
            env!("CARGO_PKG_VERSION"),
            DT_BACKEND,
            runtime_config,
        )
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use ndarray::array;
    use rand::{Rng as _, SeedableRng as _, rngs::StdRng};

    fn two_cluster_dataset(seed: u64) -> (Array2<f64>, Array1<usize>) {
        let mut rng = StdRng::seed_from_u64(seed);
        let mut feats = Vec::with_capacity(200);
        let mut labs = Vec::with_capacity(100);
        for _ in 0..50 {
            feats.push(-2.0 + rng.gen_range(-0.3..0.3));
            feats.push(-2.0 + rng.gen_range(-0.3..0.3));
            labs.push(0usize);
        }
        for _ in 0..50 {
            feats.push(2.0 + rng.gen_range(-0.3..0.3));
            feats.push(2.0 + rng.gen_range(-0.3..0.3));
            labs.push(1usize);
        }
        (
            Array2::from_shape_vec((100, 2), feats).unwrap(),
            Array1::from_vec(labs),
        )
    }

    #[test]
    fn trains_and_separates_two_clusters() {
        let (features, labels) = two_cluster_dataset(11);
        let model = DecisionTreeClassifier::train(
            &DecisionTreeConfig {
                max_depth: Some(5),
                min_weight_split: 2.0,
            },
            &features,
            &labels,
        )
        .unwrap();

        let preds = model.predict(&features).unwrap();
        let correct = preds
            .iter()
            .zip(labels.iter())
            .filter(|(p, y)| p == y)
            .count();
        assert!(
            correct >= 95,
            "expected >= 95/100 correct on linearly separable train, got {correct}"
        );
    }

    #[test]
    fn predict_proba_is_one_hot_per_row() {
        let (features, labels) = two_cluster_dataset(7);
        let model =
            DecisionTreeClassifier::train(&DecisionTreeConfig::default(), &features, &labels)
                .unwrap();

        let proba = model.predict_proba(&features).unwrap();
        for row in proba.rows() {
            let n_ones = row.iter().filter(|&&v| v == 1.0).count();
            let n_zeros = row.iter().filter(|&&v| v == 0.0).count();
            assert_eq!(n_ones, 1, "row should be one-hot");
            assert_eq!(n_ones + n_zeros, row.len());
        }
    }

    #[test]
    fn save_load_roundtrip_preserves_predictions() {
        let (features, labels) = two_cluster_dataset(2);
        let model =
            DecisionTreeClassifier::train(&DecisionTreeConfig::default(), &features, &labels)
                .unwrap();

        let tmp = tempfile::NamedTempFile::new().unwrap();
        model.save(tmp.path()).unwrap();
        let loaded = DecisionTreeClassifier::load(tmp.path()).unwrap();

        assert_eq!(
            model.predict(&features).unwrap(),
            loaded.predict(&features).unwrap()
        );
    }

    #[test]
    fn train_rejects_mismatched_lengths() {
        let features = array![[0.0]];
        let labels = array![0, 1];
        let err = DecisionTreeClassifier::train(&DecisionTreeConfig::default(), &features, &labels)
            .unwrap_err();
        assert!(err.to_string().contains("feature/label length mismatch"));
    }
}
