//! Random Forest classifier.
//!
//! Implements bagging-of-CART-trees on top of `linfa_trees::DecisionTree`.
//!
//! The training loop fits `config.n_trees` decision trees, each on a
//! bootstrap sample of the training rows (sampling with replacement).
//! At prediction time, each tree votes; the model returns the
//! majority-vote class from `predict` and the per-class vote fraction
//! from `predict_proba`.
//!
//! Per the backend-library decision in
//! `kb/Architecture/Backend Library Choices.md`, the per-tree backend is
//! linfa-trees, not Burn. Burn stays reserved for `neuro_fuzzy`, where
//! gradient descent on membership-function parameters is the right tool.
//!
//! Feature subsampling — the "random" in Random Forest's split-time
//! feature sampling — is not yet implemented. The current loop uses
//! classic bagging only. A follow-up slice adds feature subsampling
//! when an app pulls on the additional variance reduction. The trait
//! shape stays unchanged either way.

use std::{fs, path::Path};

use anyhow::{Context as _, Result};
use converge_pack::ExecutionIdentity;
use linfa::Dataset;
use linfa::prelude::*;
use linfa_trees::DecisionTree;
use ndarray::{Array1, Array2};
use ndarray_linfa as ndl;
use rand::{Rng as _, SeedableRng as _, rngs::StdRng};
use serde::{Deserialize, Serialize};

use crate::model::ClassifierModel;

/// Backend identifier embedded in every `RandomForestModel`'s
/// `ExecutionIdentity`. Tracks the workspace's linfa-trees pin.
const RF_BACKEND: &str = "linfa-trees-v0.8";

/// Hyperparameters for [`RandomForestModel`].
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct RandomForestConfig {
    /// Number of trees in the ensemble. Must be >= 1.
    pub n_trees: usize,
    /// Maximum depth per tree. `None` means unlimited.
    pub max_depth: Option<usize>,
    /// Minimum sum-of-sample-weights to allow a split. Linfa's default
    /// is `2.0`; lowering it permits more aggressive splitting at the
    /// cost of overfit risk per tree (mitigated by the ensemble).
    pub min_weight_split: f32,
    /// Seed for the bootstrap-sampling RNG. Identical seeds + identical
    /// data + identical config yield identical models.
    pub random_seed: u64,
}

impl Default for RandomForestConfig {
    fn default() -> Self {
        Self {
            n_trees: 100,
            max_depth: None,
            min_weight_split: 2.0,
            random_seed: 0,
        }
    }
}

/// Random Forest classifier.
///
/// Bagging ensemble of `linfa_trees::DecisionTree<f64, usize>`. Each
/// tree is fit on a bootstrap sample of the training rows; predictions
/// aggregate by per-class vote fraction.
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct RandomForestModel {
    config: RandomForestConfig,
    n_classes: usize,
    trees: Vec<DecisionTree<f64, usize>>,
}

fn to_linfa_features(arr: &Array2<f64>) -> ndl::Array2<f64> {
    let (rows, cols) = arr.dim();
    let data: Vec<f64> = arr.iter().copied().collect();
    ndl::Array2::from_shape_vec((rows, cols), data).expect("shape known correct from source array")
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
        anyhow::ensure!(config.n_trees > 0, "n_trees must be >= 1");

        let n_samples = features.nrows();
        let n_features = features.ncols();
        let n_classes = labels.iter().copied().max().map_or(0, |m| m + 1);

        let mut rng = StdRng::seed_from_u64(config.random_seed);

        let mut trees = Vec::with_capacity(config.n_trees);
        for tree_idx in 0..config.n_trees {
            // Bootstrap sample: n_samples row indices, with replacement.
            let indices: Vec<usize> = (0..n_samples)
                .map(|_| rng.gen_range(0..n_samples))
                .collect();

            // Materialise the bootstrap matrices in linfa's ndarray
            // namespace (0.16). Cheap relative to tree fitting.
            let mut feat_data = Vec::with_capacity(n_samples * n_features);
            let mut label_data = Vec::with_capacity(n_samples);
            for &i in &indices {
                for c in 0..n_features {
                    feat_data.push(features[(i, c)]);
                }
                label_data.push(labels[i]);
            }
            let feat = ndl::Array2::from_shape_vec((n_samples, n_features), feat_data)
                .expect("bootstrap shape correct by construction");
            let lab = ndl::Array1::from_vec(label_data);

            let dataset = Dataset::new(feat, lab);
            let tree = DecisionTree::<f64, usize>::params()
                .max_depth(config.max_depth)
                .min_weight_split(config.min_weight_split)
                .fit(&dataset)
                .with_context(|| format!("fitting bootstrap tree {tree_idx}"))?;
            trees.push(tree);
        }

        Ok(Self {
            config: config.clone(),
            n_classes,
            trees,
        })
    }

    fn n_classes(&self) -> usize {
        self.n_classes
    }

    fn predict(&self, features: &Array2<f64>) -> Result<Array1<usize>> {
        let proba = self.predict_proba(features)?;
        let n = proba.nrows();
        let mut out = Array1::<usize>::zeros(n);
        for row in 0..n {
            let mut best_class = 0_usize;
            let mut best_p = f64::NEG_INFINITY;
            for c in 0..self.n_classes {
                let p = proba[(row, c)];
                if p > best_p {
                    best_p = p;
                    best_class = c;
                }
            }
            out[row] = best_class;
        }
        Ok(out)
    }

    fn predict_proba(&self, features: &Array2<f64>) -> Result<Array2<f64>> {
        anyhow::ensure!(
            !self.trees.is_empty(),
            "predict_proba called on an unfit RandomForestModel"
        );

        let features_linfa = to_linfa_features(features);
        let n = features.nrows();
        let mut votes = Array2::<f64>::zeros((n, self.n_classes.max(1)));

        for tree in &self.trees {
            let preds: ndl::Array1<usize> = tree.predict(&features_linfa);
            for (row, &cls) in preds.iter().enumerate() {
                if cls < self.n_classes {
                    votes[(row, cls)] += 1.0;
                }
            }
        }

        let n_trees = self.trees.len() as f64;
        votes.mapv_inplace(|v| v / n_trees);
        Ok(votes)
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

    fn execution_identity(&self) -> ExecutionIdentity {
        ExecutionIdentity::non_native(
            env!("CARGO_PKG_NAME"),
            env!("CARGO_PKG_VERSION"),
            RF_BACKEND,
            ExecutionIdentity::runtime_config_from_typed(&self.config),
        )
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use ndarray::{Axis, array};

    /// Two linearly-separable Gaussian clusters in 2D. Class 0 around
    /// (-2, -2); class 1 around (+2, +2). A correctly-trained RF should
    /// classify the cluster centres without difficulty.
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
        let features = Array2::from_shape_vec((100, 2), feats).unwrap();
        let labels = Array1::from_vec(labs);
        (features, labels)
    }

    #[test]
    fn trains_and_separates_two_clusters() {
        let (features, labels) = two_cluster_dataset(7);
        let config = RandomForestConfig {
            n_trees: 25,
            max_depth: Some(6),
            min_weight_split: 2.0,
            random_seed: 11,
        };
        let model = RandomForestModel::train(&config, &features, &labels).unwrap();

        let preds = model.predict(&features).unwrap();
        let correct = preds
            .iter()
            .zip(labels.iter())
            .filter(|(p, y)| p == y)
            .count();
        // Linearly separable clusters → expect near-perfect on train.
        assert!(
            correct >= 95,
            "expected >= 95/100 correct on linearly separable train, got {correct}"
        );
    }

    #[test]
    fn predict_proba_rows_sum_to_one() {
        let (features, labels) = two_cluster_dataset(3);
        let model = RandomForestModel::train(
            &RandomForestConfig {
                n_trees: 10,
                ..RandomForestConfig::default()
            },
            &features,
            &labels,
        )
        .unwrap();

        let proba = model.predict_proba(&features).unwrap();
        for row in proba.axis_iter(Axis(0)) {
            let s: f64 = row.iter().sum();
            assert!((s - 1.0).abs() < 1e-9, "row sum {s} != 1.0");
        }
    }

    #[test]
    fn deterministic_under_fixed_seed() {
        let (features, labels) = two_cluster_dataset(42);
        let config = RandomForestConfig {
            n_trees: 5,
            max_depth: Some(4),
            min_weight_split: 2.0,
            random_seed: 1234,
        };
        let a = RandomForestModel::train(&config, &features, &labels).unwrap();
        let b = RandomForestModel::train(&config, &features, &labels).unwrap();

        let pa = a.predict_proba(&features).unwrap();
        let pb = b.predict_proba(&features).unwrap();
        assert_eq!(
            pa, pb,
            "identical config + data must yield identical predictions"
        );
    }

    #[test]
    fn save_load_roundtrip_preserves_predictions() {
        let (features, labels) = two_cluster_dataset(9);
        let model = RandomForestModel::train(
            &RandomForestConfig {
                n_trees: 5,
                ..RandomForestConfig::default()
            },
            &features,
            &labels,
        )
        .unwrap();

        let tmp = tempfile::NamedTempFile::new().unwrap();
        model.save(tmp.path()).unwrap();
        let loaded = RandomForestModel::load(tmp.path()).unwrap();

        let p1 = model.predict_proba(&features).unwrap();
        let p2 = loaded.predict_proba(&features).unwrap();
        assert_eq!(p1, p2, "save/load must preserve predictions bit-for-bit");
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

    #[test]
    fn train_rejects_zero_trees() {
        let (features, labels) = two_cluster_dataset(0);
        let err = RandomForestModel::train(
            &RandomForestConfig {
                n_trees: 0,
                ..RandomForestConfig::default()
            },
            &features,
            &labels,
        )
        .unwrap_err();
        assert!(err.to_string().contains("n_trees must be >= 1"));
    }

    #[test]
    fn n_classes_reflects_label_max_plus_one() {
        // Intent: n_classes drives the probability matrix shape;
        // off-by-one here breaks predict_proba's allocation.
        let (features, labels) = two_cluster_dataset(0);
        let model = RandomForestModel::train(
            &RandomForestConfig {
                n_trees: 3,
                ..RandomForestConfig::default()
            },
            &features,
            &labels,
        )
        .unwrap();
        assert_eq!(model.n_classes(), 2);
    }

    #[test]
    fn execution_identity_stamps_backend_and_runtime_config() {
        // Intent: ferrox / soter audit gates read execution_identity to
        // answer "which model+hyperparams emitted this row?". A missing
        // backend or empty runtime_config breaks that audit chain.
        let (features, labels) = two_cluster_dataset(1);
        let model = RandomForestModel::train(
            &RandomForestConfig {
                n_trees: 3,
                max_depth: Some(4),
                min_weight_split: 2.0,
                random_seed: 99,
            },
            &features,
            &labels,
        )
        .unwrap();
        let id = model.execution_identity();
        assert_eq!(id.backend, RF_BACKEND);
        assert!(
            id.runtime_config.contains("n_trees"),
            "runtime_config must serialize hyperparams; got {}",
            id.runtime_config
        );
        assert_eq!(id.producer.name, env!("CARGO_PKG_NAME"));
    }

    #[test]
    fn predict_proba_errors_on_unfit_model() {
        // Intent: defending against an unfit-model code path is cheap
        // insurance; without it a deserialization regression could
        // silently produce zero-vote rows.
        let (features, _) = two_cluster_dataset(0);
        let mut model = RandomForestModel::train(
            &RandomForestConfig {
                n_trees: 1,
                ..RandomForestConfig::default()
            },
            &features,
            &Array1::from_vec(vec![0; features.nrows()]),
        )
        .unwrap();
        // Forcibly clear the trees to simulate unfit / corrupted state.
        model.trees.clear();
        let err = model.predict_proba(&features).unwrap_err();
        assert!(err.to_string().contains("unfit"));
    }

    #[test]
    fn load_returns_err_on_missing_file() {
        let err = RandomForestModel::load(std::path::Path::new(
            "/tmp/this/path/definitely/does/not/exist/rf.bin",
        ))
        .unwrap_err();
        assert!(err.to_string().contains("reading"));
    }

    use proptest::prelude::*;

    proptest! {
        #[test]
        fn predict_proba_rows_sum_to_one_for_any_seed(seed in 0u64..40, n_trees in 1usize..15) {
            // Intent: probabilities not summing to 1 corrupts every
            // downstream rank-by-score consumer (e.g. top-k routing).
            let (features, labels) = two_cluster_dataset(seed);
            let model = RandomForestModel::train(
                &RandomForestConfig {
                    n_trees,
                    max_depth: Some(3),
                    min_weight_split: 2.0,
                    random_seed: seed,
                },
                &features,
                &labels,
            )
            .unwrap();
            let proba = model.predict_proba(&features).unwrap();
            for row in proba.axis_iter(Axis(0)) {
                let s: f64 = row.iter().sum();
                prop_assert!((s - 1.0).abs() < 1e-9, "row sum {} != 1.0", s);
                for &p in row {
                    prop_assert!((0.0..=1.0).contains(&p), "p={} out of [0,1]", p);
                }
            }
        }

        #[test]
        fn save_load_roundtrip_preserves_predictions_under_seed(seed in 0u64..20) {
            // Intent: an artifact must replay bit-for-bit. Any
            // serde-version drift that breaks this invariant breaks
            // every audit-replay use case.
            let (features, labels) = two_cluster_dataset(seed);
            let model = RandomForestModel::train(
                &RandomForestConfig {
                    n_trees: 3,
                    ..RandomForestConfig::default()
                },
                &features,
                &labels,
            )
            .unwrap();
            let tmp = tempfile::NamedTempFile::new().unwrap();
            model.save(tmp.path()).unwrap();
            let loaded = RandomForestModel::load(tmp.path()).unwrap();
            let p1 = model.predict_proba(&features).unwrap();
            let p2 = loaded.predict_proba(&features).unwrap();
            prop_assert_eq!(p1, p2);
        }
    }
}
