//! Train a `RandomForestModel` on synthetic loan-default data.
//!
//! Usage:
//!
//! ```sh
//! cargo run --release --bin train_loan_default
//! ```
//!
//! The CLI generates a deterministic synthetic loan-default dataset
//! (1,000 samples, 6 features), splits 80/20 into train and validation,
//! fits a Random Forest, prints validation accuracy plus the per-class
//! confusion-matrix counts, and writes the artifact to
//! `crucible-models/artifacts/`.
//!
//! This is the first CLI entry point into crucible's training pipeline.
//! It deliberately bypasses the full Suggestor / Convergence-Engine
//! protocol — the goal here is reproducibility from a CLI, not a
//! Formation. The Suggestor-driven path lifts later when a real
//! retrain trigger pulls.

use std::{fs, path::PathBuf};

use anyhow::{Context as _, Result};
use crucible::{ClassifierModel, RandomForestConfig, RandomForestModel, fixtures::loan_default};
use ndarray::{Array1, Array2, Axis};

const TOTAL_SAMPLES: usize = 1_000;
const TRAIN_FRACTION: f64 = 0.8;
const DATA_SEED: u64 = 20_260_515;
const TRAIN_SEED: u64 = 11;

fn split_train_val(
    features: &Array2<f64>,
    labels: &Array1<usize>,
    n_train: usize,
) -> (Array2<f64>, Array1<usize>, Array2<f64>, Array1<usize>) {
    let train_idx: Vec<usize> = (0..n_train).collect();
    let val_idx: Vec<usize> = (n_train..features.nrows()).collect();
    let train_x = features.select(Axis(0), &train_idx);
    let train_y = labels.select(Axis(0), &train_idx);
    let val_x = features.select(Axis(0), &val_idx);
    let val_y = labels.select(Axis(0), &val_idx);
    (train_x, train_y, val_x, val_y)
}

fn accuracy(preds: &Array1<usize>, truth: &Array1<usize>) -> f64 {
    let correct = preds
        .iter()
        .zip(truth.iter())
        .filter(|(p, y)| p == y)
        .count();
    correct as f64 / preds.len().max(1) as f64
}

fn main() -> Result<()> {
    println!("=== loan_default RandomForest trainer ===");

    let n_train = (TOTAL_SAMPLES as f64 * TRAIN_FRACTION) as usize;

    let (features, labels) = loan_default(TOTAL_SAMPLES, DATA_SEED);
    println!(
        "Generated {} samples × {} features (seed {DATA_SEED})",
        features.nrows(),
        features.ncols()
    );

    let (train_x, train_y, val_x, val_y) = split_train_val(&features, &labels, n_train);
    println!("Train: {}  Val: {}", train_y.len(), val_y.len());

    let config = RandomForestConfig {
        n_trees: 50,
        max_depth: Some(8),
        min_weight_split: 2.0,
        random_seed: TRAIN_SEED,
    };
    println!(
        "Training RandomForest: n_trees={} max_depth={:?} seed={}",
        config.n_trees, config.max_depth, config.random_seed
    );

    let model = RandomForestModel::train(&config, &train_x, &train_y)
        .context("training RandomForestModel")?;

    let val_preds = model.predict(&val_x).context("predicting on val set")?;
    let acc = accuracy(&val_preds, &val_y);

    let n_pos_true = val_y.iter().filter(|&&y| y == 1).count();
    let n_pos_pred = val_preds.iter().filter(|&&y| y == 1).count();
    let tp = val_preds
        .iter()
        .zip(val_y.iter())
        .filter(|(p, y)| **p == 1 && **y == 1)
        .count();
    let tn = val_preds
        .iter()
        .zip(val_y.iter())
        .filter(|(p, y)| **p == 0 && **y == 0)
        .count();
    let fp = n_pos_pred - tp;
    let fn_ = n_pos_true - tp;

    println!("--- validation ---");
    println!("accuracy:               {acc:.4}");
    println!("true positives:         {tp}");
    println!("true negatives:         {tn}");
    println!("false positives:        {fp}");
    println!("false negatives:        {fn_}");
    println!(
        "class-1 (default) rate: {:.3}",
        n_pos_true as f64 / val_y.len() as f64
    );

    let artifacts_dir = PathBuf::from(env!("CARGO_MANIFEST_DIR")).join("../../artifacts");
    fs::create_dir_all(&artifacts_dir).context("creating artifacts directory")?;

    let artifact_path = artifacts_dir.join(format!(
        "loan_default_seed{}_n{}.bin",
        config.random_seed, config.n_trees
    ));
    model
        .save(&artifact_path)
        .context("saving RandomForest artifact")?;
    println!("artifact: {}", artifact_path.display());

    println!("=== done ===");
    Ok(())
}
