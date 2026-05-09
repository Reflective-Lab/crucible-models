//! Decision Tree classifier (CART — Classification and Regression Trees).
//!
//! ## Algorithm sketch
//!
//! A CART tree is built by recursively splitting the training set on the
//! feature/threshold pair that maximises information gain (classification)
//! or minimises MSE (regression).  Splitting stops when a node is pure,
//! reaches `max_depth`, or contains fewer than `min_samples_split` points.
//!
//! ## Training inputs (planned)
//!
//! ```json
//! {
//!   "records": [[f64, ...]],   // n×d feature matrix
//!   "labels":  [usize, ...],   // class index per record
//!   "max_depth": 10,
//!   "min_samples_split": 2,
//!   "criterion": "gini"        // "gini" | "entropy"
//! }
//! ```
//!
//! ## Inference inputs (planned)
//!
//! ```json
//! {
//!   "records": [[f64, ...]],   // m×d feature matrix
//!   "model_artifact": "<base64-serialized tree>"
//! }
//! ```
//!
//! ## Why Burn?
//!
//! Burn provides a portable tensor backend (ndarray for CPU, WGPU for GPU)
//! and a save/load mechanism for trained model artifacts.  A Decision Tree
//! can be encoded as a Burn `Record` so the same serialisation pipeline
//! used for neural networks carries the tree to disk or an artifact
//! registry.
//!
//! ## Status: not yet implemented
