//! Decision Tree classifier (CART — Classification and Regression Trees).
//!
//! Backed by `linfa_trees::DecisionTree<f64, usize>`. See
//! [`crate::ensembles`] for the bagging ensemble (Random Forest) built
//! on top of this primitive.
//!
//! ## Algorithm sketch
//!
//! A CART tree is built by recursively splitting the training set on
//! the feature/threshold pair that maximises information gain
//! (classification) or minimises MSE (regression). Splitting stops
//! when a node is pure, reaches `max_depth`, or contains fewer than
//! the configured weight-split threshold.
//!
//! ## Status
//!
//! - `decision_tree` — shipped. Implements [`crate::model::ClassifierModel`]
//!   via linfa-trees. Used directly through
//!   [`crate::DecisionTreeClassifierSuggestor`] for interpretable
//!   single-tree inference, and indirectly by
//!   [`crate::ensembles::RandomForestModel`] as the unit element of a
//!   bagging ensemble.

pub mod decision_tree;

pub use decision_tree::{DecisionTreeClassifier, DecisionTreeConfig};
