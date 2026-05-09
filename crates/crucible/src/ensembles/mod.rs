//! Ensemble models: Random Forest and gradient-boosted trees.
//!
//! ## Random Forest
//!
//! Bagging of `n_estimators` CART trees, each trained on a bootstrap
//! sample with a random feature subset (`max_features` per split).
//! Prediction is majority vote (classification) or mean (regression).
//!
//! ## Gradient-Boosted Trees (XGBoost-style)
//!
//! Sequential weak learner training: each tree fits the residuals of the
//! previous ensemble with a learning rate `η`.  The objective can be
//! cross-entropy (classification) or MSE (regression).  Supports
//! regularisation via `lambda` (L2 on leaf weights) and `gamma`
//! (minimum gain to split).
//!
//! ## Training inputs (planned)
//!
//! ```json
//! {
//!   "records": [[f64, ...]],
//!   "labels":  [usize, ...],
//!   "n_estimators": 100,
//!   "max_depth": 6,
//!   "learning_rate": 0.1,
//!   "subsample": 0.8
//! }
//! ```
//!
//! ## Why Burn?
//!
//! Random Forest trees are independent and trivially data-parallel — Burn's
//! tensor primitives can broadcast feature comparisons across the batch
//! dimension.  Gradient boosting uses differentiable loss functions; Burn's
//! autodiff backend makes the gradient computation explicit and auditable.
//!
//! ## Status: not yet implemented
