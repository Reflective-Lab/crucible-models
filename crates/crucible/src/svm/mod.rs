//! Support Vector Machine (SVM) with kernel functions.
//!
//! ## Algorithm sketch
//!
//! An SVM finds the maximum-margin hyperplane that separates two classes.
//! Non-linearly-separable data is handled by mapping features into a
//! higher-dimensional space via a kernel function `K(x, x')`:
//!
//! | Kernel        | Formula                            | Use case               |
//! |---------------|------------------------------------|------------------------|
//! | Linear        | `x · x'`                           | linearly separable     |
//! | RBF (Gaussian)| `exp(-γ‖x-x'‖²)`                  | general purpose        |
//! | Polynomial    | `(γ x·x' + r)^d`                  | image / text           |
//! | Sigmoid       | `tanh(γ x·x' + r)`                | neural net analogue    |
//!
//! Training minimises the dual QP via Sequential Minimal Optimisation (SMO)
//! or Burn-based subgradient descent.  Soft-margin penalty `C` controls
//! the bias-variance trade-off.
//!
//! ## Training inputs (planned)
//!
//! ```json
//! {
//!   "records": [[f64, ...]],
//!   "labels":  [i32, ...],       // +1 / -1 for binary; multi-class via OvR
//!   "kernel":  "rbf",
//!   "C": 1.0,
//!   "gamma": 0.1
//! }
//! ```
//!
//! ## Status: not yet implemented
