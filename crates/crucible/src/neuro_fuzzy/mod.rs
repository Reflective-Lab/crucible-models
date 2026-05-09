//! ANFIS — Adaptive Neuro-Fuzzy Inference System.
//!
//! ## What is ANFIS?
//!
//! ANFIS is a Sugeno-type fuzzy inference system whose Gaussian membership
//! function parameters (means and widths) are learned from data via
//! backpropagation.  The architecture has five layers:
//!
//! 1. **Fuzzification** — Gaussian MF evaluation (learned μ, σ)
//! 2. **Rule firing** — product t-norm across antecedent memberships
//! 3. **Normalisation** — firing strength / sum of all firing strengths
//! 4. **Consequent** — linear Sugeno output (learned coefficients)
//! 5. **Defuzzification** — weighted sum (crisp output)
//!
//! Layers 1 and 4 hold learnable parameters; Burn's autodiff backend
//! computes the gradient of the output MSE with respect to them.
//!
//! ## Difference from prism-analytics
//!
//! `prism-analytics` ships Mamdani, Sugeno, and Tsukamoto FIS with
//! **expert-authored** rules — no training pipeline.  ANFIS is Sugeno +
//! backprop on MF parameters, which requires a loss function and an
//! optimiser.  That training pipeline belongs here, in `crucible-models`.
//!
//! See `kb/Architecture/Project Boundary.md` for the full rationale.
//!
//! ## Training inputs (planned)
//!
//! ```json
//! {
//!   "records":       [[f64, ...]],   // n×d input matrix
//!   "targets":       [f64, ...],     // n regression targets
//!   "n_rules":       5,              // number of fuzzy rules
//!   "epochs":        100,
//!   "learning_rate": 0.01
//! }
//! ```
//!
//! ## Status: not yet implemented
