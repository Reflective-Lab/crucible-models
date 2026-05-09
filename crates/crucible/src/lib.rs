// Copyright 2026 Reflective Labs
// SPDX-License-Identifier: MIT

//! # crucible-models
//!
//! Trained-model packs for the Converge Engine.
//!
//! Unlike `prism-analytics` (pure inference, no training pipeline),
//! every model here is learned from data using Burn as the training
//! framework.  Parameters are stored as trained artifacts, not
//! hand-authored rules.
//!
//! ## Planned packs
//!
//! - `trees` — Decision Tree classifier (CART, Gini / information gain)
//! - `ensembles` — Random Forest and gradient-boosted trees (XGBoost-style)
//! - `svm` — Support Vector Machine with kernel functions
//! - `neuro_fuzzy` — ANFIS (Adaptive Neuro-Fuzzy Inference System) via Burn

pub mod ensembles;
pub mod neuro_fuzzy;
pub mod svm;
pub mod trees;
