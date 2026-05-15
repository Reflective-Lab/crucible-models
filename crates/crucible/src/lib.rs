// Copyright 2026 Reflective Labs
// SPDX-License-Identifier: MIT

//! # crucible-models
//!
//! Trained-model packs and the training pipeline for the Converge Engine.
//!
//! Unlike `prism-analytics` (closed-form, hand-authored inference), every
//! model under `crucible-models` is learned from data using Burn as the
//! training framework. Parameters are stored as trained artifacts, not
//! hand-authored rules.
//!
//! ## Training pipeline
//!
//! The training pipeline lives in [`training`] and the supporting data
//! plumbing in [`ingest`] and [`storage`]. The pipeline is composed of
//! Suggestor-shaped agents (`DatasetAgent`, `DataValidationAgent`,
//! `FeatureEngineeringAgent`, `HyperparameterSearchAgent`,
//! `ModelTrainingAgent`, `ModelEvaluationAgent`, `ModelRegistryAgent`,
//! `MonitoringAgent`, `DeploymentAgent`, `SampleInferenceAgent`) that
//! today run from a binary entrypoint and can be lifted into a
//! Formation when a real retrain trigger pulls.
//!
//! ## Planned packs
//!
//! - `trees` — Decision Tree classifier (CART, Gini / information gain)
//! - `ensembles` — Random Forest and gradient-boosted trees (XGBoost-style)
//! - `svm` — Support Vector Machine with kernel functions
//! - `neuro_fuzzy` — ANFIS (Adaptive Neuro-Fuzzy Inference System) via Burn

pub mod ensembles;
pub mod fixtures;
pub mod ingest;
pub mod model;
pub mod neuro_fuzzy;
pub mod provenance;
#[cfg(feature = "storage")]
pub mod storage;
pub mod svm;
pub mod training;
pub mod trees;

pub use ensembles::{RandomForestConfig, RandomForestModel};
pub use model::ClassifierModel;
pub use provenance::{CRUCIBLE_PROVENANCE, ProvenanceSource, UnknownProvenanceSource};
pub use training::{
    DataValidationAgent, DatasetAgent, DeploymentAgent, FeatureEngineeringAgent,
    HyperparameterSearchAgent, ModelEvaluationAgent, ModelRegistryAgent, ModelTrainingAgent,
    MonitoringAgent, SampleInferenceAgent,
};
