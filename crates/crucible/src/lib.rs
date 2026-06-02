// Copyright 2026 Reflective Labs
// SPDX-License-Identifier: MIT

//! # crucible-models
//!
//! Trained-model packs and the training pipeline for the Converge Engine.
//!
//! Unlike `prism-analytics` (closed-form, hand-authored inference), Crucible
//! models are learned from data using the training backend that fits the
//! model family: Burn for differentiable packs and linfa for classical
//! models such as trees and forests. Parameters are stored as trained
//! artifacts, not hand-authored rules.
//!
//! ## Training pipeline
//!
//! The training pipeline lives in [`training`] and the supporting data
//! plumbing lives in [`ingest`]. The pipeline is composed of
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
pub mod suggestor;
pub mod svm;
pub mod training;
pub mod trees;
pub mod types;

pub use ensembles::{RandomForestConfig, RandomForestModel};
pub use model::ClassifierModel;
pub use provenance::{CRUCIBLE_PROVENANCE, Crucible};
pub use suggestor::ClassifierSuggestor;
pub use training::{
    DataValidationAgent, DatasetAgent, DeploymentAgent, FeatureEngineeringAgent,
    HyperparameterSearchAgent, ModelEvaluationAgent, ModelRegistryAgent, ModelTrainingAgent,
    MonitoringAgent, SampleInferenceAgent,
};
pub use trees::{DecisionTreeClassifier, DecisionTreeConfig};
pub use types::{ClassPredictionPayload, ClassificationFeaturesPayload};

/// Inference Suggestor for the [`DecisionTreeClassifier`] pack.
pub type DecisionTreeClassifierSuggestor = ClassifierSuggestor<DecisionTreeClassifier>;

/// Inference Suggestor for the [`RandomForestModel`] pack.
pub type RandomForestClassifierSuggestor = ClassifierSuggestor<RandomForestModel>;
