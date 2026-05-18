// Copyright 2024-2026 Reflective Labs

use anyhow::{Result, anyhow};
use converge_pack::{
    Context, ContextKey, DiagnosticPayload, FactPayload, ProposalId, ProposedFact,
    ProvenanceSource,
};
use serde::{Deserialize, Serialize};
use std::collections::HashMap;
use std::path::PathBuf;

use crate::provenance::CRUCIBLE_PROVENANCE;

pub(super) fn proposal(
    _agent_name: &str,
    key: ContextKey,
    id: impl Into<String>,
    payload: impl FactPayload + PartialEq,
) -> ProposedFact {
    CRUCIBLE_PROVENANCE.proposed_fact(key, ProposalId::new(id.into()), payload)
}

pub(super) fn diagnostic(
    agent_name: &str,
    key: ContextKey,
    id: impl Into<String>,
    message: impl Into<String>,
) -> ProposedFact {
    proposal(
        agent_name,
        key,
        id,
        DiagnosticPayload::new(agent_name, message.into()),
    )
}

macro_rules! impl_fact_payload {
    ($ty:ty, $family:literal) => {
        impl FactPayload for $ty {
            const FAMILY: &'static str = $family;
            const VERSION: u16 = 1;
        }
    };
}

#[derive(Debug, Serialize, Deserialize, Clone, PartialEq)]
#[serde(deny_unknown_fields)]
pub struct TrainingPlan {
    pub iteration: usize,
    pub max_rows: usize,
    pub train_fraction: f64,
    pub val_fraction: f64,
    pub infer_fraction: f64,
    pub quality_threshold: f64,
}

#[derive(Debug, Serialize, Deserialize, Clone, PartialEq)]
#[serde(deny_unknown_fields)]
pub struct DatasetSplit {
    pub source_path: PathBuf,
    pub train_path: PathBuf,
    pub val_path: PathBuf,
    pub infer_path: PathBuf,
    pub total_rows: usize,
    pub max_rows: usize,
    pub train_rows: usize,
    pub val_rows: usize,
    pub infer_rows: usize,
    pub iteration: usize,
}

#[derive(Debug, Serialize, Deserialize, Clone, PartialEq)]
#[serde(deny_unknown_fields)]
pub struct BaselineModel {
    pub target_column: String,
    pub mean: f64,
}

#[derive(Debug, Serialize, Deserialize, Clone, PartialEq)]
#[serde(deny_unknown_fields)]
pub struct ModelMetadata {
    pub model_path: PathBuf,
    pub target_column: String,
    pub train_rows: usize,
    pub baseline_mean: f64,
    pub iteration: usize,
}

#[derive(Debug, Serialize, Deserialize, Clone, PartialEq)]
#[serde(deny_unknown_fields)]
pub struct EvaluationReport {
    pub model_path: PathBuf,
    pub metric: String,
    pub value: f64,
    pub mean_abs_target: f64,
    pub success_ratio: f64,
    pub val_rows: usize,
    pub iteration: usize,
}

#[derive(Debug, Serialize, Deserialize, Clone, PartialEq)]
#[serde(deny_unknown_fields)]
pub struct InferenceSample {
    pub model_path: PathBuf,
    pub target_column: String,
    pub rows: usize,
    pub predictions: Vec<f64>,
    pub actuals: Vec<f64>,
    pub iteration: usize,
}

#[derive(Debug, Serialize, Deserialize, Clone, PartialEq)]
#[serde(deny_unknown_fields)]
pub struct DataQualityReport {
    pub kind: String,
    pub iteration: usize,
    pub source_path: PathBuf,
    pub rows_checked: usize,
    pub missingness: HashMap<String, f64>,
    pub numeric_means: HashMap<String, f64>,
    pub outlier_counts: HashMap<String, usize>,
    pub drift_score: Option<f64>,
}

#[derive(Debug, Serialize, Deserialize, Clone, PartialEq, Eq)]
#[serde(deny_unknown_fields)]
pub struct FeatureInteraction {
    pub name: String,
    pub left: String,
    pub right: String,
    pub op: String,
}

#[derive(Debug, Serialize, Deserialize, Clone, PartialEq, Eq)]
#[serde(deny_unknown_fields)]
pub struct FeatureSpec {
    pub kind: String,
    pub iteration: usize,
    pub target_column: String,
    pub numeric_features: Vec<String>,
    pub categorical_features: Vec<String>,
    pub normalization: String,
    pub interactions: Vec<FeatureInteraction>,
}

#[derive(Debug, Serialize, Deserialize, Clone, PartialEq)]
#[serde(deny_unknown_fields)]
pub struct HyperparameterSearchPlan {
    pub kind: String,
    pub iteration: usize,
    pub max_trials: usize,
    pub early_stopping: bool,
    pub params: HashMap<String, Vec<f64>>,
}

#[derive(Debug, Serialize, Deserialize, Clone, PartialEq)]
#[serde(deny_unknown_fields)]
pub struct HyperparameterSearchResult {
    pub kind: String,
    pub iteration: usize,
    pub best_params: HashMap<String, f64>,
    pub score: f64,
}

#[derive(Debug, Serialize, Deserialize, Clone, PartialEq)]
#[serde(deny_unknown_fields)]
pub struct ModelRegistryRecord {
    pub kind: String,
    pub iteration: usize,
    pub model_path: PathBuf,
    pub metrics: HashMap<String, f64>,
    pub provenance: String,
}

#[derive(Debug, Serialize, Deserialize, Clone, PartialEq)]
#[serde(deny_unknown_fields)]
pub struct MonitoringReport {
    pub kind: String,
    pub iteration: usize,
    pub metric: String,
    pub value: f64,
    pub baseline: f64,
    pub status: String,
}

#[derive(Debug, Serialize, Deserialize, Clone, PartialEq, Eq)]
#[serde(deny_unknown_fields)]
pub struct DeploymentDecision {
    pub kind: String,
    pub iteration: usize,
    pub action: String,
    pub reason: String,
    pub retrain: bool,
}

impl_fact_payload!(TrainingPlan, "crucible.training.plan");
impl_fact_payload!(DatasetSplit, "crucible.dataset.split");
impl_fact_payload!(BaselineModel, "crucible.baseline.model");
impl_fact_payload!(ModelMetadata, "crucible.model.metadata");
impl_fact_payload!(EvaluationReport, "crucible.evaluation.report");
impl_fact_payload!(InferenceSample, "crucible.inference.sample");
impl_fact_payload!(DataQualityReport, "crucible.data_quality.report");
impl_fact_payload!(FeatureSpec, "crucible.feature.spec");
impl_fact_payload!(
    HyperparameterSearchPlan,
    "crucible.hyperparameter_search.plan"
);
impl_fact_payload!(
    HyperparameterSearchResult,
    "crucible.hyperparameter_search.result"
);
impl_fact_payload!(ModelRegistryRecord, "crucible.model.registry_record");
impl_fact_payload!(MonitoringReport, "crucible.monitoring.report");
impl_fact_payload!(DeploymentDecision, "crucible.deployment.decision");

// ── Context read helpers ──────────────────────────────────────────────────────

pub(super) fn read_latest_split_from_ctx(ctx: &dyn Context) -> Result<DatasetSplit> {
    let facts = ctx.get(ContextKey::Signals);
    let mut latest: Option<DatasetSplit> = None;
    for fact in facts {
        if let Some(split) = fact.payload::<DatasetSplit>().cloned() {
            let should_replace = match &latest {
                Some(current) => split.iteration > current.iteration,
                None => true,
            };
            if should_replace {
                latest = Some(split);
            }
        }
    }
    latest.ok_or_else(|| anyhow!("missing dataset split"))
}

pub(super) fn read_model_path_from_ctx(ctx: &dyn Context) -> Result<PathBuf> {
    let meta = read_latest_model_meta_from_ctx(ctx)?;
    Ok(meta.model_path)
}

pub(super) fn read_model_from_ctx(ctx: &dyn Context) -> Result<BaselineModel> {
    let model_path = read_model_path_from_ctx(ctx)?;
    let content = std::fs::read_to_string(model_path)?;
    let model = serde_json::from_str(&content)?;
    Ok(model)
}

pub(super) fn read_latest_model_meta_from_ctx(ctx: &dyn Context) -> Result<ModelMetadata> {
    let facts = ctx.get(ContextKey::Strategies);
    let mut latest: Option<ModelMetadata> = None;
    for fact in facts {
        if let Some(meta) = fact.payload::<ModelMetadata>().cloned() {
            let should_replace = match &latest {
                Some(current) => meta.iteration > current.iteration,
                None => true,
            };
            if should_replace {
                latest = Some(meta);
            }
        }
    }
    latest.ok_or_else(|| anyhow!("missing model metadata"))
}

pub(super) fn read_latest_plan_from_ctx(ctx: &dyn Context) -> Option<TrainingPlan> {
    let facts = ctx.get(ContextKey::Constraints);
    let mut latest: Option<TrainingPlan> = None;
    for fact in facts {
        if let Some(plan) = fact.payload::<TrainingPlan>().cloned() {
            let should_replace = match &latest {
                Some(current) => plan.iteration > current.iteration,
                None => true,
            };
            if should_replace {
                latest = Some(plan);
            }
        }
    }
    latest
}

pub(super) fn read_feature_spec_from_ctx(ctx: &dyn Context, iteration: usize) -> Option<FeatureSpec> {
    ctx.get(ContextKey::Constraints).iter().find_map(|fact| {
        fact.payload::<FeatureSpec>()
            .filter(|spec| spec.iteration == iteration)
            .cloned()
    })
}

pub(super) fn has_split_for_iteration(ctx: &dyn Context, iteration: usize) -> bool {
    ctx.get(ContextKey::Signals).iter().any(|fact| {
        fact.payload::<DatasetSplit>()
            .is_some_and(|split| split.iteration == iteration)
    })
}

pub(super) fn has_model_for_iteration(ctx: &dyn Context, iteration: usize) -> bool {
    ctx.get(ContextKey::Strategies).iter().any(|fact| {
        fact.payload::<ModelMetadata>()
            .is_some_and(|meta| meta.iteration == iteration)
    })
}

pub(super) fn has_evaluation_for_iteration(ctx: &dyn Context, iteration: usize) -> bool {
    ctx.get(ContextKey::Evaluations).iter().any(|fact| {
        fact.payload::<EvaluationReport>()
            .is_some_and(|report| report.iteration == iteration)
    })
}

pub(super) fn has_inference_for_iteration(ctx: &dyn Context, iteration: usize) -> bool {
    ctx.get(ContextKey::Hypotheses).iter().any(|fact| {
        fact.payload::<InferenceSample>()
            .is_some_and(|sample| sample.iteration == iteration)
    })
}

pub(super) fn has_data_quality_for_iteration(ctx: &dyn Context, iteration: usize) -> bool {
    ctx.get(ContextKey::Signals).iter().any(|fact| {
        fact.payload::<DataQualityReport>()
            .is_some_and(|report| report.iteration == iteration)
    })
}

pub(super) fn has_feature_spec_for_iteration(ctx: &dyn Context, iteration: usize) -> bool {
    ctx.get(ContextKey::Constraints).iter().any(|fact| {
        fact.payload::<FeatureSpec>()
            .is_some_and(|spec| spec.iteration == iteration)
    })
}

pub(super) fn has_hyperparam_result_for_iteration(ctx: &dyn Context, iteration: usize) -> bool {
    ctx.get(ContextKey::Evaluations).iter().any(|fact| {
        fact.payload::<HyperparameterSearchResult>()
            .is_some_and(|result| result.iteration == iteration)
    })
}

pub(super) fn has_registry_record_for_iteration(ctx: &dyn Context, iteration: usize) -> bool {
    ctx.get(ContextKey::Strategies).iter().any(|fact| {
        fact.payload::<ModelRegistryRecord>()
            .is_some_and(|record| record.iteration == iteration)
    })
}

pub(super) fn has_monitoring_report_for_iteration(ctx: &dyn Context, iteration: usize) -> bool {
    ctx.get(ContextKey::Evaluations).iter().any(|fact| {
        fact.payload::<MonitoringReport>()
            .is_some_and(|report| report.iteration == iteration)
    })
}

pub(super) fn has_deployment_decision_for_iteration(ctx: &dyn Context, iteration: usize) -> bool {
    ctx.get(ContextKey::Strategies).iter().any(|fact| {
        fact.payload::<DeploymentDecision>()
            .is_some_and(|decision| decision.iteration == iteration)
    })
}

pub(super) fn latest_evaluation_report(ctx: &dyn Context, iteration: usize) -> Option<EvaluationReport> {
    let mut latest: Option<EvaluationReport> = None;
    for fact in ctx.get(ContextKey::Evaluations) {
        if let Some(report) = fact.payload::<EvaluationReport>().cloned() {
            if iteration > 0 {
                if report.iteration == iteration {
                    return Some(report);
                }
            } else if latest
                .as_ref()
                .is_none_or(|current| report.iteration > current.iteration)
            {
                latest = Some(report);
            }
        }
    }
    if iteration > 0 { None } else { latest }
}

pub(super) fn latest_data_quality_before_iteration(
    ctx: &dyn Context,
    iteration: usize,
) -> Option<DataQualityReport> {
    let mut latest: Option<DataQualityReport> = None;
    for fact in ctx.get(ContextKey::Signals) {
        if let Some(report) = fact.payload::<DataQualityReport>().cloned()
            && report.iteration < iteration
            && latest
                .as_ref()
                .is_none_or(|current| report.iteration > current.iteration)
        {
            latest = Some(report);
        }
    }
    latest
}

pub(super) fn drift_score_from_ctx(
    ctx: &dyn Context,
    iteration: usize,
    numeric_means: &HashMap<String, f64>,
) -> Option<f64> {
    let previous = latest_data_quality_before_iteration(ctx, iteration)?;
    let mut total_delta = 0.0;
    let mut count = 0usize;
    for (name, mean) in numeric_means {
        if let Some(prev_mean) = previous.numeric_means.get(name) {
            total_delta += (mean - prev_mean).abs();
            count += 1;
        }
    }
    if count == 0 {
        None
    } else {
        Some(total_delta / count as f64)
    }
}
