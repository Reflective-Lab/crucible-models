// Copyright 2024-2026 Reflective Labs

use converge_pack::{AgentEffect, Context, ContextKey, Provenance, ProvenanceSource, Suggestor};
use std::collections::HashMap;

use crate::provenance::CRUCIBLE_PROVENANCE;

use super::features::apply_feature_spec;
use super::io::{get_numeric_series, load_dataframe, mean_abs_error, mean_abs_value};
use super::types::{
    DeploymentDecision, EvaluationReport, InferenceSample, ModelRegistryRecord, MonitoringReport,
    diagnostic, has_deployment_decision_for_iteration, has_evaluation_for_iteration,
    has_inference_for_iteration, has_monitoring_report_for_iteration,
    has_registry_record_for_iteration, latest_evaluation_report, proposal,
    read_feature_spec_from_ctx, read_latest_model_meta_from_ctx, read_latest_plan_from_ctx,
    read_latest_split_from_ctx, read_model_from_ctx, read_model_path_from_ctx,
};

#[derive(Debug, Default)]
pub struct ModelEvaluationAgent;

impl ModelEvaluationAgent {
    pub fn new() -> Self {
        Self
    }
}

#[async_trait::async_trait]
impl Suggestor for ModelEvaluationAgent {
    fn name(&self) -> &'static str {
        "ModelEvaluationAgent (MAE)"
    }

    fn dependencies(&self) -> &[ContextKey] {
        &[ContextKey::Signals, ContextKey::Strategies]
    }

    fn accepts(&self, ctx: &dyn Context) -> bool {
        ctx.has(ContextKey::Signals)
            && ctx.has(ContextKey::Strategies)
            && match read_latest_split_from_ctx(ctx) {
                Ok(split) => !has_evaluation_for_iteration(ctx, split.iteration),
                Err(_) => false,
            }
    }

    fn provenance(&self) -> Provenance {
        Provenance::from(CRUCIBLE_PROVENANCE.as_str())
    }

    async fn execute(&self, ctx: &dyn Context) -> AgentEffect {
        let split = match read_latest_split_from_ctx(ctx) {
            Ok(split) => split,
            Err(err) => {
                return AgentEffect::with_proposal(diagnostic(
                    self.name(),
                    ContextKey::Diagnostic,
                    "model-eval-error",
                    err.to_string(),
                ));
            }
        };

        let model = match read_model_from_ctx(ctx) {
            Ok(model) => model,
            Err(err) => {
                return AgentEffect::with_proposal(diagnostic(
                    self.name(),
                    ContextKey::Diagnostic,
                    "model-eval-error",
                    err.to_string(),
                ));
            }
        };

        let raw_val_df = match load_dataframe(&split.val_path) {
            Ok(df) => df,
            Err(err) => {
                return AgentEffect::with_proposal(diagnostic(
                    self.name(),
                    ContextKey::Diagnostic,
                    "model-eval-error",
                    err.to_string(),
                ));
            }
        };

        // Apply FeatureSpec transformation if available
        let val_df = match read_feature_spec_from_ctx(ctx, split.iteration) {
            Some(spec) => apply_feature_spec(&raw_val_df, &spec).unwrap_or(raw_val_df),
            None => raw_val_df,
        };

        let target = match get_numeric_series(&val_df, &model.target_column) {
            Ok(series) => series,
            Err(err) => {
                return AgentEffect::with_proposal(diagnostic(
                    self.name(),
                    ContextKey::Diagnostic,
                    "model-eval-error",
                    err.to_string(),
                ));
            }
        };

        let mae = match mean_abs_error(&target, model.mean) {
            Ok(value) => value,
            Err(err) => {
                return AgentEffect::with_proposal(diagnostic(
                    self.name(),
                    ContextKey::Diagnostic,
                    "model-eval-error",
                    err.to_string(),
                ));
            }
        };

        let mean_abs = match mean_abs_value(&target) {
            Ok(value) => value,
            Err(err) => {
                return AgentEffect::with_proposal(diagnostic(
                    self.name(),
                    ContextKey::Diagnostic,
                    "model-eval-error",
                    err.to_string(),
                ));
            }
        };

        let success_ratio = if mean_abs > 0.0 {
            (1.0 - (mae / mean_abs)).clamp(0.0, 1.0)
        } else {
            0.0
        };

        let report = EvaluationReport {
            model_path: read_model_path_from_ctx(ctx).unwrap_or_default(),
            metric: "mae".to_string(),
            value: mae,
            mean_abs_target: mean_abs,
            success_ratio,
            val_rows: split.val_rows,
            iteration: split.iteration,
        };

        AgentEffect::with_proposal(proposal(
            self.name(),
            ContextKey::Evaluations,
            format!("model-eval-{}", split.iteration),
            report,
        ))
    }
}

#[derive(Debug)]
pub struct SampleInferenceAgent {
    pub max_rows: usize,
}

impl SampleInferenceAgent {
    pub fn new(max_rows: usize) -> Self {
        Self { max_rows }
    }
}

#[async_trait::async_trait]
impl Suggestor for SampleInferenceAgent {
    fn name(&self) -> &'static str {
        "SampleInferenceAgent (Baseline)"
    }

    fn dependencies(&self) -> &[ContextKey] {
        &[ContextKey::Signals, ContextKey::Strategies]
    }

    fn accepts(&self, ctx: &dyn Context) -> bool {
        ctx.has(ContextKey::Signals)
            && ctx.has(ContextKey::Strategies)
            && match read_latest_split_from_ctx(ctx) {
                Ok(split) => !has_inference_for_iteration(ctx, split.iteration),
                Err(_) => false,
            }
    }

    fn provenance(&self) -> Provenance {
        Provenance::from(CRUCIBLE_PROVENANCE.as_str())
    }

    async fn execute(&self, ctx: &dyn Context) -> AgentEffect {
        let split = match read_latest_split_from_ctx(ctx) {
            Ok(split) => split,
            Err(err) => {
                return AgentEffect::with_proposal(diagnostic(
                    self.name(),
                    ContextKey::Diagnostic,
                    "model-infer-error",
                    err.to_string(),
                ));
            }
        };

        let model = match read_model_from_ctx(ctx) {
            Ok(model) => model,
            Err(err) => {
                return AgentEffect::with_proposal(diagnostic(
                    self.name(),
                    ContextKey::Diagnostic,
                    "model-infer-error",
                    err.to_string(),
                ));
            }
        };

        let infer_df = match load_dataframe(&split.infer_path) {
            Ok(df) => df,
            Err(err) => {
                return AgentEffect::with_proposal(diagnostic(
                    self.name(),
                    ContextKey::Diagnostic,
                    "model-infer-error",
                    err.to_string(),
                ));
            }
        };

        let target = match get_numeric_series(&infer_df, &model.target_column) {
            Ok(series) => series,
            Err(err) => {
                return AgentEffect::with_proposal(diagnostic(
                    self.name(),
                    ContextKey::Diagnostic,
                    "model-infer-error",
                    err.to_string(),
                ));
            }
        };

        let sample_rows = self.max_rows.min(infer_df.height().max(1));
        let actuals = match target.f64() {
            Ok(series) => series
                .into_no_null_iter()
                .take(sample_rows)
                .collect::<Vec<_>>(),
            Err(err) => {
                return AgentEffect::with_proposal(diagnostic(
                    self.name(),
                    ContextKey::Diagnostic,
                    "model-infer-error",
                    err.to_string(),
                ));
            }
        };

        let predictions = vec![model.mean; actuals.len()];
        let sample = InferenceSample {
            model_path: read_model_path_from_ctx(ctx).unwrap_or_default(),
            target_column: model.target_column,
            rows: actuals.len(),
            predictions,
            actuals,
            iteration: split.iteration,
        };

        AgentEffect::with_proposal(proposal(
            self.name(),
            ContextKey::Hypotheses,
            format!("inference-sample-{}", split.iteration),
            sample,
        ))
    }
}

#[derive(Debug, Default)]
pub struct ModelRegistryAgent;

impl ModelRegistryAgent {
    pub fn new() -> Self {
        Self
    }
}

#[async_trait::async_trait]
impl Suggestor for ModelRegistryAgent {
    fn name(&self) -> &'static str {
        "ModelRegistryAgent"
    }

    fn dependencies(&self) -> &[ContextKey] {
        &[ContextKey::Strategies, ContextKey::Evaluations]
    }

    fn accepts(&self, ctx: &dyn Context) -> bool {
        ctx.has(ContextKey::Strategies)
            && ctx.has(ContextKey::Evaluations)
            && match read_latest_model_meta_from_ctx(ctx) {
                Ok(meta) => !has_registry_record_for_iteration(ctx, meta.iteration),
                Err(_) => false,
            }
    }

    fn provenance(&self) -> Provenance {
        Provenance::from(CRUCIBLE_PROVENANCE.as_str())
    }

    async fn execute(&self, ctx: &dyn Context) -> AgentEffect {
        let meta = match read_latest_model_meta_from_ctx(ctx) {
            Ok(meta) => meta,
            Err(err) => {
                return AgentEffect::with_proposal(diagnostic(
                    self.name(),
                    ContextKey::Diagnostic,
                    "model-registry-error",
                    err.to_string(),
                ));
            }
        };

        let report = latest_evaluation_report(ctx, meta.iteration);
        let mut metrics = HashMap::new();
        if let Some(report) = report {
            metrics.insert(report.metric, report.value);
            metrics.insert("success_ratio".to_string(), report.success_ratio);
        }

        let record = ModelRegistryRecord {
            kind: "model_registry".to_string(),
            iteration: meta.iteration,
            model_path: meta.model_path,
            metrics,
            provenance: "training_flow".to_string(),
        };

        AgentEffect::with_proposal(proposal(
            self.name(),
            ContextKey::Strategies,
            format!("model-registry-{}", record.iteration),
            record,
        ))
    }
}

#[derive(Debug, Default)]
pub struct MonitoringAgent;

impl MonitoringAgent {
    pub fn new() -> Self {
        Self
    }
}

#[async_trait::async_trait]
impl Suggestor for MonitoringAgent {
    fn name(&self) -> &'static str {
        "MonitoringAgent"
    }

    fn dependencies(&self) -> &[ContextKey] {
        &[ContextKey::Evaluations]
    }

    fn accepts(&self, ctx: &dyn Context) -> bool {
        ctx.has(ContextKey::Evaluations)
            && match latest_evaluation_report(ctx, 0) {
                Some(report) => !has_monitoring_report_for_iteration(ctx, report.iteration),
                None => false,
            }
    }

    fn provenance(&self) -> Provenance {
        Provenance::from(CRUCIBLE_PROVENANCE.as_str())
    }

    async fn execute(&self, ctx: &dyn Context) -> AgentEffect {
        let report = match latest_evaluation_report(ctx, 0) {
            Some(report) => report,
            None => return AgentEffect::empty(),
        };

        let status = if report.success_ratio >= 0.75 {
            "healthy"
        } else {
            "needs_attention"
        };

        let monitoring = MonitoringReport {
            kind: "monitoring".to_string(),
            iteration: report.iteration,
            metric: report.metric,
            value: report.value,
            baseline: report.mean_abs_target,
            status: status.to_string(),
        };

        AgentEffect::with_proposal(proposal(
            self.name(),
            ContextKey::Evaluations,
            format!("monitoring-{}", report.iteration),
            monitoring,
        ))
    }
}

#[derive(Debug, Default)]
pub struct DeploymentAgent;

impl DeploymentAgent {
    pub fn new() -> Self {
        Self
    }
}

#[async_trait::async_trait]
impl Suggestor for DeploymentAgent {
    fn name(&self) -> &'static str {
        "DeploymentAgent"
    }

    fn dependencies(&self) -> &[ContextKey] {
        &[ContextKey::Evaluations, ContextKey::Strategies]
    }

    fn accepts(&self, ctx: &dyn Context) -> bool {
        ctx.has(ContextKey::Evaluations)
            && ctx.has(ContextKey::Strategies)
            && match latest_evaluation_report(ctx, 0) {
                Some(report) => !has_deployment_decision_for_iteration(ctx, report.iteration),
                None => false,
            }
    }

    fn provenance(&self) -> Provenance {
        Provenance::from(CRUCIBLE_PROVENANCE.as_str())
    }

    async fn execute(&self, ctx: &dyn Context) -> AgentEffect {
        let report = match latest_evaluation_report(ctx, 0) {
            Some(report) => report,
            None => return AgentEffect::empty(),
        };

        let quality_threshold =
            read_latest_plan_from_ctx(ctx).map_or(0.75, |plan| plan.quality_threshold);

        let (action, retrain, reason) = if report.success_ratio >= quality_threshold {
            ("deploy", false, "meets quality threshold")
        } else {
            ("hold", true, "below quality threshold")
        };

        let decision = DeploymentDecision {
            kind: "deployment_decision".to_string(),
            iteration: report.iteration,
            action: action.to_string(),
            reason: reason.to_string(),
            retrain,
        };

        AgentEffect::with_proposal(proposal(
            self.name(),
            ContextKey::Strategies,
            format!("deployment-{}", report.iteration),
            decision,
        ))
    }
}
