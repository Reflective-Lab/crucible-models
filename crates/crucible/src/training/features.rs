// Copyright 2024-2026 Reflective Labs

use anyhow::{Context as _, Result, anyhow};
use converge_pack::{AgentEffect, Context, ContextKey, ProvenanceSource, Suggestor};
use polars::prelude::*;

use crate::provenance::CRUCIBLE_PROVENANCE;

use super::io::{
    compute_mean_std, compute_numeric_stats, is_numeric_dtype, load_dataframe, select_target_column,
    split_feature_columns,
};
use super::types::{
    DataQualityReport, FeatureInteraction, FeatureSpec, HyperparameterSearchPlan,
    HyperparameterSearchResult, TrainingPlan, diagnostic, drift_score_from_ctx,
    has_data_quality_for_iteration, has_feature_spec_for_iteration,
    has_hyperparam_result_for_iteration, proposal, read_latest_plan_from_ctx,
    read_latest_split_from_ctx,
};

use std::collections::HashMap;

#[derive(Debug, Default)]
pub struct DataValidationAgent;

impl DataValidationAgent {
    pub fn new() -> Self {
        Self
    }
}

#[async_trait::async_trait]
impl Suggestor for DataValidationAgent {
    fn name(&self) -> &'static str {
        "DataValidationAgent"
    }

    fn dependencies(&self) -> &[ContextKey] {
        &[ContextKey::Signals]
    }

    fn accepts(&self, ctx: &dyn Context) -> bool {
        ctx.has(ContextKey::Signals)
            && match read_latest_split_from_ctx(ctx) {
                Ok(split) => !has_data_quality_for_iteration(ctx, split.iteration),
                Err(_) => false,
            }
    }

    fn provenance(&self) -> &'static str {
        CRUCIBLE_PROVENANCE.as_str()
    }

    async fn execute(&self, ctx: &dyn Context) -> AgentEffect {
        let split = match read_latest_split_from_ctx(ctx) {
            Ok(split) => split,
            Err(err) => {
                return AgentEffect::with_proposal(diagnostic(
                    self.name(),
                    ContextKey::Diagnostic,
                    "data-validation-error",
                    err.to_string(),
                ));
            }
        };

        let df = match load_dataframe(&split.train_path) {
            Ok(df) => df,
            Err(err) => {
                return AgentEffect::with_proposal(diagnostic(
                    self.name(),
                    ContextKey::Diagnostic,
                    "data-validation-error",
                    err.to_string(),
                ));
            }
        };

        let rows = df.height();
        let mut missingness = HashMap::new();
        let mut numeric_means = HashMap::new();
        let mut outlier_counts = HashMap::new();

        for series in df.get_columns() {
            let name = series.name().to_string();
            let null_ratio = if rows > 0 {
                series.null_count() as f64 / rows as f64
            } else {
                0.0
            };
            missingness.insert(name.clone(), null_ratio);

            if is_numeric_dtype(series.dtype())
                && let Ok((mean, _std, outliers)) =
                    compute_numeric_stats(series.as_materialized_series())
            {
                numeric_means.insert(name.clone(), mean);
                outlier_counts.insert(name, outliers);
            }
        }

        let drift_score = drift_score_from_ctx(ctx, split.iteration, &numeric_means);

        let report = DataQualityReport {
            kind: "data_quality".to_string(),
            iteration: split.iteration,
            source_path: split.train_path.clone(),
            rows_checked: rows,
            missingness,
            numeric_means,
            outlier_counts,
            drift_score,
        };

        AgentEffect::with_proposal(proposal(
            self.name(),
            ContextKey::Signals,
            format!("data-quality-{}", split.iteration),
            report,
        ))
    }
}

#[derive(Debug, Default)]
pub struct FeatureEngineeringAgent;

impl FeatureEngineeringAgent {
    pub fn new() -> Self {
        Self
    }
}

#[async_trait::async_trait]
impl Suggestor for FeatureEngineeringAgent {
    fn name(&self) -> &'static str {
        "FeatureEngineeringAgent"
    }

    fn dependencies(&self) -> &[ContextKey] {
        &[ContextKey::Signals]
    }

    fn accepts(&self, ctx: &dyn Context) -> bool {
        ctx.has(ContextKey::Signals)
            && match read_latest_split_from_ctx(ctx) {
                Ok(split) => !has_feature_spec_for_iteration(ctx, split.iteration),
                Err(_) => false,
            }
    }

    fn provenance(&self) -> &'static str {
        CRUCIBLE_PROVENANCE.as_str()
    }

    async fn execute(&self, ctx: &dyn Context) -> AgentEffect {
        let split = match read_latest_split_from_ctx(ctx) {
            Ok(split) => split,
            Err(err) => {
                return AgentEffect::with_proposal(diagnostic(
                    self.name(),
                    ContextKey::Diagnostic,
                    "feature-engineering-error",
                    err.to_string(),
                ));
            }
        };

        let df = match load_dataframe(&split.train_path) {
            Ok(df) => df,
            Err(err) => {
                return AgentEffect::with_proposal(diagnostic(
                    self.name(),
                    ContextKey::Diagnostic,
                    "feature-engineering-error",
                    err.to_string(),
                ));
            }
        };

        let (target_column, _) = match select_target_column(&df) {
            Ok(value) => value,
            Err(err) => {
                return AgentEffect::with_proposal(diagnostic(
                    self.name(),
                    ContextKey::Diagnostic,
                    "feature-engineering-error",
                    err.to_string(),
                ));
            }
        };

        let (numeric_features, categorical_features) = split_feature_columns(&df, &target_column);

        let mut interactions = Vec::new();
        if numeric_features.len() >= 2 {
            interactions.push(FeatureInteraction {
                name: format!("{}_x_{}", numeric_features[0], numeric_features[1]),
                left: numeric_features[0].clone(),
                right: numeric_features[1].clone(),
                op: "multiply".to_string(),
            });
        }

        let spec = FeatureSpec {
            kind: "feature_spec".to_string(),
            iteration: split.iteration,
            target_column,
            numeric_features,
            categorical_features,
            normalization: "standardize".to_string(),
            interactions,
        };

        AgentEffect::with_proposal(proposal(
            self.name(),
            ContextKey::Constraints,
            format!("feature-spec-{}", split.iteration),
            spec,
        ))
    }
}

#[derive(Debug)]
pub struct HyperparameterSearchAgent {
    pub max_trials: usize,
}

impl HyperparameterSearchAgent {
    pub fn new(max_trials: usize) -> Self {
        Self { max_trials }
    }
}

#[async_trait::async_trait]
impl Suggestor for HyperparameterSearchAgent {
    fn name(&self) -> &'static str {
        "HyperparameterSearchAgent"
    }

    fn dependencies(&self) -> &[ContextKey] {
        &[ContextKey::Constraints, ContextKey::Signals]
    }

    fn accepts(&self, ctx: &dyn Context) -> bool {
        ctx.has(ContextKey::Signals)
            && match read_latest_split_from_ctx(ctx) {
                Ok(split) => !has_hyperparam_result_for_iteration(ctx, split.iteration),
                Err(_) => false,
            }
    }

    fn provenance(&self) -> &'static str {
        CRUCIBLE_PROVENANCE.as_str()
    }

    async fn execute(&self, ctx: &dyn Context) -> AgentEffect {
        let split = match read_latest_split_from_ctx(ctx) {
            Ok(split) => split,
            Err(err) => {
                return AgentEffect::with_proposal(diagnostic(
                    self.name(),
                    ContextKey::Diagnostic,
                    "hyperparam-search-error",
                    err.to_string(),
                ));
            }
        };

        let training_plan = read_latest_plan_from_ctx(ctx).unwrap_or(TrainingPlan {
            iteration: split.iteration,
            max_rows: split.max_rows,
            train_fraction: 0.8,
            val_fraction: 0.15,
            infer_fraction: 0.05,
            quality_threshold: 0.75,
        });

        let mut params = HashMap::new();
        params.insert("learning_rate".to_string(), vec![0.001, 0.01, 0.1]);
        params.insert("hidden_size".to_string(), vec![8.0, 16.0, 32.0]);

        let plan = HyperparameterSearchPlan {
            kind: "hyperparam_plan".to_string(),
            iteration: split.iteration,
            max_trials: self.max_trials,
            early_stopping: true,
            params,
        };

        let mut best_params = HashMap::new();
        best_params.insert("learning_rate".to_string(), 0.01);
        best_params.insert("hidden_size".to_string(), 16.0);
        let score = (1.0 - training_plan.quality_threshold) * plan.max_trials as f64
            / plan.iteration.max(1) as f64;
        let result = HyperparameterSearchResult {
            kind: "hyperparam_result".to_string(),
            iteration: split.iteration,
            best_params,
            score,
        };

        AgentEffect::builder()
            .proposal(proposal(
                self.name(),
                ContextKey::Constraints,
                format!("hyperparam-plan-{}", split.iteration),
                plan,
            ))
            .proposal(proposal(
                self.name(),
                ContextKey::Evaluations,
                format!("hyperparam-result-{}", split.iteration),
                result,
            ))
            .build()
    }
}

/// Apply a FeatureSpec to a DataFrame, creating interaction features and normalizing
pub fn apply_feature_spec(df: &DataFrame, spec: &FeatureSpec) -> Result<DataFrame> {
    let mut result = df.clone();

    // Apply feature interactions
    for interaction in &spec.interactions {
        let left_col = result
            .column(&interaction.left)
            .map_err(|_| anyhow!("missing column {} for interaction", interaction.left))?
            .cast(&DataType::Float64)?;
        let right_col = result
            .column(&interaction.right)
            .map_err(|_| anyhow!("missing column {} for interaction", interaction.right))?
            .cast(&DataType::Float64)?;

        let left_vals = left_col.f64().context("left column not f64")?;
        let right_vals = right_col.f64().context("right column not f64")?;

        let interaction_series = match interaction.op.as_str() {
            "multiply" => left_vals * right_vals,
            "add" => left_vals + right_vals,
            "subtract" => left_vals - right_vals,
            "divide" => {
                // Safe division: use map to handle division safely
                left_vals
                    .into_iter()
                    .zip(right_vals.into_iter())
                    .map(|(l, r)| match (l, r) {
                        (Some(lv), Some(rv)) if rv.abs() > 1e-10 => Some(lv / rv),
                        _ => None,
                    })
                    .collect::<Float64Chunked>()
            }
            _ => return Err(anyhow!("unsupported interaction op: {}", interaction.op)),
        };

        let named_series = interaction_series.with_name(interaction.name.clone().into());
        result = result
            .hstack(&[named_series.into_series().into()])
            .context("failed to add interaction column")?;
    }

    // Apply normalization to numeric features
    if spec.normalization == "standardize" {
        for col_name in &spec.numeric_features {
            if let Ok(col) = result.column(col_name) {
                let casted = col.cast(&DataType::Float64)?;
                let values = casted.f64().context("column not f64")?;

                // Compute mean and std
                let (mean, std) = compute_mean_std(values)?;

                if std > 0.0 {
                    // Standardize: (x - mean) / std
                    let standardized = (values - mean) / std;
                    let named = standardized.with_name(col_name.clone().into());

                    // Replace the column
                    result = result.drop(col_name)?;
                    result = result
                        .hstack(&[named.into_series().into()])
                        .context("failed to replace standardized column")?;
                }
            }
        }
    }

    Ok(result)
}
