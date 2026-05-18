// Copyright 2024-2026 Reflective Labs

use converge_pack::{AgentEffect, Context, ContextKey, ProvenanceSource, Suggestor};
use std::fs::create_dir_all;
use std::path::PathBuf;

use crate::provenance::CRUCIBLE_PROVENANCE;

use super::features::apply_feature_spec;
use super::io::{load_dataframe, mean_of_series, select_target_column, write_json};
use super::types::{
    BaselineModel, ModelMetadata, diagnostic, has_model_for_iteration, proposal,
    read_feature_spec_from_ctx, read_latest_split_from_ctx,
};

#[derive(Debug)]
pub struct ModelTrainingAgent {
    model_dir: PathBuf,
}

impl ModelTrainingAgent {
    pub fn new(model_dir: PathBuf) -> Self {
        Self { model_dir }
    }

    fn model_path(&self) -> PathBuf {
        self.model_dir.join("baseline_mean.json")
    }
}

#[async_trait::async_trait]
impl Suggestor for ModelTrainingAgent {
    fn name(&self) -> &'static str {
        "ModelTrainingAgent (Baseline)"
    }

    fn dependencies(&self) -> &[ContextKey] {
        &[ContextKey::Signals]
    }

    fn accepts(&self, ctx: &dyn Context) -> bool {
        if !ctx.has(ContextKey::Signals) {
            return false;
        }
        let split = match read_latest_split_from_ctx(ctx) {
            Ok(split) => split,
            Err(_) => return false,
        };
        !has_model_for_iteration(ctx, split.iteration)
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
                    "model-training-error",
                    err.to_string(),
                ));
            }
        };

        if let Err(err) = create_dir_all(&self.model_dir) {
            return AgentEffect::with_proposal(diagnostic(
                self.name(),
                ContextKey::Diagnostic,
                "model-training-error",
                err.to_string(),
            ));
        }

        let raw_train_df = match load_dataframe(&split.train_path) {
            Ok(df) => df,
            Err(err) => {
                return AgentEffect::with_proposal(diagnostic(
                    self.name(),
                    ContextKey::Diagnostic,
                    "model-training-error",
                    err.to_string(),
                ));
            }
        };

        // Apply FeatureSpec transformation if available
        let train_df = match read_feature_spec_from_ctx(ctx, split.iteration) {
            Some(spec) => match apply_feature_spec(&raw_train_df, &spec) {
                Ok(df) => df,
                Err(err) => {
                    return AgentEffect::with_proposal(diagnostic(
                        self.name(),
                        ContextKey::Diagnostic,
                        "model-training-error",
                        format!("feature spec application failed: {}", err),
                    ));
                }
            },
            None => raw_train_df,
        };

        let (target_name, target) = match select_target_column(&train_df) {
            Ok(value) => value,
            Err(err) => {
                return AgentEffect::with_proposal(diagnostic(
                    self.name(),
                    ContextKey::Diagnostic,
                    "model-training-error",
                    err.to_string(),
                ));
            }
        };

        let mean = match mean_of_series(&target) {
            Ok(value) => value,
            Err(err) => {
                return AgentEffect::with_proposal(diagnostic(
                    self.name(),
                    ContextKey::Diagnostic,
                    "model-training-error",
                    err.to_string(),
                ));
            }
        };

        let model = BaselineModel {
            target_column: target_name.clone(),
            mean,
        };

        let model_path = self.model_path();
        if let Err(err) = write_json(&model_path, &model) {
            return AgentEffect::with_proposal(diagnostic(
                self.name(),
                ContextKey::Diagnostic,
                "model-training-error",
                err.to_string(),
            ));
        }

        let meta = ModelMetadata {
            model_path,
            target_column: target_name,
            train_rows: split.train_rows,
            baseline_mean: mean,
            iteration: split.iteration,
        };

        AgentEffect::with_proposal(proposal(
            self.name(),
            ContextKey::Strategies,
            format!("trained-model-{}", split.iteration),
            meta,
        ))
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn model_training_agent_model_path() {
        let agent = ModelTrainingAgent::new(PathBuf::from("/tmp/models"));
        assert_eq!(
            agent.model_path(),
            PathBuf::from("/tmp/models/baseline_mean.json")
        );
    }
}
