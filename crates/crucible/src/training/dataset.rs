// Copyright 2024-2026 Reflective Labs

use converge_pack::{AgentEffect, Context, ContextKey, ProvenanceSource, Suggestor};
use std::fs::create_dir_all;
use std::path::PathBuf;

use crate::provenance::CRUCIBLE_PROVENANCE;

use super::io::{download_dataset_if_missing, load_dataframe, write_parquet};
use super::types::{DatasetSplit, TrainingPlan, diagnostic, has_split_for_iteration, proposal, read_latest_plan_from_ctx};

#[derive(Debug)]
pub struct DatasetAgent {
    data_dir: PathBuf,
}

impl DatasetAgent {
    pub fn new(data_dir: PathBuf) -> Self {
        Self { data_dir }
    }

    fn dataset_path(&self) -> PathBuf {
        self.data_dir.join("california_housing_train.parquet")
    }

    fn split_paths(&self) -> (PathBuf, PathBuf, PathBuf) {
        (
            self.data_dir.join("train.parquet"),
            self.data_dir.join("val.parquet"),
            self.data_dir.join("infer.parquet"),
        )
    }
}

#[async_trait::async_trait]
impl Suggestor for DatasetAgent {
    fn name(&self) -> &'static str {
        "DatasetAgent (HuggingFace)"
    }

    fn dependencies(&self) -> &[ContextKey] {
        &[ContextKey::Seeds]
    }

    fn accepts(&self, ctx: &dyn Context) -> bool {
        if !ctx.has(ContextKey::Seeds) {
            return false;
        }

        let plan = read_latest_plan_from_ctx(ctx);
        if let Some(plan) = plan {
            return !has_split_for_iteration(ctx, plan.iteration);
        }

        !ctx.has(ContextKey::Signals)
    }

    fn provenance(&self) -> &'static str {
        CRUCIBLE_PROVENANCE.as_str()
    }

    async fn execute(&self, ctx: &dyn Context) -> AgentEffect {
        if let Err(err) = create_dir_all(&self.data_dir) {
            return AgentEffect::with_proposal(diagnostic(
                self.name(),
                ContextKey::Diagnostic,
                "dataset-agent-error",
                err.to_string(),
            ));
        }

        let dataset_path = self.dataset_path();
        if let Err(err) = download_dataset_if_missing(&dataset_path) {
            return AgentEffect::with_proposal(diagnostic(
                self.name(),
                ContextKey::Diagnostic,
                "dataset-agent-error",
                err.to_string(),
            ));
        }

        let df = match load_dataframe(&dataset_path) {
            Ok(df) => df,
            Err(err) => {
                return AgentEffect::with_proposal(diagnostic(
                    self.name(),
                    ContextKey::Diagnostic,
                    "dataset-agent-error",
                    err.to_string(),
                ));
            }
        };

        let total_rows = df.height();
        if total_rows < 10 {
            return AgentEffect::with_proposal(diagnostic(
                self.name(),
                ContextKey::Diagnostic,
                "dataset-agent-error",
                "dataset too small for splitting",
            ));
        }

        let plan = read_latest_plan_from_ctx(ctx).unwrap_or(TrainingPlan {
            iteration: 1,
            max_rows: total_rows,
            train_fraction: 0.8,
            val_fraction: 0.15,
            infer_fraction: 0.05,
            quality_threshold: 0.75,
        });

        let max_rows = plan.max_rows.min(total_rows).max(10);
        let df = df.slice(0, max_rows);

        let mut train_rows = ((max_rows as f64) * plan.train_fraction).floor() as usize;
        let mut val_rows = ((max_rows as f64) * plan.val_fraction).floor() as usize;
        let mut infer_rows = max_rows.saturating_sub(train_rows + val_rows);
        if infer_rows == 0 {
            if val_rows > 1 {
                val_rows -= 1;
            } else if train_rows > 1 {
                train_rows -= 1;
            }
            infer_rows = max_rows.saturating_sub(train_rows + val_rows).max(1);
        }

        let (train_path, val_path, infer_path) = self.split_paths();
        let train_df = df.slice(0, train_rows);
        let val_df = df.slice(train_rows as i64, val_rows);
        let infer_df = df.slice((train_rows + val_rows) as i64, infer_rows);

        if let Err(err) = write_parquet(&train_df, &train_path)
            .and_then(|()| write_parquet(&val_df, &val_path))
            .and_then(|()| write_parquet(&infer_df, &infer_path))
        {
            return AgentEffect::with_proposal(diagnostic(
                self.name(),
                ContextKey::Diagnostic,
                "dataset-agent-error",
                err.to_string(),
            ));
        }

        let split = DatasetSplit {
            source_path: dataset_path,
            train_path,
            val_path,
            infer_path,
            total_rows,
            max_rows,
            train_rows,
            val_rows,
            infer_rows,
            iteration: plan.iteration,
        };

        AgentEffect::with_proposal(proposal(
            self.name(),
            ContextKey::Signals,
            format!("dataset-split-{}", plan.iteration),
            split,
        ))
    }
}

// Tests that are specific to DatasetAgent paths
#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn dataset_agent_paths() {
        let agent = DatasetAgent::new(PathBuf::from("/tmp/data"));
        assert_eq!(
            agent.dataset_path(),
            PathBuf::from("/tmp/data/california_housing_train.parquet")
        );
        let (train, val, infer) = agent.split_paths();
        assert_eq!(train, PathBuf::from("/tmp/data/train.parquet"));
        assert_eq!(val, PathBuf::from("/tmp/data/val.parquet"));
        assert_eq!(infer, PathBuf::from("/tmp/data/infer.parquet"));
    }
}

