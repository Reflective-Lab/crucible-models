// Copyright 2024-2026 Reflective Labs

mod dataset;
mod evaluation;
mod features;
mod io;
mod pipeline;
mod types;

// ── Public re-exports ─────────────────────────────────────────────────────────

pub use dataset::DatasetAgent;
pub use evaluation::{
    DeploymentAgent, ModelEvaluationAgent, ModelRegistryAgent, MonitoringAgent,
    SampleInferenceAgent,
};
pub use features::{
    DataValidationAgent, FeatureEngineeringAgent, HyperparameterSearchAgent, apply_feature_spec,
};
pub use pipeline::ModelTrainingAgent;
pub use types::{
    BaselineModel, DataQualityReport, DatasetSplit, DeploymentDecision, EvaluationReport,
    FeatureInteraction, FeatureSpec, HyperparameterSearchPlan, HyperparameterSearchResult,
    InferenceSample, ModelMetadata, ModelRegistryRecord, MonitoringReport, TrainingPlan,
};

#[cfg(test)]
mod tests {
    use super::*;
    use converge_pack::{ContextKey, DiagnosticPayload};
    use polars::prelude::*;
    use std::collections::HashMap;
    use std::path::PathBuf;

    use crate::training::io::{
        compute_numeric_stats, is_numeric_dtype, mean_abs_error, mean_abs_value, mean_of_series,
        select_target_column, split_feature_columns,
    };
    use crate::training::types::diagnostic;

    #[test]
    fn proposal_helper_builds_correct_fact() {
        let p = diagnostic("my-agent", ContextKey::Diagnostic, "id-1", "content-1");
        assert_eq!(p.provenance(), "crucible");
        assert_eq!(p.key, ContextKey::Diagnostic);
        assert_eq!(p.id, "id-1");
        assert_eq!(
            p.require_payload::<DiagnosticPayload>().unwrap().message(),
            "content-1"
        );
    }

    #[test]
    fn training_plan_serde_roundtrip() {
        let plan = TrainingPlan {
            iteration: 3,
            max_rows: 1000,
            train_fraction: 0.7,
            val_fraction: 0.2,
            infer_fraction: 0.1,
            quality_threshold: 0.8,
        };
        let json = serde_json::to_string(&plan).unwrap();
        let restored: TrainingPlan = serde_json::from_str(&json).unwrap();
        assert_eq!(restored.iteration, 3);
        assert_eq!(restored.max_rows, 1000);
        assert!((restored.train_fraction - 0.7).abs() < f64::EPSILON);
    }

    #[test]
    fn dataset_split_serde_roundtrip() {
        let split = DatasetSplit {
            source_path: "/data/src.parquet".into(),
            train_path: "/data/train.parquet".into(),
            val_path: "/data/val.parquet".into(),
            infer_path: "/data/infer.parquet".into(),
            total_rows: 1000,
            max_rows: 800,
            train_rows: 640,
            val_rows: 120,
            infer_rows: 40,
            iteration: 1,
        };
        let json = serde_json::to_string(&split).unwrap();
        let restored: DatasetSplit = serde_json::from_str(&json).unwrap();
        assert_eq!(restored.total_rows, 1000);
        assert_eq!(
            restored.train_rows + restored.val_rows + restored.infer_rows,
            800
        );
    }

    #[test]
    fn baseline_model_serde_roundtrip() {
        let model = BaselineModel {
            target_column: "price".into(),
            mean: 42.5,
        };
        let json = serde_json::to_string(&model).unwrap();
        let restored: BaselineModel = serde_json::from_str(&json).unwrap();
        assert_eq!(restored.target_column, "price");
        assert!((restored.mean - 42.5).abs() < f64::EPSILON);
    }

    #[test]
    fn evaluation_report_success_ratio_bounds() {
        let report = EvaluationReport {
            model_path: "/model".into(),
            metric: "mae".into(),
            value: 10.0,
            mean_abs_target: 100.0,
            success_ratio: 0.9,
            val_rows: 50,
            iteration: 1,
        };
        assert!(report.success_ratio >= 0.0 && report.success_ratio <= 1.0);
    }

    #[test]
    fn feature_interaction_construction() {
        let fi = FeatureInteraction {
            name: "a_x_b".into(),
            left: "a".into(),
            right: "b".into(),
            op: "multiply".into(),
        };
        assert_eq!(fi.name, "a_x_b");
        assert_eq!(fi.op, "multiply");
    }

    #[test]
    fn feature_spec_serde_roundtrip() {
        let spec = FeatureSpec {
            kind: "feature_spec".into(),
            iteration: 2,
            target_column: "target".into(),
            numeric_features: vec!["a".into(), "b".into()],
            categorical_features: vec!["c".into()],
            normalization: "standardize".into(),
            interactions: vec![],
        };
        let json = serde_json::to_string(&spec).unwrap();
        let restored: FeatureSpec = serde_json::from_str(&json).unwrap();
        assert_eq!(restored.numeric_features.len(), 2);
        assert_eq!(restored.categorical_features.len(), 1);
    }

    #[test]
    fn hyperparam_search_plan_construction() {
        let mut params = HashMap::new();
        params.insert("lr".to_string(), vec![0.001, 0.01]);
        let plan = HyperparameterSearchPlan {
            kind: "hyperparam_plan".into(),
            iteration: 1,
            max_trials: 10,
            early_stopping: true,
            params,
        };
        assert_eq!(plan.max_trials, 10);
        assert!(plan.early_stopping);
        assert_eq!(plan.params["lr"].len(), 2);
    }

    #[test]
    fn hyperparam_search_result_serde_roundtrip() {
        let mut best = HashMap::new();
        best.insert("lr".to_string(), 0.01);
        let result = HyperparameterSearchResult {
            kind: "hyperparam_result".into(),
            iteration: 1,
            best_params: best,
            score: 0.85,
        };
        let json = serde_json::to_string(&result).unwrap();
        let restored: HyperparameterSearchResult = serde_json::from_str(&json).unwrap();
        assert!((restored.score - 0.85).abs() < f64::EPSILON);
        assert!((restored.best_params["lr"] - 0.01).abs() < f64::EPSILON);
    }

    #[test]
    fn model_registry_record_construction() {
        let mut metrics = HashMap::new();
        metrics.insert("mae".to_string(), 5.0);
        let record = ModelRegistryRecord {
            kind: "model_registry".into(),
            iteration: 1,
            model_path: "/models/v1.json".into(),
            metrics,
            provenance: "test".into(),
        };
        assert_eq!(record.metrics["mae"], 5.0);
    }

    #[test]
    fn monitoring_report_status_values() {
        let healthy = MonitoringReport {
            kind: "monitoring".into(),
            iteration: 1,
            metric: "mae".into(),
            value: 5.0,
            baseline: 100.0,
            status: "healthy".into(),
        };
        assert_eq!(healthy.status, "healthy");

        let needs_attention = MonitoringReport {
            status: "needs_attention".into(),
            ..healthy.clone()
        };
        assert_eq!(needs_attention.status, "needs_attention");
    }

    #[test]
    fn deployment_decision_deploy_vs_hold() {
        let deploy = DeploymentDecision {
            kind: "deployment_decision".into(),
            iteration: 1,
            action: "deploy".into(),
            reason: "meets threshold".into(),
            retrain: false,
        };
        assert!(!deploy.retrain);

        let hold = DeploymentDecision {
            action: "hold".into(),
            retrain: true,
            ..deploy.clone()
        };
        assert!(hold.retrain);
        assert_eq!(hold.action, "hold");
    }

    #[test]
    fn data_validation_agent_default() {
        let agent = DataValidationAgent;
        let agent2 = DataValidationAgent::new();
        assert_eq!(format!("{:?}", agent), format!("{:?}", agent2));
    }

    #[test]
    fn feature_engineering_agent_default() {
        let agent = FeatureEngineeringAgent;
        let agent2 = FeatureEngineeringAgent::new();
        assert_eq!(format!("{:?}", agent), format!("{:?}", agent2));
    }

    #[test]
    fn model_evaluation_agent_default() {
        let agent = ModelEvaluationAgent;
        let agent2 = ModelEvaluationAgent::new();
        assert_eq!(format!("{:?}", agent), format!("{:?}", agent2));
    }

    #[test]
    fn model_registry_agent_default() {
        let agent = ModelRegistryAgent;
        let agent2 = ModelRegistryAgent::new();
        assert_eq!(format!("{:?}", agent), format!("{:?}", agent2));
    }

    #[test]
    fn monitoring_agent_default() {
        let agent = MonitoringAgent;
        let agent2 = MonitoringAgent::new();
        assert_eq!(format!("{:?}", agent), format!("{:?}", agent2));
    }

    #[test]
    fn deployment_agent_default() {
        let agent = DeploymentAgent;
        let agent2 = DeploymentAgent::new();
        assert_eq!(format!("{:?}", agent), format!("{:?}", agent2));
    }

    #[test]
    fn dataset_agent_paths() {
        let agent = DatasetAgent::new(PathBuf::from("/tmp/data"));
        // DatasetAgent paths are tested in dataset::tests
        // Verify construction works from public API
        let _ = agent;
    }

    #[test]
    fn sample_inference_agent_construction() {
        let agent = SampleInferenceAgent::new(100);
        assert_eq!(agent.max_rows, 100);
    }

    #[test]
    fn hyperparameter_search_agent_construction() {
        let agent = HyperparameterSearchAgent::new(50);
        assert_eq!(agent.max_trials, 50);
    }

    #[test]
    fn is_numeric_dtype_comprehensive() {
        assert!(is_numeric_dtype(&DataType::Float32));
        assert!(is_numeric_dtype(&DataType::Float64));
        assert!(is_numeric_dtype(&DataType::Int64));
        assert!(!is_numeric_dtype(&DataType::String));
        assert!(!is_numeric_dtype(&DataType::Boolean));
    }

    #[test]
    fn split_feature_columns_separates_types() {
        let df = df! {
            "num1" => [1.0, 2.0],
            "num2" => [3i32, 4],
            "cat1" => ["a", "b"],
            "target" => [10.0, 20.0]
        }
        .unwrap();
        let (numeric, categorical) = split_feature_columns(&df, "target");
        assert!(numeric.contains(&"num1".to_string()));
        assert!(numeric.contains(&"num2".to_string()));
        assert!(categorical.contains(&"cat1".to_string()));
        assert!(!numeric.contains(&"target".to_string()));
        assert!(!categorical.contains(&"target".to_string()));
    }

    #[test]
    fn select_target_column_prefers_named_target() {
        let df = df! {
            "x" => [1.0, 2.0],
            "median_house_value" => [100.0, 200.0]
        }
        .unwrap();
        let (name, _) = select_target_column(&df).unwrap();
        assert_eq!(name, "median_house_value");
    }

    #[test]
    fn select_target_column_falls_back_to_last_numeric() {
        let df = df! {
            "a" => [1.0, 2.0],
            "b" => [3.0, 4.0]
        }
        .unwrap();
        let (name, _) = select_target_column(&df).unwrap();
        assert_eq!(name, "b");
    }

    #[test]
    fn select_target_column_fails_with_no_numeric() {
        let df = df! {
            "text" => ["a", "b"]
        }
        .unwrap();
        assert!(select_target_column(&df).is_err());
    }

    #[test]
    fn mean_of_series_computes_correctly() {
        let series = Series::new("v".into(), &[2.0f64, 4.0, 6.0]);
        let mean = mean_of_series(&series).unwrap();
        assert!((mean - 4.0).abs() < f64::EPSILON);
    }

    #[test]
    fn mean_abs_error_computes_correctly() {
        let series = Series::new("v".into(), &[10.0f64, 20.0, 30.0]);
        let mae = mean_abs_error(&series, 20.0).unwrap();
        // |10-20| + |20-20| + |30-20| = 10+0+10 = 20, /3 = 6.666...
        assert!((mae - 20.0 / 3.0).abs() < 1e-10);
    }

    #[test]
    fn mean_abs_value_computes_correctly() {
        let series = Series::new("v".into(), &[-5.0f64, 5.0, 10.0]);
        let mav = mean_abs_value(&series).unwrap();
        assert!((mav - 20.0 / 3.0).abs() < 1e-10);
    }

    #[test]
    fn compute_numeric_stats_basic() {
        let series = Series::new("v".into(), &[2.0f64, 4.0, 6.0]);
        let casted = series.cast(&DataType::Float64).unwrap();
        let (mean, std, outliers) = compute_numeric_stats(&casted).unwrap();
        assert!((mean - 4.0).abs() < 1e-10);
        assert!(std > 0.0);
        assert_eq!(outliers, 0);
    }

    #[test]
    fn compute_numeric_stats_empty_fails() {
        let series = Series::new("v".into(), Vec::<f64>::new());
        assert!(compute_numeric_stats(&series).is_err());
    }

    #[test]
    fn compute_numeric_stats_constant_series() {
        let series = Series::new("v".into(), &[5.0f64, 5.0, 5.0]);
        let (mean, std, outliers) = compute_numeric_stats(&series).unwrap();
        assert!((mean - 5.0).abs() < f64::EPSILON);
        assert!((std - 0.0).abs() < f64::EPSILON);
        assert_eq!(outliers, 0);
    }

    #[test]
    fn data_quality_report_serde_roundtrip() {
        let mut missingness = HashMap::new();
        missingness.insert("a".to_string(), 0.1);
        let report = DataQualityReport {
            kind: "data_quality".into(),
            iteration: 1,
            source_path: "/data/train.parquet".into(),
            rows_checked: 100,
            missingness,
            numeric_means: HashMap::new(),
            outlier_counts: HashMap::new(),
            drift_score: Some(0.05),
        };
        let json = serde_json::to_string(&report).unwrap();
        let restored: DataQualityReport = serde_json::from_str(&json).unwrap();
        assert_eq!(restored.rows_checked, 100);
        assert!((restored.drift_score.unwrap() - 0.05).abs() < f64::EPSILON);
    }

    #[test]
    fn inference_sample_serde_roundtrip() {
        let sample = InferenceSample {
            model_path: "/model".into(),
            target_column: "target".into(),
            rows: 3,
            predictions: vec![1.0, 2.0, 3.0],
            actuals: vec![1.1, 2.1, 3.1],
            iteration: 1,
        };
        let json = serde_json::to_string(&sample).unwrap();
        let restored: InferenceSample = serde_json::from_str(&json).unwrap();
        assert_eq!(restored.predictions.len(), 3);
        assert_eq!(restored.actuals.len(), 3);
    }

    #[test]
    fn apply_feature_spec_creates_interaction_column() {
        let df = df! {
            "a" => [1.0, 2.0, 3.0],
            "b" => [4.0, 5.0, 6.0],
            "target" => [10.0, 20.0, 30.0]
        }
        .unwrap();

        let spec = FeatureSpec {
            kind: "feature_spec".to_string(),
            iteration: 1,
            target_column: "target".to_string(),
            numeric_features: vec!["a".to_string(), "b".to_string()],
            categorical_features: vec![],
            normalization: "none".to_string(),
            interactions: vec![FeatureInteraction {
                name: "a_x_b".to_string(),
                left: "a".to_string(),
                right: "b".to_string(),
                op: "multiply".to_string(),
            }],
        };

        let result = apply_feature_spec(&df, &spec).unwrap();

        // Check interaction column exists
        assert!(result.column("a_x_b").is_ok());

        // Check values: 1*4=4, 2*5=10, 3*6=18
        let interaction = result.column("a_x_b").unwrap().f64().unwrap();
        let values: Vec<f64> = interaction.into_no_null_iter().collect();
        assert_eq!(values, vec![4.0, 10.0, 18.0]);
    }

    #[test]
    fn apply_feature_spec_standardizes_numeric_features() {
        let df = df! {
            "a" => [1.0, 2.0, 3.0, 4.0, 5.0],
            "target" => [10.0, 20.0, 30.0, 40.0, 50.0]
        }
        .unwrap();

        let spec = FeatureSpec {
            kind: "feature_spec".to_string(),
            iteration: 1,
            target_column: "target".to_string(),
            numeric_features: vec!["a".to_string()],
            categorical_features: vec![],
            normalization: "standardize".to_string(),
            interactions: vec![],
        };

        let result = apply_feature_spec(&df, &spec).unwrap();

        // Standardized values should have mean ~0 and std ~1
        let a_col = result.column("a").unwrap().f64().unwrap();
        let values: Vec<f64> = a_col.into_no_null_iter().collect();

        let mean: f64 = values.iter().sum::<f64>() / values.len() as f64;
        assert!(mean.abs() < 1e-10, "mean should be ~0, got {}", mean);

        let variance: f64 =
            values.iter().map(|v| (v - mean).powi(2)).sum::<f64>() / values.len() as f64;
        let std = variance.sqrt();
        assert!((std - 1.0).abs() < 1e-10, "std should be ~1, got {}", std);
    }

    #[test]
    fn apply_feature_spec_handles_add_operation() {
        let df = df! {
            "a" => [1.0, 2.0, 3.0],
            "b" => [10.0, 20.0, 30.0]
        }
        .unwrap();

        let spec = FeatureSpec {
            kind: "feature_spec".to_string(),
            iteration: 1,
            target_column: "target".to_string(),
            numeric_features: vec![],
            categorical_features: vec![],
            normalization: "none".to_string(),
            interactions: vec![FeatureInteraction {
                name: "a_plus_b".to_string(),
                left: "a".to_string(),
                right: "b".to_string(),
                op: "add".to_string(),
            }],
        };

        let result = apply_feature_spec(&df, &spec).unwrap();
        let col = result.column("a_plus_b").unwrap().f64().unwrap();
        let values: Vec<f64> = col.into_no_null_iter().collect();
        assert_eq!(values, vec![11.0, 22.0, 33.0]);
    }
}
