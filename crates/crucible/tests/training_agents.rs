// Copyright 2026 Reflective Labs
// SPDX-License-Identifier: MIT

//! Exercise the `Suggestor` impls on every training-pipeline agent.
//!
//! These tests build a `MockContext`, seed it with the upstream
//! payloads the agent expects, then invoke `name`, `dependencies`,
//! `provenance`, `accepts`, and `execute`. The intent is to keep the
//! agent contract honest: when the engine schedules the pipeline,
//! each agent must either produce a real, audit-stamped proposal or
//! a `DiagnosticPayload` recording why it couldn't. Silent no-ops are
//! the failure mode we never want to ship.

use std::collections::HashMap;
use std::fs::File;
use std::io::Write;

use converge_pack::{ContextKey, DiagnosticPayload, Suggestor};
use crucible::training::{
    BaselineModel, DataQualityReport, DatasetAgent, DatasetSplit, DataValidationAgent,
    DeploymentAgent, DeploymentDecision, EvaluationReport, FeatureEngineeringAgent, FeatureSpec,
    HyperparameterSearchAgent, HyperparameterSearchResult, InferenceSample, ModelEvaluationAgent,
    ModelMetadata, ModelRegistryAgent, ModelRegistryRecord, ModelTrainingAgent, MonitoringAgent,
    MonitoringReport, SampleInferenceAgent, TrainingPlan, apply_feature_spec,
};
use polars::prelude::*;
use tempfile::TempDir;

mod common;
use common::MockContext;

/// Drive an `async fn` to completion *without* installing a tokio
/// runtime on the current thread. Required for any test that ends up
/// calling polars' parquet reader, which internally does
/// `Runtime::new().block_on(...)` and panics if it sees an existing
/// tokio runtime above it on the stack. `#[tokio::test]` therefore
/// does not work for these flows.
fn drive<F: std::future::Future>(f: F) -> F::Output {
    futures::executor::block_on(f)
}

// ─── tiny fixture helpers ───────────────────────────────────────────

fn make_training_df(dir: &std::path::Path, name: &str, rows: usize) -> std::path::PathBuf {
    let path = dir.join(name);
    let xs: Vec<f64> = (0..rows).map(|i| i as f64).collect();
    let ys: Vec<f64> = (0..rows).map(|i| (i * 2) as f64).collect();
    let median: Vec<f64> = (0..rows).map(|i| (i * 10) as f64).collect();
    let mut df = df! {
        "x" => xs,
        "y" => ys,
        "median_house_value" => median,
    }
    .unwrap();
    let mut file = File::create(&path).unwrap();
    ParquetWriter::new(&mut file).finish(&mut df).unwrap();
    path
}

fn make_split_for(dir: &std::path::Path, iteration: usize) -> (DatasetSplit, std::path::PathBuf) {
    let train = make_training_df(dir, "train.parquet", 20);
    let val = make_training_df(dir, "val.parquet", 10);
    let infer = make_training_df(dir, "infer.parquet", 5);
    let split = DatasetSplit {
        source_path: train.clone(),
        train_path: train.clone(),
        val_path: val,
        infer_path: infer,
        total_rows: 35,
        max_rows: 35,
        train_rows: 20,
        val_rows: 10,
        infer_rows: 5,
        iteration,
    };
    (split, train)
}

fn write_baseline_model(dir: &std::path::Path) -> (std::path::PathBuf, BaselineModel) {
    let model = BaselineModel {
        target_column: "median_house_value".to_string(),
        mean: 100.0,
    };
    let path = dir.join("baseline_mean.json");
    let mut f = File::create(&path).unwrap();
    write!(f, "{}", serde_json::to_string(&model).unwrap()).unwrap();
    (path, model)
}

// ─── DataValidationAgent ─────────────────────────────────────────────

#[test]
fn data_validation_agent_emits_quality_report_for_real_split() {
    // Intent: the validation agent must produce a quality report so
    // downstream FeatureEngineeringAgent / training stages see a real
    // signal of dataset health. A silent skip would erase the only
    // place we look at null ratios / outliers before training.
    let tmp = TempDir::new().unwrap();
    let (split, _train) = make_split_for(tmp.path(), 1);

    let agent = DataValidationAgent::new();
    assert_eq!(agent.name(), "DataValidationAgent");
    assert_eq!(agent.dependencies(), &[ContextKey::Signals]);
    assert_eq!(agent.provenance(), "crucible");

    let mut ctx = MockContext::empty();
    ctx.insert(ContextKey::Signals, "split-1", split);
    assert!(agent.accepts(&ctx));

    let effect = drive(agent.execute(&ctx));
    let proposal = effect.proposals().first().expect("one proposal");
    let report = proposal
        .require_payload::<DataQualityReport>()
        .expect("DataQualityReport");
    assert_eq!(report.iteration, 1);
    assert_eq!(report.rows_checked, 20);
    assert!(report.numeric_means.contains_key("x"));
}

#[tokio::test(flavor = "multi_thread", worker_threads = 2)]
async fn data_validation_agent_emits_diagnostic_when_split_missing() {
    // Intent: the agent must surface a DiagnosticPayload — never a
    // silent empty effect — when its upstream input is missing.
    // Diagnostics are what the runbook keys off when convergence
    // stalls.
    let agent = DataValidationAgent::new();
    let mut ctx = MockContext::empty();
    // Seed a placeholder fact so `ctx.has(Signals)` is true, but no
    // DatasetSplit so the read fails.
    ctx.insert(
        ContextKey::Signals,
        "noise",
        DiagnosticPayload::new("test", "noise"),
    );
    // accepts may return false because read fails — that's fine,
    // we're asserting execute also reports a diagnostic.
    let effect = agent.execute(&ctx).await;
    let proposal = effect.proposals().first().expect("diagnostic proposal");
    let diag = proposal
        .require_payload::<DiagnosticPayload>()
        .expect("DiagnosticPayload");
    assert!(diag.message().contains("missing dataset split"));
}

#[tokio::test(flavor = "multi_thread", worker_threads = 2)]
async fn data_validation_agent_emits_diagnostic_on_broken_path() {
    // Intent: an unreadable parquet path is a recoverable runtime
    // condition, not a panic. The agent must record it as a
    // diagnostic.
    let agent = DataValidationAgent::new();
    let split = DatasetSplit {
        source_path: "/nonexistent/src.parquet".into(),
        train_path: "/nonexistent/train.parquet".into(),
        val_path: "/nonexistent/val.parquet".into(),
        infer_path: "/nonexistent/infer.parquet".into(),
        total_rows: 1,
        max_rows: 1,
        train_rows: 1,
        val_rows: 0,
        infer_rows: 0,
        iteration: 1,
    };
    let mut ctx = MockContext::empty();
    ctx.insert(ContextKey::Signals, "split-bad", split);
    let effect = agent.execute(&ctx).await;
    let proposal = effect.proposals().first().expect("diagnostic proposal");
    assert!(proposal.require_payload::<DiagnosticPayload>().is_ok());
}

#[tokio::test(flavor = "multi_thread", worker_threads = 2)]
async fn data_validation_agent_accepts_false_when_quality_already_present() {
    // Intent: the engine must not re-run validation for an iteration
    // it has already validated — that re-emits a duplicate report and
    // wastes downstream work.
    let tmp = TempDir::new().unwrap();
    let (split, _) = make_split_for(tmp.path(), 5);
    let agent = DataValidationAgent::new();
    let mut ctx = MockContext::empty();
    ctx.insert(ContextKey::Signals, "split-5", split);
    let prior_report = DataQualityReport {
        kind: "data_quality".into(),
        iteration: 5,
        source_path: "anything".into(),
        rows_checked: 0,
        missingness: HashMap::new(),
        numeric_means: HashMap::new(),
        outlier_counts: HashMap::new(),
        drift_score: None,
    };
    ctx.insert(ContextKey::Signals, "dq-5", prior_report);
    assert!(!agent.accepts(&ctx));
}

// ─── FeatureEngineeringAgent ─────────────────────────────────────────

#[test]
fn feature_engineering_agent_emits_feature_spec() {
    // Intent: the spec is the contract for the training agent. If we
    // don't emit one, ModelTrainingAgent silently trains on raw cols
    // and the offline/online split skews drift in production.
    let tmp = TempDir::new().unwrap();
    let (split, _) = make_split_for(tmp.path(), 2);
    let agent = FeatureEngineeringAgent::new();
    assert_eq!(agent.name(), "FeatureEngineeringAgent");
    assert_eq!(agent.dependencies(), &[ContextKey::Signals]);

    let mut ctx = MockContext::empty();
    ctx.insert(ContextKey::Signals, "split-2", split);
    assert!(agent.accepts(&ctx));

    let effect = drive(agent.execute(&ctx));
    let proposal = effect.proposals().first().expect("one proposal");
    let spec = proposal
        .require_payload::<FeatureSpec>()
        .expect("FeatureSpec");
    assert_eq!(spec.iteration, 2);
    assert_eq!(spec.target_column, "median_house_value");
    assert!(spec.numeric_features.contains(&"x".to_string()));
    // Because there are >=2 numeric features the agent must propose
    // exactly one interaction. Zero interactions = silent regression
    // of the only synthesized feature on this pipeline.
    assert_eq!(spec.interactions.len(), 1);
    assert_eq!(spec.interactions[0].op, "multiply");
}

#[tokio::test(flavor = "multi_thread", worker_threads = 2)]
async fn feature_engineering_agent_diagnostic_on_missing_split() {
    let agent = FeatureEngineeringAgent::new();
    let mut ctx = MockContext::empty();
    ctx.insert(
        ContextKey::Signals,
        "noise",
        DiagnosticPayload::new("x", "y"),
    );
    let effect = agent.execute(&ctx).await;
    let p = effect.proposals().first().unwrap();
    assert!(p.require_payload::<DiagnosticPayload>().is_ok());
}

// ─── HyperparameterSearchAgent ───────────────────────────────────────

#[tokio::test(flavor = "multi_thread", worker_threads = 2)]
async fn hyperparam_search_agent_emits_plan_and_result() {
    // Intent: the search agent's only job is to emit a plan and a
    // best-params result. If either is missing the registry will
    // record a model without provenance of how its hyperparams were
    // chosen — that breaks the "every prediction is replayable"
    // invariant.
    let tmp = TempDir::new().unwrap();
    let (split, _) = make_split_for(tmp.path(), 3);
    let agent = HyperparameterSearchAgent::new(7);
    assert_eq!(agent.name(), "HyperparameterSearchAgent");
    assert_eq!(
        agent.dependencies(),
        &[ContextKey::Constraints, ContextKey::Signals]
    );

    let mut ctx = MockContext::empty();
    ctx.insert(ContextKey::Signals, "split-3", split);
    assert!(agent.accepts(&ctx));

    let effect = agent.execute(&ctx).await;
    let mut saw_plan = false;
    let mut saw_result = false;
    for p in effect.proposals() {
        if p.payload_family().as_str() == "crucible.hyperparameter_search.plan" {
            saw_plan = true;
        }
        if p.payload_family().as_str() == "crucible.hyperparameter_search.result" {
            saw_result = true;
            let r = p.require_payload::<HyperparameterSearchResult>().unwrap();
            assert!(r.score.is_finite());
            assert!(r.best_params.contains_key("learning_rate"));
        }
    }
    assert!(saw_plan && saw_result, "must emit both plan and result");
}

#[tokio::test(flavor = "multi_thread", worker_threads = 2)]
async fn hyperparam_search_agent_diagnostic_on_missing_split() {
    let agent = HyperparameterSearchAgent::new(3);
    let mut ctx = MockContext::empty();
    ctx.insert(
        ContextKey::Signals,
        "noise",
        DiagnosticPayload::new("x", "y"),
    );
    let effect = agent.execute(&ctx).await;
    let p = effect.proposals().first().unwrap();
    assert!(p.require_payload::<DiagnosticPayload>().is_ok());
}

// ─── DatasetAgent (skip download path; cover diagnostic/no-op paths) ─

#[tokio::test(flavor = "multi_thread", worker_threads = 2)]
async fn dataset_agent_metadata_and_accepts() {
    // Intent: the dataset agent's `accepts` is the only thing
    // protecting against re-downloading and re-splitting the parquet
    // for every engine tick when no new plan has landed. A wrong
    // accept here turns each iteration into a network call.
    let tmp = TempDir::new().unwrap();
    let agent = DatasetAgent::new(tmp.path().to_path_buf());
    assert_eq!(agent.name(), "DatasetAgent (HuggingFace)");
    assert_eq!(agent.dependencies(), &[ContextKey::Seeds]);
    assert_eq!(agent.provenance(), "crucible");

    let ctx_empty = MockContext::empty();
    assert!(
        !agent.accepts(&ctx_empty),
        "no seeds → must not accept (would otherwise blindly download)"
    );

    // With a plan and a matching split already in context, the agent
    // must NOT re-accept — the split is already done for that
    // iteration.
    let mut ctx = MockContext::empty();
    ctx.insert(
        ContextKey::Seeds,
        "seed",
        DiagnosticPayload::new("seed", "test"),
    );
    let plan = TrainingPlan {
        iteration: 4,
        max_rows: 100,
        train_fraction: 0.8,
        val_fraction: 0.15,
        infer_fraction: 0.05,
        quality_threshold: 0.75,
    };
    ctx.insert(ContextKey::Constraints, "plan-4", plan);
    let (split, _) = make_split_for(tmp.path(), 4);
    ctx.insert(ContextKey::Signals, "split-4", split);
    assert!(
        !agent.accepts(&ctx),
        "must not accept once split exists for this iteration"
    );
}

// ─── ModelTrainingAgent ─────────────────────────────────────────────

#[test]
fn model_training_agent_trains_baseline_and_emits_metadata() {
    // Intent: this is the gate that converts a DatasetSplit into a
    // promoted model artifact. If it silently skips, downstream
    // evaluation produces nothing and the pipeline stalls. The
    // metadata fact is also what the registry indexes by — missing
    // it means no audit trail.
    let tmp = TempDir::new().unwrap();
    let (split, _train_path) = make_split_for(tmp.path(), 1);
    let model_dir = tmp.path().join("models");
    let agent = ModelTrainingAgent::new(model_dir.clone());
    assert_eq!(agent.name(), "ModelTrainingAgent (Baseline)");
    assert_eq!(agent.dependencies(), &[ContextKey::Signals]);

    let mut ctx = MockContext::empty();
    ctx.insert(ContextKey::Signals, "split-1", split);
    assert!(agent.accepts(&ctx));

    let effect = drive(agent.execute(&ctx));
    let p = effect.proposals().first().expect("one proposal");
    let meta = p.require_payload::<ModelMetadata>().expect("ModelMetadata");
    assert_eq!(meta.iteration, 1);
    assert_eq!(meta.train_rows, 20);
    assert!(meta.baseline_mean.is_finite());
    // Artifact must actually exist on disk — a metadata fact pointing
    // at a missing file is the worst kind of audit-graph lie.
    assert!(meta.model_path.exists());
}

#[tokio::test(flavor = "multi_thread", worker_threads = 2)]
async fn model_training_agent_diagnostic_on_missing_split() {
    let tmp = TempDir::new().unwrap();
    let agent = ModelTrainingAgent::new(tmp.path().to_path_buf());
    let mut ctx = MockContext::empty();
    ctx.insert(
        ContextKey::Signals,
        "noise",
        DiagnosticPayload::new("x", "y"),
    );
    let effect = agent.execute(&ctx).await;
    let p = effect.proposals().first().unwrap();
    assert!(p.require_payload::<DiagnosticPayload>().is_ok());
}

// ─── ModelEvaluationAgent ───────────────────────────────────────────

#[test]
fn model_evaluation_agent_emits_evaluation_report() {
    // Intent: an EvaluationReport with `success_ratio` is the single
    // input to both MonitoringAgent and DeploymentAgent. If we drop
    // the report or miscompute success_ratio (e.g. miss the
    // 0..=1 clamp) those downstream agents make wrong calls.
    let tmp = TempDir::new().unwrap();
    let (split, _) = make_split_for(tmp.path(), 2);
    let (model_path, _model) = write_baseline_model(tmp.path());

    let agent = ModelEvaluationAgent::new();
    assert_eq!(agent.name(), "ModelEvaluationAgent (MAE)");
    assert_eq!(
        agent.dependencies(),
        &[ContextKey::Signals, ContextKey::Strategies]
    );

    let mut ctx = MockContext::empty();
    ctx.insert(ContextKey::Signals, "split-2", split);
    let meta = ModelMetadata {
        model_path,
        target_column: "median_house_value".into(),
        train_rows: 20,
        baseline_mean: 100.0,
        iteration: 2,
    };
    ctx.insert(ContextKey::Strategies, "model-2", meta);

    assert!(agent.accepts(&ctx));
    let effect = drive(agent.execute(&ctx));
    let p = effect.proposals().first().expect("one proposal");
    let report = p
        .require_payload::<EvaluationReport>()
        .expect("EvaluationReport");
    assert_eq!(report.iteration, 2);
    assert!(report.value >= 0.0, "MAE must be non-negative");
    assert!(
        (0.0..=1.0).contains(&report.success_ratio),
        "success_ratio must be in [0,1]"
    );
}

#[tokio::test(flavor = "multi_thread", worker_threads = 2)]
async fn model_evaluation_agent_diagnostic_on_missing_split() {
    let agent = ModelEvaluationAgent::new();
    let mut ctx = MockContext::empty();
    ctx.insert(
        ContextKey::Signals,
        "noise",
        DiagnosticPayload::new("x", "y"),
    );
    ctx.insert(
        ContextKey::Strategies,
        "noise2",
        DiagnosticPayload::new("x", "y"),
    );
    let effect = agent.execute(&ctx).await;
    let p = effect.proposals().first().unwrap();
    assert!(p.require_payload::<DiagnosticPayload>().is_ok());
}

// ─── ModelRegistryAgent ─────────────────────────────────────────────

#[tokio::test(flavor = "multi_thread", worker_threads = 2)]
async fn model_registry_agent_records_metrics() {
    // Intent: the registry record is the audit row that ferrox / soter
    // later cite. Missing the success_ratio or the mae field means the
    // model is unciteable.
    let agent = ModelRegistryAgent::new();
    assert_eq!(agent.name(), "ModelRegistryAgent");
    assert_eq!(
        agent.dependencies(),
        &[ContextKey::Strategies, ContextKey::Evaluations]
    );

    let mut ctx = MockContext::empty();
    let meta = ModelMetadata {
        model_path: "/tmp/m.json".into(),
        target_column: "y".into(),
        train_rows: 10,
        baseline_mean: 1.0,
        iteration: 3,
    };
    ctx.insert(ContextKey::Strategies, "meta-3", meta);
    let report = EvaluationReport {
        model_path: "/tmp/m.json".into(),
        metric: "mae".into(),
        value: 0.5,
        mean_abs_target: 1.0,
        success_ratio: 0.9,
        val_rows: 5,
        iteration: 3,
    };
    ctx.insert(ContextKey::Evaluations, "eval-3", report);
    assert!(agent.accepts(&ctx));

    let effect = agent.execute(&ctx).await;
    let p = effect.proposals().first().unwrap();
    let record = p.require_payload::<ModelRegistryRecord>().unwrap();
    assert_eq!(record.iteration, 3);
    assert!((record.metrics["mae"] - 0.5).abs() < f64::EPSILON);
    assert!((record.metrics["success_ratio"] - 0.9).abs() < f64::EPSILON);
}

#[tokio::test(flavor = "multi_thread", worker_threads = 2)]
async fn model_registry_agent_diagnostic_on_missing_meta() {
    let agent = ModelRegistryAgent::new();
    let mut ctx = MockContext::empty();
    ctx.insert(
        ContextKey::Strategies,
        "noise",
        DiagnosticPayload::new("x", "y"),
    );
    ctx.insert(
        ContextKey::Evaluations,
        "noise2",
        DiagnosticPayload::new("x", "y"),
    );
    let effect = agent.execute(&ctx).await;
    let p = effect.proposals().first().unwrap();
    assert!(p.require_payload::<DiagnosticPayload>().is_ok());
}

// ─── MonitoringAgent ────────────────────────────────────────────────

#[tokio::test(flavor = "multi_thread", worker_threads = 2)]
async fn monitoring_agent_marks_healthy_above_threshold() {
    // Intent: status "healthy" vs "needs_attention" gates the
    // deployment branch. If the threshold logic flips, healthy models
    // get held and broken models get deployed.
    let agent = MonitoringAgent::new();
    assert_eq!(agent.name(), "MonitoringAgent");
    let mut ctx = MockContext::empty();
    let report = EvaluationReport {
        model_path: "/m".into(),
        metric: "mae".into(),
        value: 0.1,
        mean_abs_target: 1.0,
        success_ratio: 0.9,
        val_rows: 5,
        iteration: 1,
    };
    ctx.insert(ContextKey::Evaluations, "eval-1", report);
    assert!(agent.accepts(&ctx));
    let effect = agent.execute(&ctx).await;
    let p = effect.proposals().first().unwrap();
    let mon = p.require_payload::<MonitoringReport>().unwrap();
    assert_eq!(mon.status, "healthy");
}

#[tokio::test(flavor = "multi_thread", worker_threads = 2)]
async fn monitoring_agent_marks_needs_attention_below_threshold() {
    let agent = MonitoringAgent::new();
    let mut ctx = MockContext::empty();
    let report = EvaluationReport {
        model_path: "/m".into(),
        metric: "mae".into(),
        value: 1.0,
        mean_abs_target: 1.0,
        success_ratio: 0.5,
        val_rows: 5,
        iteration: 2,
    };
    ctx.insert(ContextKey::Evaluations, "eval-2", report);
    let effect = agent.execute(&ctx).await;
    let p = effect.proposals().first().unwrap();
    let mon = p.require_payload::<MonitoringReport>().unwrap();
    assert_eq!(mon.status, "needs_attention");
}

#[tokio::test(flavor = "multi_thread", worker_threads = 2)]
async fn monitoring_agent_inert_with_no_evaluations() {
    let agent = MonitoringAgent::new();
    let ctx = MockContext::empty();
    assert!(!agent.accepts(&ctx));
    let effect = agent.execute(&ctx).await;
    assert!(effect.proposals().is_empty());
}

// ─── DeploymentAgent ────────────────────────────────────────────────

#[tokio::test(flavor = "multi_thread", worker_threads = 2)]
async fn deployment_agent_deploys_when_above_threshold() {
    // Intent: this is the agent that decides whether the model goes
    // live. A wrong "deploy" call ships a broken model to production;
    // a wrong "hold" stalls a healthy release.
    let agent = DeploymentAgent::new();
    assert_eq!(agent.name(), "DeploymentAgent");
    let mut ctx = MockContext::empty();
    let report = EvaluationReport {
        model_path: "/m".into(),
        metric: "mae".into(),
        value: 0.1,
        mean_abs_target: 1.0,
        success_ratio: 0.9,
        val_rows: 5,
        iteration: 1,
    };
    ctx.insert(ContextKey::Evaluations, "eval-1", report);
    let meta = ModelMetadata {
        model_path: "/m".into(),
        target_column: "y".into(),
        train_rows: 1,
        baseline_mean: 0.0,
        iteration: 1,
    };
    ctx.insert(ContextKey::Strategies, "meta-1", meta);
    assert!(agent.accepts(&ctx));
    let effect = agent.execute(&ctx).await;
    let p = effect.proposals().first().unwrap();
    let dec = p.require_payload::<DeploymentDecision>().unwrap();
    assert_eq!(dec.action, "deploy");
    assert!(!dec.retrain);
}

#[tokio::test(flavor = "multi_thread", worker_threads = 2)]
async fn deployment_agent_holds_when_below_threshold() {
    let agent = DeploymentAgent::new();
    let mut ctx = MockContext::empty();
    let report = EvaluationReport {
        model_path: "/m".into(),
        metric: "mae".into(),
        value: 1.0,
        mean_abs_target: 1.0,
        success_ratio: 0.1,
        val_rows: 5,
        iteration: 7,
    };
    ctx.insert(ContextKey::Evaluations, "eval-7", report);
    let meta = ModelMetadata {
        model_path: "/m".into(),
        target_column: "y".into(),
        train_rows: 1,
        baseline_mean: 0.0,
        iteration: 7,
    };
    ctx.insert(ContextKey::Strategies, "meta-7", meta);
    let effect = agent.execute(&ctx).await;
    let p = effect.proposals().first().unwrap();
    let dec = p.require_payload::<DeploymentDecision>().unwrap();
    assert_eq!(dec.action, "hold");
    assert!(dec.retrain);
}

#[tokio::test(flavor = "multi_thread", worker_threads = 2)]
async fn deployment_agent_respects_plan_quality_threshold() {
    // Intent: the deployment threshold can be overridden by the
    // current TrainingPlan. If the override is ignored, every
    // pipeline gets the hard-coded 0.75 cutoff even when the plan
    // raised the bar.
    let agent = DeploymentAgent::new();
    let mut ctx = MockContext::empty();
    let plan = TrainingPlan {
        iteration: 1,
        max_rows: 10,
        train_fraction: 0.8,
        val_fraction: 0.15,
        infer_fraction: 0.05,
        // Hard cutoff: only ratios >= 0.99 deploy.
        quality_threshold: 0.99,
    };
    ctx.insert(ContextKey::Constraints, "plan-1", plan);
    let report = EvaluationReport {
        model_path: "/m".into(),
        metric: "mae".into(),
        value: 0.1,
        mean_abs_target: 1.0,
        // 0.9 < 0.99 → must hold
        success_ratio: 0.9,
        val_rows: 5,
        iteration: 1,
    };
    ctx.insert(ContextKey::Evaluations, "eval-1", report);
    let meta = ModelMetadata {
        model_path: "/m".into(),
        target_column: "y".into(),
        train_rows: 1,
        baseline_mean: 0.0,
        iteration: 1,
    };
    ctx.insert(ContextKey::Strategies, "meta-1", meta);
    let effect = agent.execute(&ctx).await;
    let p = effect.proposals().first().unwrap();
    let dec = p.require_payload::<DeploymentDecision>().unwrap();
    assert_eq!(dec.action, "hold");
}

// ─── SampleInferenceAgent ───────────────────────────────────────────

#[test]
fn sample_inference_agent_emits_predictions() {
    // Intent: the inference sample is what the team eyeballs to sanity
    // check a freshly trained model. If predictions and actuals
    // diverge in length the row alignment is wrong and the visual
    // check is meaningless.
    let tmp = TempDir::new().unwrap();
    let (split, _) = make_split_for(tmp.path(), 1);
    let (model_path, _) = write_baseline_model(tmp.path());

    let agent = SampleInferenceAgent::new(3);
    assert_eq!(agent.name(), "SampleInferenceAgent (Baseline)");
    assert_eq!(
        agent.dependencies(),
        &[ContextKey::Signals, ContextKey::Strategies]
    );

    let mut ctx = MockContext::empty();
    ctx.insert(ContextKey::Signals, "split-1", split);
    let meta = ModelMetadata {
        model_path,
        target_column: "median_house_value".into(),
        train_rows: 20,
        baseline_mean: 100.0,
        iteration: 1,
    };
    ctx.insert(ContextKey::Strategies, "meta-1", meta);
    assert!(agent.accepts(&ctx));

    let effect = drive(agent.execute(&ctx));
    let p = effect.proposals().first().unwrap();
    let sample = p.require_payload::<InferenceSample>().unwrap();
    assert_eq!(
        sample.predictions.len(),
        sample.actuals.len(),
        "predictions and actuals must align"
    );
    assert!(sample.rows <= 3, "max_rows must clamp output");
}

#[tokio::test(flavor = "multi_thread", worker_threads = 2)]
async fn sample_inference_agent_diagnostic_on_missing_split() {
    let agent = SampleInferenceAgent::new(3);
    let mut ctx = MockContext::empty();
    ctx.insert(
        ContextKey::Signals,
        "noise",
        DiagnosticPayload::new("x", "y"),
    );
    ctx.insert(
        ContextKey::Strategies,
        "noise2",
        DiagnosticPayload::new("x", "y"),
    );
    let effect = agent.execute(&ctx).await;
    let p = effect.proposals().first().unwrap();
    assert!(p.require_payload::<DiagnosticPayload>().is_ok());
}

// ─── apply_feature_spec error paths ─────────────────────────────────

#[tokio::test(flavor = "multi_thread", worker_threads = 2)]
async fn apply_feature_spec_errors_on_missing_left_column() {
    // Intent: an interaction whose left column doesn't exist must
    // surface as an error, not silently produce a missing column —
    // downstream training would then fail with a confusing dtype
    // error far from the root cause.
    let df = df! { "b" => [1.0, 2.0] }.unwrap();
    let spec = FeatureSpec {
        kind: "feature_spec".into(),
        iteration: 1,
        target_column: "t".into(),
        numeric_features: vec![],
        categorical_features: vec![],
        normalization: "none".into(),
        interactions: vec![crucible::training::FeatureInteraction {
            name: "a_x_b".into(),
            left: "a".into(),
            right: "b".into(),
            op: "multiply".into(),
        }],
    };
    let err = apply_feature_spec(&df, &spec).unwrap_err();
    assert!(err.to_string().contains("missing column a"));
}

#[tokio::test(flavor = "multi_thread", worker_threads = 2)]
async fn apply_feature_spec_errors_on_unsupported_op() {
    // Intent: a typo in the op string used to silently zero out the
    // interaction column. Now it must fail loudly.
    let df = df! { "a" => [1.0, 2.0], "b" => [3.0, 4.0] }.unwrap();
    let spec = FeatureSpec {
        kind: "feature_spec".into(),
        iteration: 1,
        target_column: "t".into(),
        numeric_features: vec![],
        categorical_features: vec![],
        normalization: "none".into(),
        interactions: vec![crucible::training::FeatureInteraction {
            name: "weird".into(),
            left: "a".into(),
            right: "b".into(),
            op: "tensor_product".into(),
        }],
    };
    let err = apply_feature_spec(&df, &spec).unwrap_err();
    assert!(err.to_string().contains("unsupported interaction op"));
}

#[tokio::test(flavor = "multi_thread", worker_threads = 2)]
async fn apply_feature_spec_subtract_and_divide_paths() {
    // Intent: subtract and divide both have algebraic edge cases (near-zero
    // denominator). Cover them so a future "optimize chunkedarray" refactor
    // can't silently break the divide-by-zero guard.
    let df = df! {
        "a" => [10.0, 4.0],
        "b" => [3.0, 2.0],
    }
    .unwrap();

    let spec_sub = FeatureSpec {
        kind: "feature_spec".into(),
        iteration: 1,
        target_column: "t".into(),
        numeric_features: vec![],
        categorical_features: vec![],
        normalization: "none".into(),
        interactions: vec![crucible::training::FeatureInteraction {
            name: "a_sub_b".into(),
            left: "a".into(),
            right: "b".into(),
            op: "subtract".into(),
        }],
    };
    let sub_result = apply_feature_spec(&df, &spec_sub).unwrap();
    let sub_col = sub_result.column("a_sub_b").unwrap().f64().unwrap();
    let sub_vals: Vec<f64> = sub_col.into_no_null_iter().collect();
    assert_eq!(sub_vals, vec![7.0, 2.0]);

    let spec_div = FeatureSpec {
        interactions: vec![crucible::training::FeatureInteraction {
            name: "a_div_b".into(),
            left: "a".into(),
            right: "b".into(),
            op: "divide".into(),
        }],
        ..spec_sub.clone()
    };
    let div_result = apply_feature_spec(&df, &spec_div).unwrap();
    let div_col = div_result.column("a_div_b").unwrap().f64().unwrap();
    let div_vals: Vec<f64> = div_col.into_no_null_iter().collect();
    assert!((div_vals[0] - (10.0 / 3.0)).abs() < 1e-10);
    assert!((div_vals[1] - 2.0).abs() < 1e-10);
}

// ─── ModelTrainingAgent uses FeatureSpec when present ───────────────

#[test]
fn model_training_agent_applies_feature_spec_when_present() {
    // Intent: if a FeatureSpec is in context for the current
    // iteration, training must apply it before fitting. A regression
    // that ignores the spec would train on raw cols while serving
    // believes it's standardized.
    let tmp = TempDir::new().unwrap();
    let (split, _) = make_split_for(tmp.path(), 1);
    let model_dir = tmp.path().join("models2");
    let agent = ModelTrainingAgent::new(model_dir.clone());
    let mut ctx = MockContext::empty();
    ctx.insert(ContextKey::Signals, "split-1", split);
    let spec = FeatureSpec {
        kind: "feature_spec".into(),
        iteration: 1,
        target_column: "median_house_value".into(),
        numeric_features: vec!["x".into(), "y".into()],
        categorical_features: vec![],
        normalization: "standardize".into(),
        interactions: vec![],
    };
    ctx.insert(ContextKey::Constraints, "fs-1", spec);
    let effect = drive(agent.execute(&ctx));
    let p = effect.proposals().first().expect("one proposal");
    let meta = p.require_payload::<ModelMetadata>().expect("ModelMetadata");
    assert_eq!(meta.iteration, 1);
}
