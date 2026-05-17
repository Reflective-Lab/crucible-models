// Copyright 2026 Reflective Labs
// SPDX-License-Identifier: MIT

//! End-to-end exercise of `ClassifierSuggestor` over a real
//! `RandomForestModel`, against a hand-built `MockContext`. Verifies
//! the full inference path that ships in production: fact-in, model
//! forward, `ClassPredictionPayload`-out, including the audit-only
//! `ExecutionIdentity` that ferrox / soter rely on.

use std::sync::Arc;

use converge_pack::{ContextKey, Suggestor};
use crucible::{
    ClassifierModel, ClassificationFeaturesPayload, ClassPredictionPayload,
    DecisionTreeClassifierSuggestor, RandomForestClassifierSuggestor, RandomForestConfig,
    RandomForestModel,
    fixtures::loan_default,
};
use crucible::{DecisionTreeClassifier, DecisionTreeConfig};

mod common;
use common::MockContext;

fn small_rf() -> Arc<RandomForestModel> {
    let (features, labels) = loan_default(120, 31);
    let model = RandomForestModel::train(
        &RandomForestConfig {
            n_trees: 5,
            max_depth: Some(4),
            min_weight_split: 2.0,
            random_seed: 17,
        },
        &features,
        &labels,
    )
    .expect("RF train");
    Arc::new(model)
}

fn small_dt() -> Arc<DecisionTreeClassifier> {
    let (features, labels) = loan_default(80, 19);
    let model = DecisionTreeClassifier::train(
        &DecisionTreeConfig {
            max_depth: Some(4),
            min_weight_split: 2.0,
        },
        &features,
        &labels,
    )
    .expect("DT train");
    Arc::new(model)
}

#[tokio::test]
async fn rf_suggestor_emits_one_prediction_per_features_fact() {
    // Intent: a SignalSuggestor that drops samples silently breaks
    // every downstream evaluation gate that pairs predictions to
    // inputs by index — predictions must be 1-to-1 with features.
    let model = small_rf();
    let n_features = 6;
    let sugg = RandomForestClassifierSuggestor::new(
        "rf-test",
        model,
        ContextKey::Seeds,
        ContextKey::Hypotheses,
    );

    let mut ctx = MockContext::empty();
    ctx.insert(
        ContextKey::Seeds,
        "row-1",
        ClassificationFeaturesPayload::new(vec![700.0, 50_000.0, 0.3, 10_000.0, 5.0, 0.0]),
    );
    ctx.insert(
        ContextKey::Seeds,
        "row-2",
        ClassificationFeaturesPayload::new(vec![500.0, 30_000.0, 0.7, 100_000.0, 1.0, 1.0]),
    );
    let _ = n_features;

    assert!(sugg.accepts(&ctx), "accepts must be true when features present");
    assert_eq!(sugg.name(), "rf-test");
    assert_eq!(sugg.dependencies(), &[ContextKey::Seeds]);
    assert_eq!(sugg.provenance(), "crucible");

    let effect = sugg.execute(&ctx).await;
    let proposals = effect.proposals();
    assert_eq!(
        proposals.len(),
        2,
        "one prediction per features fact; got {}",
        proposals.len()
    );

    for proposal in proposals {
        let pred = proposal
            .require_payload::<ClassPredictionPayload>()
            .expect("payload must be ClassPredictionPayload");
        assert!(
            (pred.class_probabilities().iter().sum::<f64>() - 1.0).abs() < 1e-9,
            "probabilities must sum to 1; got {:?}",
            pred.class_probabilities()
        );
        // Argmax(probabilities) must equal predicted_class — anything
        // else corrupts rank-by-score consumers.
        let argmax = pred
            .class_probabilities()
            .iter()
            .enumerate()
            .max_by(|a, b| a.1.partial_cmp(b.1).unwrap())
            .map(|(i, _)| i)
            .unwrap();
        assert_eq!(argmax, pred.predicted_class());
    }
}

#[tokio::test]
async fn rf_suggestor_skips_empty_feature_vectors() {
    // Intent: an empty feature row would otherwise call into
    // `Array2::from_shape_vec((1, 0), vec![])` which is valid shape
    // but produces NaN-laden votes — the warn-and-skip branch protects
    // downstream consumers from getting garbage probabilities.
    let model = small_rf();
    let sugg = RandomForestClassifierSuggestor::new(
        "rf-skip-empty",
        model,
        ContextKey::Seeds,
        ContextKey::Hypotheses,
    );

    let mut ctx = MockContext::empty();
    ctx.insert(
        ContextKey::Seeds,
        "row-empty",
        ClassificationFeaturesPayload::new(vec![]),
    );

    let effect = sugg.execute(&ctx).await;
    assert!(
        effect.proposals().is_empty(),
        "empty feature vector must be skipped, not emit a proposal"
    );
}

#[tokio::test]
async fn rf_suggestor_emits_finite_probabilities_for_extreme_inputs() {
    // Intent: extreme-but-correctly-shaped inputs must still produce
    // finite probabilities — a NaN-producing regression would feed
    // poison to every downstream rank-by-score consumer.
    let model = small_rf();
    let sugg = RandomForestClassifierSuggestor::new(
        "rf-extreme",
        model,
        ContextKey::Seeds,
        ContextKey::Hypotheses,
    );

    let mut ctx = MockContext::empty();
    // Extreme but correctly-shaped (6 feature) input.
    ctx.insert(
        ContextKey::Seeds,
        "row-extreme-hi",
        ClassificationFeaturesPayload::new(vec![1e10, 1e10, 1e10, 1e10, 1e10, 1.0]),
    );
    ctx.insert(
        ContextKey::Seeds,
        "row-extreme-lo",
        ClassificationFeaturesPayload::new(vec![-1e10, -1e10, -1e10, -1e10, -1e10, 0.0]),
    );

    let effect = sugg.execute(&ctx).await;
    assert_eq!(effect.proposals().len(), 2);
    for proposal in effect.proposals() {
        let pred = proposal
            .require_payload::<ClassPredictionPayload>()
            .expect("payload must be ClassPredictionPayload");
        let s: f64 = pred.class_probabilities().iter().sum();
        assert!(s.is_finite(), "no NaN/Inf probabilities allowed");
        assert!((s - 1.0).abs() < 1e-9, "row sum != 1.0: got {s}");
    }
}

#[tokio::test]
async fn dt_suggestor_carries_execution_identity_into_payload() {
    // Intent: a serde change that drops `execution_identity` breaks
    // every audit replay. The suggestor must always thread the
    // model's identity through to the emitted prediction payload.
    let model = small_dt();
    let expected_identity = model.execution_identity();
    let sugg = DecisionTreeClassifierSuggestor::new(
        "dt-identity",
        model,
        ContextKey::Seeds,
        ContextKey::Hypotheses,
    );

    let mut ctx = MockContext::empty();
    ctx.insert(
        ContextKey::Seeds,
        "row-1",
        ClassificationFeaturesPayload::new(vec![700.0, 80_000.0, 0.2, 25_000.0, 4.0, 0.0]),
    );

    let effect = sugg.execute(&ctx).await;
    let proposal = effect.proposals().first().expect("one proposal");
    let pred = proposal
        .require_payload::<ClassPredictionPayload>()
        .expect("ClassPredictionPayload");
    assert_eq!(
        pred.execution_identity(),
        &expected_identity,
        "the payload identity must equal the model identity"
    );
}

#[tokio::test]
async fn rf_suggestor_accepts_returns_false_for_wrong_payload_family() {
    // Intent: dispatching `execute` on a fact whose payload family
    // does not match wastes CPU; `accepts` is the engine's fast-path
    // skip. If we accept the wrong family the engine pays the cost of
    // attempting decode for every unrelated fact.
    let model = small_rf();
    let sugg = RandomForestClassifierSuggestor::new(
        "rf-accepts-filter",
        model,
        ContextKey::Seeds,
        ContextKey::Hypotheses,
    );

    let mut ctx = MockContext::empty();
    // Wrong family: ClassPredictionPayload under the input key.
    ctx.insert(
        ContextKey::Seeds,
        "wrong",
        ClassPredictionPayload::new(
            0,
            vec![1.0, 0.0],
            converge_pack::ExecutionIdentity::non_native("p", "v", "b", String::new()),
        ),
    );

    assert!(
        !sugg.accepts(&ctx),
        "accepts must reject when no ClassificationFeaturesPayload present"
    );
    let effect = sugg.execute(&ctx).await;
    assert!(
        effect.proposals().is_empty(),
        "execute on wrong family must produce zero proposals"
    );
}

#[tokio::test]
async fn rf_suggestor_no_features_in_context_is_inert() {
    // Intent: an empty context must never coerce the suggestor into
    // emitting empty / placeholder predictions; downstream audit will
    // surface those as real predictions for rows that never existed.
    let model = small_rf();
    let sugg = RandomForestClassifierSuggestor::new(
        "rf-empty-ctx",
        model,
        ContextKey::Seeds,
        ContextKey::Hypotheses,
    );

    let ctx = MockContext::empty();
    assert!(!sugg.accepts(&ctx));
    let effect = sugg.execute(&ctx).await;
    assert!(effect.proposals().is_empty());
}
