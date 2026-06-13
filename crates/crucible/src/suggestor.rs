//! Crucible classifier Suggestor: bridges a trained
//! [`ClassifierModel`] into the Convergence Engine via the typed
//! payload contract.
//!
//! The Suggestor reads `ClassificationFeaturesPayload`s from its
//! configured input `ContextKey`, runs `predict_proba` on the loaded
//! model, and emits a `ClassPredictionPayload` per input under its
//! configured output key with [`CRUCIBLE_PROVENANCE`]. The
//! `converge-core` engine emits a uniform `suggestor.execute`
//! tracing span around every `Suggestor::execute` call carrying
//! the suggestor's `name()`, `provenance()`, and `dependencies()`
//! as fields.
//!
//! `ClassifierSuggestor` is generic over the concrete model type. The
//! re-exported aliases [`crate::DecisionTreeClassifierSuggestor`] and
//! [`crate::RandomForestClassifierSuggestor`] pick concrete models;
//! new packs only need to define an alias.

use std::sync::Arc;

use async_trait::async_trait;
use converge_pack::{
    AgentEffect, Context, ContextKey, FactPayload, ProposalId, Provenance, ProvenanceSource,
    Suggestor,
};
use ndarray::Array2;
use tracing::warn;

use crate::model::ClassifierModel;
use crate::provenance::CRUCIBLE_PROVENANCE;
use crate::types::{ClassPredictionPayload, ClassificationFeaturesPayload};

/// Generic inference Suggestor for any [`ClassifierModel`].
pub struct ClassifierSuggestor<M: ClassifierModel> {
    suggestor_name: &'static str,
    model: Arc<M>,
    input_key: ContextKey,
    output_key: ContextKey,
    dependencies: [ContextKey; 1],
}

impl<M: ClassifierModel> ClassifierSuggestor<M> {
    /// Create a new classifier Suggestor.
    ///
    /// - `suggestor_name` — stable name surfaced in tracing spans and
    ///   Formation discovery.
    /// - `model` — a trained [`ClassifierModel`]; shared via `Arc` so
    ///   the same artifact can be wired into multiple Formations.
    /// - `input_key` — Suggestor reads
    ///   `ClassificationFeaturesPayload`s from this key.
    /// - `output_key` — Suggestor writes `ClassPredictionPayload`s
    ///   under this key.
    pub fn new(
        suggestor_name: &'static str,
        model: Arc<M>,
        input_key: ContextKey,
        output_key: ContextKey,
    ) -> Self {
        Self {
            suggestor_name,
            model,
            input_key,
            output_key,
            dependencies: [input_key],
        }
    }
}

#[async_trait]
impl<M: ClassifierModel + Send + Sync + 'static> Suggestor for ClassifierSuggestor<M> {
    fn name(&self) -> &str {
        self.suggestor_name
    }

    fn dependencies(&self) -> &[ContextKey] {
        &self.dependencies
    }

    fn accepts(&self, ctx: &dyn Context) -> bool {
        ctx.get(self.input_key)
            .iter()
            .any(|fact| fact.payload::<ClassificationFeaturesPayload>().is_some())
    }

    fn provenance(&self) -> Provenance {
        Provenance::from(CRUCIBLE_PROVENANCE.as_str())
    }

    async fn execute(&self, ctx: &dyn Context) -> AgentEffect {
        // No local span: `converge-core`'s engine emits the
        // `suggestor.execute` span automatically, with this
        // Suggestor's `name()`, `provenance()`, and `dependencies()`
        // as fields. Replaces the previous per-crate `suggestor_span`
        // helper.
        let inputs = ctx.get(self.input_key);
        let mut proposals = Vec::new();

        for fact in inputs {
            let Some(features_payload) = fact.payload::<ClassificationFeaturesPayload>() else {
                continue;
            };
            let feats = features_payload.features();
            if feats.is_empty() {
                warn!(
                    suggestor = self.suggestor_name,
                    fact_id = %fact.id(),
                    "skipping empty feature vector"
                );
                continue;
            }

            let n_features = feats.len();
            let row = match Array2::<f64>::from_shape_vec((1, n_features), feats.to_vec()) {
                Ok(arr) => arr,
                Err(err) => {
                    warn!(
                        suggestor = self.suggestor_name,
                        error = %err,
                        "feature reshape failed"
                    );
                    continue;
                }
            };

            let proba = match self.model.predict_proba(&row) {
                Ok(p) => p,
                Err(err) => {
                    warn!(
                        suggestor = self.suggestor_name,
                        error = %err,
                        "predict_proba failed"
                    );
                    continue;
                }
            };

            let probs: Vec<f64> = proba.row(0).to_vec();
            let predicted = argmax(&probs);

            let payload =
                ClassPredictionPayload::new(predicted, probs, self.model.execution_identity());
            let id = ProposalId::new(format!("{}-{}", self.suggestor_name, fact.id()));
            let proposal =
                CRUCIBLE_PROVENANCE.proposed_fact_for(fact, self.output_key, id, payload);
            proposals.push(proposal);
        }

        AgentEffect::with_proposals(proposals)
    }
}

fn argmax(probs: &[f64]) -> usize {
    let mut best_idx = 0_usize;
    let mut best_p = f64::NEG_INFINITY;
    for (i, &p) in probs.iter().enumerate() {
        if p > best_p {
            best_p = p;
            best_idx = i;
        }
    }
    best_idx
}

/// Compile-time assertion stub: `ProposedFact` constructor accepts
/// our `ClassPredictionPayload` via the `FactPayload` trait.
#[allow(dead_code)]
const _: fn() = || {
    fn assert_payload<T: FactPayload + PartialEq>() {}
    assert_payload::<ClassPredictionPayload>();
    assert_payload::<ClassificationFeaturesPayload>();
};

// Integration tests for `ClassifierSuggestor` live alongside the
// loan-application showcase in atelier-showcase (slice 2e). They
// exercise the full Convergence-engine path with a real `Context`
// implementation; constructing the necessary `ContextFact` plumbing
// here would duplicate the kernel's test scaffolding for no payoff.
