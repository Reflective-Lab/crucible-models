//! Crucible-local typed payloads for `ProposedFact` content.
//!
//! These types implement [`converge_pack::FactPayload`] with stable
//! `(FAMILY, VERSION)` identifiers in the `crucible.*` namespace.
//! Crucible Suggestors read and write only these typed payloads; the
//! string-content escape hatch on `ProposedFact` is reserved for
//! diagnostics and wire / replay borders.

use converge_pack::{ExecutionIdentity, FactPayload};
use serde::{Deserialize, Serialize};

/// Feature vector for a single sample to be classified.
///
/// Suggestors that consume features for inference (e.g.
/// [`crate::DecisionTreeClassifierSuggestor`],
/// [`crate::RandomForestClassifierSuggestor`]) accept this payload as
/// input. `features` is a row of `n_features` `f64` values; the
/// feature semantics — credit_score, debt_to_income, etc. — are
/// owned by the upstream pack or product that generated them.
#[derive(Debug, Clone, PartialEq, Serialize, Deserialize)]
#[serde(deny_unknown_fields)]
pub struct ClassificationFeaturesPayload {
    pub features: Vec<f64>,
}

impl ClassificationFeaturesPayload {
    #[must_use]
    pub fn new(features: Vec<f64>) -> Self {
        Self { features }
    }

    #[must_use]
    pub fn features(&self) -> &[f64] {
        &self.features
    }
}

impl FactPayload for ClassificationFeaturesPayload {
    const FAMILY: &'static str = "crucible.classification.features";
    const VERSION: u16 = 1;
}

/// Class prediction with per-class probabilities for a single sample.
///
/// Emitted by Crucible classifier Suggestors. `predicted_class` is the
/// argmax of `class_probabilities`; `class_probabilities` sums to 1.0
/// within floating-point tolerance. `execution_identity` records the
/// producer crate, backend library (e.g. `linfa-trees-v0.8`), and the
/// serialized runtime config (hyperparameters) of the model that
/// emitted this prediction, mirroring the audit pattern already in
/// use by Ferrox and Soter.
///
/// Version bumped from `1` to `2` on 2026-05-15 when
/// `execution_identity` became a required field. v1 had no external
/// consumers; the rename is breaking only for in-tree call sites.
#[derive(Debug, Clone, PartialEq, Serialize, Deserialize)]
#[serde(deny_unknown_fields)]
pub struct ClassPredictionPayload {
    pub predicted_class: usize,
    pub class_probabilities: Vec<f64>,
    pub execution_identity: ExecutionIdentity,
}

impl ClassPredictionPayload {
    #[must_use]
    pub fn new(
        predicted_class: usize,
        class_probabilities: Vec<f64>,
        execution_identity: ExecutionIdentity,
    ) -> Self {
        Self {
            predicted_class,
            class_probabilities,
            execution_identity,
        }
    }

    #[must_use]
    pub fn predicted_class(&self) -> usize {
        self.predicted_class
    }

    #[must_use]
    pub fn class_probabilities(&self) -> &[f64] {
        &self.class_probabilities
    }

    #[must_use]
    pub fn execution_identity(&self) -> &ExecutionIdentity {
        &self.execution_identity
    }
}

impl FactPayload for ClassPredictionPayload {
    const FAMILY: &'static str = "crucible.classification.prediction";
    const VERSION: u16 = 2;
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn features_payload_round_trips() {
        let p = ClassificationFeaturesPayload::new(vec![1.0, 2.0, 3.0]);
        assert_eq!(p.features(), &[1.0, 2.0, 3.0]);
        let json = serde_json::to_string(&p).unwrap();
        let back: ClassificationFeaturesPayload = serde_json::from_str(&json).unwrap();
        assert_eq!(back, p);
    }

    fn fixture_identity() -> ExecutionIdentity {
        ExecutionIdentity::non_native(
            "test-producer",
            "0.0.0",
            "fixture-backend",
            "{\"hyperparam\":42}",
        )
    }

    #[test]
    fn prediction_payload_round_trips() {
        let p = ClassPredictionPayload::new(1, vec![0.2, 0.8], fixture_identity());
        assert_eq!(p.predicted_class(), 1);
        assert_eq!(p.class_probabilities(), &[0.2, 0.8]);
        assert_eq!(p.execution_identity().backend, "fixture-backend");
        let json = serde_json::to_string(&p).unwrap();
        let back: ClassPredictionPayload = serde_json::from_str(&json).unwrap();
        assert_eq!(back, p);
    }

    #[test]
    fn payload_family_strings_are_stable() {
        assert_eq!(
            ClassificationFeaturesPayload::FAMILY,
            "crucible.classification.features"
        );
        assert_eq!(
            ClassPredictionPayload::FAMILY,
            "crucible.classification.prediction"
        );
        assert_eq!(ClassificationFeaturesPayload::VERSION, 1);
        assert_eq!(ClassPredictionPayload::VERSION, 2);
    }
}
