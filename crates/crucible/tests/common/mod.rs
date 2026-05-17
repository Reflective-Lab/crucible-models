// Copyright 2026 Reflective Labs
// SPDX-License-Identifier: MIT

//! Shared test scaffolding for crucible integration tests.
//!
//! Builds a `MockContext` that satisfies the `converge_pack::Context`
//! trait so we can exercise Suggestor::accepts / Suggestor::execute
//! end-to-end without depending on the live engine.

use std::collections::HashMap;

use converge_pack::{
    Context, ContextFact, ContextKey, FactActor, FactActorKind, FactLocalTrace, FactPayload,
    FactPromotionRecord, FactTraceLink, FactValidationSummary,
};
use converge_pack::{ContentHash, Timestamp};

/// Minimal `Context` implementation that holds owned `ContextFact`s in
/// a per-key vector. Mirrors the shape used in
/// `converge-pack`'s own `context.rs` test scaffolding so that any
/// behavioural divergence between the two mocks is intentional.
pub struct MockContext {
    facts: HashMap<ContextKey, Vec<ContextFact>>,
}

impl MockContext {
    pub fn empty() -> Self {
        Self {
            facts: HashMap::new(),
        }
    }

    /// Build a deterministic projection record. The Suggestors under
    /// test never inspect the promotion record; we just need a real
    /// `ContextFact`.
    fn projection_record() -> FactPromotionRecord {
        FactPromotionRecord::new_projection(
            "test-gate",
            ContentHash::zero(),
            FactActor::new_projection("test-actor", FactActorKind::System),
            FactValidationSummary::default(),
            Vec::new(),
            FactTraceLink::Local(FactLocalTrace::new_projection(
                "trace-test",
                "span-test",
                None,
                true,
            )),
            Timestamp::epoch(),
        )
    }

    /// Insert a fact under a key, building it from a typed payload.
    pub fn insert<T>(&mut self, key: ContextKey, id: &str, payload: T)
    where
        T: FactPayload + PartialEq,
    {
        let fact = ContextFact::new_projection(
            key,
            id,
            payload,
            Self::projection_record(),
            Timestamp::epoch(),
        );
        self.facts.entry(key).or_default().push(fact);
    }
}

impl Context for MockContext {
    fn has(&self, key: ContextKey) -> bool {
        self.facts.get(&key).is_some_and(|v| !v.is_empty())
    }

    fn get(&self, key: ContextKey) -> &[ContextFact] {
        self.facts.get(&key).map_or(&[], Vec::as_slice)
    }
}
