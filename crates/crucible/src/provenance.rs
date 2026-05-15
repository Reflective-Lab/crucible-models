//! Crucible's `ProvenanceSource` marker.
//!
//! Migrated from the per-crate `ProvenanceSource` enum (which carried
//! one variant per workspace extension, duplicated across every
//! fact-emitting crate) to a tiny zero-sized type that implements
//! [`converge_pack::ProvenanceSource`]. The const surface is
//! unchanged: `CRUCIBLE_PROVENANCE.proposed_fact(...)` reads exactly
//! the same at call sites.
//!
//! The `converge-core` engine now emits a uniform `suggestor.execute`
//! tracing span automatically around every `Suggestor::execute` call,
//! with the suggestor name + provenance string + dependency keys as
//! fields. Suggestors override `Suggestor::provenance()` to return
//! `CRUCIBLE_PROVENANCE.as_str()` so the engine's span carries the
//! right origin without each crate hand-rolling its own span helper.
//!
//! A `pub(crate)` `suggestor_span` legacy helper remains for the
//! training-pipeline agents in [`crate::training`] until their per-agent
//! migration to the engine span lands; it is intentionally not exposed
//! on the public surface.

use converge_pack::{ContextKey, ProvenanceSource};
use tracing::info_span;

/// Marker type identifying crucible-emitted facts.
#[derive(Debug, Clone, Copy, PartialEq, Eq, Hash)]
pub struct Crucible;

impl ProvenanceSource for Crucible {
    fn as_str(&self) -> &'static str {
        "crucible"
    }
}

/// Canonical provenance constant for crucible. Use it to construct
/// proposals: `CRUCIBLE_PROVENANCE.proposed_fact(key, id, payload)`.
pub const CRUCIBLE_PROVENANCE: Crucible = Crucible;

/// Legacy per-crate suggestor span helper.
///
/// Internal-only. The engine emits the canonical `suggestor.execute`
/// span automatically; this helper exists only so the training-pipeline
/// agents in [`crate::training`] keep compiling until their per-agent
/// migration to the engine span lands. New code does not call this.
pub(crate) fn suggestor_span(
    name: &str,
    input_key: ContextKey,
    output_key: ContextKey,
    input_count: usize,
) -> tracing::Span {
    info_span!(
        "crucible.suggestor.execute",
        provenance = CRUCIBLE_PROVENANCE.as_str(),
        suggestor = name,
        input_key = ?input_key,
        output_key = ?output_key,
        input_count
    )
}

#[cfg(test)]
mod tests {
    use super::*;
    use converge_pack::TextPayload;

    #[test]
    fn provenance_string_is_stable() {
        assert_eq!(CRUCIBLE_PROVENANCE.as_str(), "crucible");
    }

    #[test]
    fn proposed_fact_carries_crucible_provenance() {
        let fact = CRUCIBLE_PROVENANCE.proposed_fact(
            ContextKey::Diagnostic,
            "diagnostic",
            TextPayload::new("content"),
        );
        assert_eq!(fact.provenance(), "crucible");
    }
}
