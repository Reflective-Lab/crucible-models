//! Crucible's `ProvenanceSource` marker.
//!
//! Migrated from the per-crate `ProvenanceSource` enum (which carried
//! one variant per workspace extension, duplicated across every
//! fact-emitting crate) to a tiny zero-sized type that implements
//! [`converge_pack::ProvenanceSource`]. The const surface is
//! unchanged: `CRUCIBLE_PROVENANCE.proposed_fact(...)` reads exactly
//! the same at call sites.
//!
//! The `converge-core` engine emits the uniform `suggestor.execute`
//! tracing span automatically around every `Suggestor::execute`
//! call, with the suggestor name + provenance string + dependency
//! keys as fields. Suggestors override `Suggestor::provenance()` to
//! return `CRUCIBLE_PROVENANCE.as_str()` so the engine's span
//! carries the right origin. The previous transitional
//! `suggestor_span` helper has been deleted — every crucible
//! Suggestor now relies on the engine span exclusively.

use converge_pack::ProvenanceSource;

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

#[cfg(test)]
mod tests {
    use super::*;
    use converge_pack::{ContextKey, TextPayload};

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
