//! Typed provenance source vocabulary for extension-originated facts.

use std::{error::Error, fmt, str::FromStr};

use converge_pack::{ContextKey, FactPayload, ProposalId, ProposedFact};
use serde::{Deserialize, Serialize};
use tracing::info_span;

pub const CRUCIBLE_PROVENANCE: ProvenanceSource = ProvenanceSource::Crucible;

#[derive(Debug, Clone, Copy, PartialEq, Eq, Hash, Serialize, Deserialize)]
#[serde(rename_all = "kebab-case")]
pub enum ProvenanceSource {
    Arbiter,
    Atelier,
    Crucible,
    Embassy,
    Ferrox,
    Manifold,
    Mnemos,
    Prism,
}

impl ProvenanceSource {
    pub const ALL: [Self; 8] = [
        Self::Arbiter,
        Self::Atelier,
        Self::Crucible,
        Self::Embassy,
        Self::Ferrox,
        Self::Manifold,
        Self::Mnemos,
        Self::Prism,
    ];

    #[must_use]
    pub const fn as_str(self) -> &'static str {
        match self {
            Self::Arbiter => "arbiter",
            Self::Atelier => "atelier",
            Self::Crucible => "crucible",
            Self::Embassy => "embassy",
            Self::Ferrox => "ferrox",
            Self::Manifold => "manifold",
            Self::Mnemos => "mnemos",
            Self::Prism => "prism",
        }
    }

    #[must_use]
    pub fn proposed_fact(
        self,
        key: ContextKey,
        id: impl Into<ProposalId>,
        payload: impl FactPayload + PartialEq,
    ) -> ProposedFact {
        ProposedFact::new(key, id, payload, self.as_str())
    }
}

impl fmt::Display for ProvenanceSource {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        f.write_str(self.as_str())
    }
}

impl FromStr for ProvenanceSource {
    type Err = UnknownProvenanceSource;

    fn from_str(value: &str) -> Result<Self, Self::Err> {
        match value {
            "arbiter" => Ok(Self::Arbiter),
            "atelier" => Ok(Self::Atelier),
            "crucible" => Ok(Self::Crucible),
            "embassy" => Ok(Self::Embassy),
            "ferrox" => Ok(Self::Ferrox),
            "manifold" => Ok(Self::Manifold),
            "mnemos" => Ok(Self::Mnemos),
            "prism" => Ok(Self::Prism),
            other => Err(UnknownProvenanceSource(other.to_string())),
        }
    }
}

#[derive(Debug, Clone, PartialEq, Eq)]
pub struct UnknownProvenanceSource(String);

impl fmt::Display for UnknownProvenanceSource {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        write!(f, "unknown provenance source: {}", self.0)
    }
}

impl Error for UnknownProvenanceSource {}

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
    fn provenance_sources_round_trip_through_strings() {
        for source in ProvenanceSource::ALL {
            let parsed: ProvenanceSource = source.as_str().parse().unwrap();
            assert_eq!(parsed, source);
            assert_eq!(source.to_string(), source.as_str());
        }
    }

    #[test]
    fn proposed_fact_uses_canonical_source_string() {
        let fact = CRUCIBLE_PROVENANCE.proposed_fact(
            ContextKey::Diagnostic,
            "diagnostic",
            TextPayload::new("content"),
        );

        assert_eq!(fact.provenance(), "crucible");
    }
}
