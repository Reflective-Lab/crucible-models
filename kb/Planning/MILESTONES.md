> **Archived 2026-07-02** — active milestone tracking moved to Linear (Reflective team).
> This file is kept for historical context only. Do not add new items here.

---
source: human
date: 2026-05-15
---
# Milestones — crucible-models

The original v0.1.0 → v1.0.0 plan was written when crucible was a stub.
Major boundary and library decisions in May 2026 reshaped it. This page
records what shipped and what remains pull-driven.

## v0.1.0 — Scaffold

**Status:** shipped 2026-05-09.

- [x] Workspace `Cargo.toml` with Burn 0.20 and Converge 3.8.1 deps
- [x] Crate skeleton (`trees`, `ensembles`, `svm`, `neuro_fuzzy` modules)
- [x] KB seeded: Home, INDEX, Surface, Model Types, Project Boundary,
      AI Paradigms
- [x] AGENTS.md, CLAUDE.md, Justfile, deny.toml

## v0.2.0 — Training pipeline + first classifiers

**Status:** shipped 2026-05-15.

This milestone collapsed the original v0.2.0 (Decision Tree MVP) and
v0.3.0 (Random Forest) into one coherent slice once the prism/crucible
boundary was restored and the backend-library decision clarified the
per-pack tooling.

- [x] **Boundary correction.** Training pipeline lifted out of
  `prism-analytics` into `crucible-models` per the stated boundary
  (prism never fits; crucible never owns expert rules). Burn dep moves
  here; `reqwest` / `bincode` / `converge-storage` that prism no longer
  needs drop out of prism. BREAKING for prism consumers; one-line
  import path change.
- [x] **Backend-library decision.** Recorded in
  [`Architecture/Backend Library Choices`](../Architecture/Backend%20Library%20Choices.md).
  Burn for ANFIS only; linfa for everything else.
- [x] **Typed payload adoption** at the `ProposedFact` boundary:
  `FactPayload` / `TextPayload` / `DiagnosticPayload`. All crucible
  proposal-construction sites pass typed payloads.
- [x] **`ClassifierModel` trait.** Narrow contract: `train`,
  `n_classes`, `predict`, `predict_proba`, `save`, `load`. Companion
  `RegressorModel` and `ClusteringModel` traits will follow when
  pulled.
- [x] **`crucible::trees::DecisionTreeClassifier`** — single CART tree
  via `linfa_trees::DecisionTree<f64, usize>`. Bincode-serializable.
- [x] **`crucible::ensembles::RandomForestModel`** — real
  bagging-of-CART. Deterministic under fixed seed. Bincode-serializable.
- [x] **Typed payloads:** `ClassificationFeaturesPayload` and
  `ClassPredictionPayload` with stable family / version constants in
  the `crucible.classification.*` namespace.
- [x] **`ClassifierSuggestor<M>`** — generic inference Suggestor with
  type aliases `RandomForestClassifierSuggestor` and
  `DecisionTreeClassifierSuggestor`. Wraps `predict_proba` calls in a
  `crucible.suggestor.execute` tracing span.
- [x] **Synthetic loan-default fixture** + `train_loan_default` CLI
  binary. Reproducible from seed; writes versioned artifacts under
  `crucible-models/artifacts/` (gitignored).
- [x] **End-to-end Engine integration** verified in
  `mosaic-integration-harness/tests/crucible_loan_classifier.rs` (3
  tests: high-risk applicant, low-risk applicant,
  non-features-payload rejection).
- [x] **Capability Roadmap** recorded at
  [`Capability Roadmap`](Capability%20Roadmap.md).

**Deferred from this slice:** Random Forest feature subsampling (the
"random" in split-time feature sampling). Current implementation is
classic bootstrap bagging. Trait shape and artifact format are
unchanged when feature subsampling is added. Re-open when an app
pulls on the additional variance reduction.

## v0.2.1 — Typed runtime config — shipped 2026-05-15

- [x] Adopt `ExecutionIdentity::runtime_config_from_typed` for model
      hyperparameters at the `ProposedFact` boundary.

## v0.2.2 — converge-storage cutover — shipped 2026-05-15

- [x] Migrate Parquet/ObjectStore bridge consumers to
      `converge-storage::polars_bridge`.

## v0.3.0 — Storage module removal — shipped 2026-05-17

- [x] Delete `crucible::storage` module (BREAKING). All callers move to
      `converge_storage::polars_bridge::*` in the same commit; no shim.
- [x] Bump `converge-pack` / `converge-optimization` pins to `3.9.1`.
- [x] Untrack `target/` build artifacts that pre-dated `.gitignore`.
- [x] Back-port the 5-gate `release-check` recipe + coverage /
      performance-profile / soak recipes from the converge-extension
      template (crucible previously shipped without these gates).
- [x] First publish to crates.io at `v0.3.0` (prior 0.2.x patch
      releases were tagged but never published).

**Coverage caveat:** Coverage was 66.3% at release time. `just coverage`
template floor is 80%; CI floor is 70%. v0.3.0 shipped with
`COVERAGE_FLOOR=66` as a one-time local override; CI floor was NOT
changed. Restoring crucible coverage to ≥70% (to clear the CI gate
naturally) is tracked as a follow-up before v0.3.1.

## Next slices — pull-driven

The remaining capability work is now ordered by which
`atelier-showcase` scenario or app actually pulls on it. See
[`Capability Roadmap`](Capability%20Roadmap.md) for the full table.

- **Gradient-boosted classifier** — same trait; the smallest next step.
- **`RegressorModel` trait + regression packs** — when an app needs a
  continuous score.
- **`ClusteringModel` trait + clustering packs** — when a segmentation
  showcase pulls. Unsupervised, so `DataValidationAgent` /
  `ModelEvaluationAgent` need silhouette / Davies–Bouldin metrics
  rather than accuracy / AUC.
- **ANFIS** — the only Burn pack. Niche; lands when an app needs
  fuzzy-rule interpretability that RF / GBT cannot offer.
- **Kernel SVMs** — for small-feature, sharp-boundary problems.

## v1.0.0 — Release Checklist (open)
**Epic:** E9

- [ ] Second classifier family in production use (RF + one of GBT / DT
      / SVM exercised by an app).
- [ ] Coverage ≥ 80 % across `crucible::*` excluding training-CLI
      bins.
- [ ] First clean `just lint` run.
- [ ] First clean `just release-check` run (security audit, coverage,
      performance profile, soak).
- [ ] CHANGELOG cleaned and versioned.
- [ ] Tag `v1.0.0`.
