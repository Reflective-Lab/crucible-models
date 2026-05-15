---
tags: [architecture, decision]
source: mixed
date: 2026-05-15
status: accepted
---
# Backend Library Choices

Architectural decision record. Records the per-pack ML library choice
for `crucible-models` and the rationale behind not standardising on a
single backend.

## Context

`crucible-models` was scaffolded with the framing *"trained models via
Burn."* That framing is correct for one pack family and wrong for the
others. The four planned packs differ in their mathematical character:

| Pack | Family | Trained via |
|---|---|---|
| `ensembles` | Random Forest, gradient-boosted trees | combinatorial split search; non-differentiable |
| `trees` | CART decision trees | information gain / Gini; non-differentiable |
| `svm` | kernel SVMs | quadratic programming; non-differentiable |
| `neuro_fuzzy` | ANFIS | gradient descent on membership-function parameters; **differentiable** |

Burn is a deep-learning framework optimised for differentiable models.
Forcing tree ensembles, kernel SVMs, or decision trees through Burn
means reimplementing well-known algorithms on top of tensor primitives
that are not their natural representation. That is reinvention, not
engineering.

## Decision

Per-pack backend choice. Each pack module imports its own ML library;
the public `crucible::model::ClassifierModel` trait abstracts over
them.

| Pack | Backend library | Reason |
|---|---|---|
| `ensembles` | `linfa-trees` (with `linfa-ensemble` if needed) | Canonical Rust crate for tree ensembles. Mature, idiomatic. |
| `trees` | `linfa-trees` | Same library; just the CART path. |
| `svm` | `linfa-svm` (or `smartcore` if linfa's kernel coverage is thin) | Kernel methods are linfa's domain. |
| `neuro_fuzzy` | `burn` | ANFIS *is* differentiable; Burn is the right tool. The original framing applies here. |

The crate depends on **both** `burn` and `linfa` (and possibly other
linfa sub-crates per pack) as the packs land. Compile cost is paid only
where each is actually used; future feature flags can gate the heavier
deps if compile times become a problem.

## Rationale

- **Right tool for the right job.** Burn excels at backprop-trained
  models. linfa excels at non-differentiable classical ML.
- **The `ClassifierModel` trait provides the abstraction.** Callers do
  not know — and do not need to know — what library trained a given
  artifact. The trait shape (`train` / `predict` / `predict_proba` /
  `save` / `load`) is enough.
- **No reinvention.** Random Forest and SVM are textbook algorithms
  with mature Rust implementations; rebuilding them on top of Burn is
  unjustified cost.
- **Layer hygiene preserved.** Each pack pulls only what it needs; no
  pack imports another pack's backend.

## Consequences

- The crate's dependency footprint is larger than a single-library
  alternative. This is an explicit cost we accept for correctness.
- The `crucible-models/README.md` and the workspace `kb/Modules/Crucible.md`
  framing must move away from *"all trained via Burn"* to *"trained via
  the appropriate library per pack family; Burn for ANFIS, linfa for the
  rest."* The workspace `Modules/Crucible.md` has already been updated.
- Future packs (when an app pulls) must declare their backend choice in
  this document before implementation. New backend introductions warrant
  a new row in the table above.

## Alternatives considered

1. **Burn-only.** Rejected. Roll our own RF / SVM / tree on top of Burn
   tensor primitives. High implementation cost, ignores mature Rust ML
   crates, reinvents well-understood algorithms.
2. **linfa-only.** Rejected. linfa's neural-network coverage
   (`linfa-neural`) is less mature than Burn for the ANFIS use case,
   which needs autodiff for membership-function parameter learning.
3. **`smartcore`-only.** Rejected. Full ML library; pulls every
   algorithm even when unused. Less modular than linfa, no autodiff for
   ANFIS.
4. **Custom in-crate from scratch.** Rejected. Engineering cost
   disproportionate to the value; the algorithms in question are not
   the place where crucible adds differentiation.

## See also

- [`crucible::model::ClassifierModel`](../../crates/crucible/src/model.rs)
- [Model Types](Model%20Types.md) — per-pack algorithm sketches
- [Project Boundary](Project%20Boundary.md) — prism vs crucible
- Workspace [Modules/Crucible](../../../kb/Modules/Crucible.md)
- Workspace [Architecture/Pluralist Reasoning Substrate](../../../kb/Architecture/Pluralist%20Reasoning%20Substrate.md)
