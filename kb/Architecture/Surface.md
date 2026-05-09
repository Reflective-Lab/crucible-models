---
tags: [architecture, surface]
source: llm
date: 2026-05-09
---
# Public Surface — crucible-models

## Crate name

`converge-crucible-models` (published as `crucible-models` on crates.io)

## Top-level module layout

```
crucible/
├── trees/          Decision Tree (CART)
├── ensembles/      Random Forest, Gradient-Boosted Trees
├── svm/            Support Vector Machine
└── neuro_fuzzy/    ANFIS
```

## Pack contract

Each model will expose a Converge `Pack`:

- `validate_inputs` — type check and range validation
- `solve` — run training (if no artifact) or inference (if artifact provided)
- `check_invariants` — verify output validity
- `evaluate_gate` — promote or hold in the convergence loop

## Artifact storage

Trained model artifacts (Burn `Record` serialised to bincode) are expected to
be stored in a Converge artifact registry and passed back as a base64 payload
on the inference path.  The training and inference paths are separate pack
invocations.

## Status

All modules are stubs.  No public API is stable yet.
