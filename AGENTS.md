# crucible-models Agent Guide

This is the canonical agent entrypoint for `crucible-models`.

`crucible-models` is a Converge extension for trained-model inference packs.
Unlike `prism-analytics` (pure inference from hand-authored or pre-fit params),
every model here requires a Burn training pipeline.

## Start Here

1. Read `README.md`.
2. Read `kb/Architecture/Project Boundary.md` — understand what belongs here
   vs. in `prism-analytics`.
3. Read `kb/Architecture/Model Types.md` — the planned model inventory.
4. Check `Cargo.toml` for Burn feature flags and Converge version.

## Commands

```bash
cargo build
cargo test
cargo clippy -- -D warnings
```

## Boundaries

- Converge owns the pack and suggestor contracts.
- `crucible-models` owns the training pipeline, model artifacts, and
  trained-parameter storage.
- `prism-analytics` owns deterministic inference from hand-authored rules or
  pre-fit parameters — do not duplicate its packs here.
- Products own domain-specific datasets, model rollout decisions, and runtime
  assembly.

## Rules

- Preserve `unsafe_code = "forbid"`.
- Keep pack outputs as proposals, not facts (Converge contract).
- All trained artifacts must be serialisable via Burn's `Record` mechanism.
- Update `kb/LOG.md` and `kb/Planning/MILESTONES.md` when packs change.
