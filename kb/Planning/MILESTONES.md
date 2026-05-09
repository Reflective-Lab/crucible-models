---
source: human
date: 2026-05-09
---
# Milestones — crucible-models

## v0.1.0 — Scaffold (current)

**Status:** in progress

- [x] Workspace `Cargo.toml` with Burn 0.20 and Converge 3.8.1 deps
- [x] Crate skeleton (`trees`, `ensembles`, `svm`, `neuro_fuzzy` stubs)
- [x] KB: Home, INDEX, Surface, Model Types, Project Boundary, AI Paradigms
- [x] AGENTS.md, CLAUDE.md, Justfile
- [ ] CI: `cargo check` and `cargo clippy` gates
- [ ] `deny.toml` for supply-chain auditing

---

## v0.2.0 — Decision Tree MVP

- [ ] `trees::CartTree` — training from records + labels
- [ ] `trees::CartTreePack` — Converge pack with training and inference paths
- [ ] Burn `Record` serialisation round-trip test
- [ ] Reference validation: known split on toy dataset

---

## v0.3.0 — Random Forest

- [ ] `ensembles::RandomForest` — bootstrap + feature subsampling
- [ ] `ensembles::RandomForestPack`
- [ ] Parallel tree training via Burn backend

---

## v0.4.0 — Gradient-Boosted Trees

- [ ] `ensembles::GradientBoostedTrees`
- [ ] `ensembles::GradientBoostedTreesPack`
- [ ] Differentiable loss via Burn autodiff

---

## v0.5.0 — SVM

- [ ] `svm::SupportVectorMachine` — RBF kernel, C-SVM
- [ ] `svm::SvmPack`
- [ ] SMO solver or Burn subgradient descent

---

## v1.0.0 — ANFIS + Full Release Checklist

- [ ] `neuro_fuzzy::Anfis` — five-layer architecture in Burn
- [ ] `neuro_fuzzy::AnfisPack`
- [ ] All five models with integration tests
- [ ] Coverage ≥ 80%
- [ ] First clean `just lint` run
- [ ] Tag v1.0.0
