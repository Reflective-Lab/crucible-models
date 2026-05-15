---
source: human
date: 2026-05-09
---
# crucible-models Knowledge Base

`crucible-models` is the trained-model extension for the Converge Engine.
It owns every model that must be fit to data: Decision Trees, Random
Forests, SVMs, and ANFIS. Backend libraries are chosen per pack — `linfa`
for the non-differentiable packs (trees, ensembles, SVMs), `burn` for
ANFIS. See [Architecture/Backend Library Choices](Architecture/Backend%20Library%20Choices.md)
for the rationale.

See `../prism-analytics` for deterministic, training-free inference packs.

## Navigation

- [INDEX](INDEX.md) — entity catalog
- [Architecture](Architecture/) — design decisions and boundaries
- [Planning](Planning/MILESTONES.md) — roadmap
- [LOG](LOG.md) — session log
