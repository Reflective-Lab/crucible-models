# Claude Code Entrypoint

Read and follow `AGENTS.md` — it is the canonical project documentation.

## Session Scope

- **Milestones:** `kb/Planning/MILESTONES.md`
- **Project boundary:** `kb/Architecture/Project Boundary.md`

## Claude-Specific Notes

- Prefer Edit over Write for existing files.
- Knowledge belongs in `kb/`, not as doc comments in source.
- Run `cargo clippy -- -D warnings` before considering work done.
- Never push to main without confirmation.

## Floor versions

This extension targets:

- Converge >= 3.8.1
- Burn >= 0.20.0
- MSRV 1.94.0
- Edition 2024
- `unsafe_code = "forbid"`
