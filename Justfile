default:
    just --list

build:
    cargo build

test:
    cargo test

lint:
    just fmt-check
    just clippy

fmt:
    cargo fmt

fmt-check:
    cargo fmt --all -- --check

clippy:
    cargo clippy --workspace --all-targets -- -D warnings

check:
    cargo check

security-audit:
    cargo audit --deny warnings \
        --ignore RUSTSEC-2025-0141 \
        --ignore RUSTSEC-2024-0436 \
        --ignore RUSTSEC-2026-0002
    cargo deny check

doc:
    cargo doc --no-deps --open
