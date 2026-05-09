default:
    just --list

build:
    cargo build

test:
    cargo test

lint:
    cargo clippy -- -D warnings
    cargo fmt --check

fmt:
    cargo fmt

check:
    cargo check

doc:
    cargo doc --no-deps --open
