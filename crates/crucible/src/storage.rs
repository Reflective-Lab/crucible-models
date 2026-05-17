// Copyright 2024-2026 Reflective Labs
// SPDX-License-Identifier: MIT

//! Storage-aware data loading for analytics.
//!
//! The implementation moved to
//! [`converge_storage::polars_bridge`](converge_storage::polars_bridge)
//! in `converge-storage 3.9.1` so any extension that needs Parquet from
//! a remote store (training pipelines, KB persistence, app dataset
//! ingest) shares a single implementation rather than re-inventing the
//! cache logic per crate.
//!
//! This module remains as a deprecation shim. Migrate to
//! `converge_storage::polars_bridge::*` directly; the re-exports here
//! will be removed in the next crucible MAJOR.

#![allow(deprecated)]

#[deprecated(
    since = "0.2.2",
    note = "use `converge_storage::polars_bridge::fetch_parquet` directly"
)]
pub use converge_storage::polars_bridge::fetch_parquet;

#[deprecated(
    since = "0.2.2",
    note = "use `converge_storage::polars_bridge::fetch_to_cache` directly"
)]
pub use converge_storage::polars_bridge::fetch_to_cache;

#[deprecated(
    since = "0.2.2",
    note = "use `converge_storage::polars_bridge::write_parquet_to_store` directly"
)]
pub use converge_storage::polars_bridge::write_parquet_to_store;

#[deprecated(
    since = "0.2.2",
    note = "use `converge_storage::polars_bridge::scan_local_parquet` directly"
)]
pub use converge_storage::polars_bridge::scan_local_parquet;
