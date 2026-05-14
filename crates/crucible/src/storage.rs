// Copyright 2024-2026 Reflective Labs
// SPDX-License-Identifier: MIT

//! Storage-aware data loading for analytics.
//!
//! Bridges `converge-storage` with Polars by fetching blobs from any
//! [`ObjectStore`] backend into a local cache directory, then handing
//! the local path to Polars for zero-copy memory-mapped reads.
//!
//! Also provides writing parquet back to any storage backend.

use std::path::{Path, PathBuf};
use std::{
    collections::hash_map::DefaultHasher,
    hash::{Hash, Hasher},
};

use anyhow::{Result, anyhow};
use converge_storage::object_store::ObjectStoreExt;
use converge_storage::{ObjectPath, ObjectStore};
use polars::prelude::*;

/// Fetch a parquet file from storage into a local cache and return the local path.
///
/// Consumers can then pass this path to Polars `scan_parquet` on their own
/// terms (sync context avoids nested tokio runtime issues).
///
/// If the file already exists in `cache_dir`, it is reused without re-downloading.
pub async fn fetch_parquet(
    store: &dyn ObjectStore,
    key: &str,
    cache_dir: &Path,
) -> Result<PathBuf> {
    fetch_to_cache(store, key, cache_dir).await
}

/// Fetch a blob from storage into a local cache directory.
///
/// Returns the local path to the cached file. Skips download if the file
/// already exists (by name — no checksum validation for now).
pub async fn fetch_to_cache(
    store: &dyn ObjectStore,
    key: &str,
    cache_dir: &Path,
) -> Result<PathBuf> {
    let local_path = cache_path_for_key(key, cache_dir);

    if local_path.exists() {
        tracing::debug!(path = %local_path.display(), "using cached file");
        return Ok(local_path);
    }

    std::fs::create_dir_all(cache_dir)?;

    let obj_path = ObjectPath::from(key);
    let result: converge_storage::GetResult = store.get(&obj_path).await?;
    let bytes = result.bytes().await?;

    std::fs::write(&local_path, &bytes)?;
    tracing::info!(
        key,
        path = %local_path.display(),
        bytes = bytes.len(),
        "fetched from storage to cache"
    );

    Ok(local_path)
}

fn cache_path_for_key(key: &str, cache_dir: &Path) -> PathBuf {
    let mut hasher = DefaultHasher::new();
    key.hash(&mut hasher);
    let hash = hasher.finish();

    let filename = key
        .rsplit('/')
        .next()
        .filter(|part| !part.is_empty())
        .unwrap_or("blob");
    let filename = filename
        .chars()
        .map(|ch| match ch {
            ':' | '<' | '>' | '"' | '\\' | '|' | '?' | '*' => '_',
            _ if ch.is_ascii_control() => '_',
            _ => ch,
        })
        .collect::<String>();

    cache_dir.join(format!("{hash:016x}-{filename}"))
}

/// Write a Polars `DataFrame` as parquet to an object store.
pub async fn write_parquet_to_store(
    store: &dyn ObjectStore,
    key: &str,
    df: &mut DataFrame,
) -> Result<()> {
    let mut buf = Vec::new();
    ParquetWriter::new(&mut buf).finish(df)?;

    let obj_path = ObjectPath::from(key);
    let payload = converge_storage::object_store::PutPayload::from(buf);
    store
        .put(&obj_path, payload)
        .await
        .map_err(|e| anyhow!("failed to write parquet to storage: {e}"))?;

    tracing::info!(key, "wrote parquet to storage");
    Ok(())
}

/// Load a parquet file from a local path into a Polars `LazyFrame`.
///
/// This is a sync helper — call it outside of an async context if Polars
/// uses tokio internally (avoids nested runtime panics).
pub fn scan_local_parquet(path: &Path) -> Result<LazyFrame> {
    let path_str = path
        .to_str()
        .ok_or_else(|| anyhow!("path is not valid utf-8: {:?}", path))?;
    let pl_path = PlPath::from_str(path_str);
    Ok(LazyFrame::scan_parquet(pl_path, Default::default())?)
}

#[cfg(test)]
mod tests {
    use super::*;
    use converge_storage::object_store::memory::InMemory;

    #[tokio::test]
    async fn roundtrip_parquet_through_store() {
        let store = InMemory::new();
        let cache_dir = tempfile::tempdir().unwrap();

        // Create a small DataFrame and write to store
        let mut df = df! {
            "id" => &[1i64, 2, 3],
            "value" => &[10.0f64, 20.0, 30.0],
        }
        .unwrap();

        write_parquet_to_store(&store, "test/data.parquet", &mut df)
            .await
            .unwrap();

        // Fetch to cache
        let local_path = fetch_parquet(&store, "test/data.parquet", cache_dir.path())
            .await
            .unwrap();

        // Read back with Polars in a blocking context
        let result = tokio::task::spawn_blocking(move || {
            scan_local_parquet(&local_path).unwrap().collect().unwrap()
        })
        .await
        .unwrap();

        assert_eq!(result.height(), 3);
        assert_eq!(result.width(), 2);
    }

    #[tokio::test]
    async fn fetch_uses_cache_on_second_call() {
        let store = InMemory::new();
        let cache_dir = tempfile::tempdir().unwrap();

        // Put some data
        let obj_path = ObjectPath::from("cached.parquet");
        let payload = converge_storage::object_store::PutPayload::from(vec![1u8, 2, 3]);
        store.put(&obj_path, payload).await.unwrap();

        // First fetch — downloads
        let path1 = fetch_to_cache(&store, "cached.parquet", cache_dir.path())
            .await
            .unwrap();
        assert!(path1.exists());

        // Second fetch — hits cache
        let path2 = fetch_to_cache(&store, "cached.parquet", cache_dir.path())
            .await
            .unwrap();
        assert_eq!(path1, path2);
    }

    #[tokio::test]
    async fn fetch_distinguishes_keys_with_same_basename() {
        let store = InMemory::new();
        let cache_dir = tempfile::tempdir().unwrap();

        let path_a = ObjectPath::from("tenant-a/data.parquet");
        let path_b = ObjectPath::from("tenant-b/data.parquet");

        store
            .put(
                &path_a,
                converge_storage::object_store::PutPayload::from(vec![1u8, 2, 3]),
            )
            .await
            .unwrap();
        store
            .put(
                &path_b,
                converge_storage::object_store::PutPayload::from(vec![4u8, 5, 6]),
            )
            .await
            .unwrap();

        let local_a = fetch_to_cache(&store, "tenant-a/data.parquet", cache_dir.path())
            .await
            .unwrap();
        let local_b = fetch_to_cache(&store, "tenant-b/data.parquet", cache_dir.path())
            .await
            .unwrap();

        assert_ne!(local_a, local_b);
        assert_eq!(std::fs::read(local_a).unwrap(), vec![1u8, 2, 3]);
        assert_eq!(std::fs::read(local_b).unwrap(), vec![4u8, 5, 6]);
    }
}
