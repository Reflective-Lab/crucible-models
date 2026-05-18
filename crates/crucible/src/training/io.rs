// Copyright 2024-2026 Reflective Labs

use anyhow::{Context as _, Result, anyhow};
use polars::prelude::*;
use serde::Serialize;
use std::fs::File;
use std::io::Write;
use std::path::Path;

pub(super) const DATASET_URL: &str = "https://huggingface.co/datasets/gvlassis/california_housing/resolve/refs%2Fconvert%2Fparquet/default/train/0000.parquet";
pub(super) const TARGET_COLUMN: &str = "median_house_value";

pub(super) fn download_dataset_if_missing(path: &Path) -> Result<()> {
    if path.exists() {
        return Ok(());
    }

    let response = reqwest::blocking::get(DATASET_URL)?;
    let content = response.bytes()?;

    let mut file = File::create(path)?;
    file.write_all(&content)?;

    Ok(())
}

pub(super) fn load_dataframe(path: &Path) -> Result<DataFrame> {
    let extension = path
        .extension()
        .and_then(|ext| ext.to_str())
        .unwrap_or("")
        .to_ascii_lowercase();

    let path_str = path
        .to_str()
        .ok_or_else(|| anyhow!("path is not valid utf-8: {}", path.display()))?;

    match extension.as_str() {
        "parquet" => {
            let pl_path = PlPath::from_str(path_str);
            Ok(LazyFrame::scan_parquet(pl_path, Default::default())?.collect()?)
        }
        "csv" => Ok(CsvReadOptions::default()
            .with_has_header(true)
            .try_into_reader_with_file_path(Some(path.to_path_buf()))?
            .finish()?),
        _ => Err(anyhow!(
            "unsupported data format for path {} (expected .csv or .parquet)",
            path.display()
        )),
    }
}

pub(super) fn write_parquet(df: &DataFrame, path: &Path) -> Result<()> {
    let mut file = File::create(path)?;
    let mut owned = df.clone();
    ParquetWriter::new(&mut file).finish(&mut owned)?;
    Ok(())
}

pub(super) fn write_json<T: Serialize>(path: &Path, value: &T) -> Result<()> {
    let content = serde_json::to_string_pretty(value)?;
    let mut file = File::create(path)?;
    file.write_all(content.as_bytes())?;
    Ok(())
}

pub(super) fn is_numeric_dtype(dtype: &DataType) -> bool {
    matches!(
        dtype,
        DataType::Int8
            | DataType::Int16
            | DataType::Int32
            | DataType::Int64
            | DataType::UInt8
            | DataType::UInt16
            | DataType::UInt32
            | DataType::UInt64
            | DataType::Float32
            | DataType::Float64
    )
}

pub(super) fn compute_numeric_stats(series: &Series) -> Result<(f64, f64, usize)> {
    let casted = series.cast(&DataType::Float64)?;
    let values: Vec<f64> = casted
        .f64()
        .context("numeric series not f64")?
        .into_no_null_iter()
        .collect();
    if values.is_empty() {
        return Err(anyhow!("no numeric values to compute stats"));
    }

    let mut total = 0.0;
    for value in &values {
        total += *value;
    }
    let mean = total / values.len() as f64;

    let mut variance_sum = 0.0;
    for value in &values {
        let diff = *value - mean;
        variance_sum += diff * diff;
    }
    let std = (variance_sum / values.len() as f64).sqrt();

    let outliers = if std > 0.0 {
        values
            .iter()
            .filter(|value| (*value - mean).abs() > 3.0 * std)
            .count()
    } else {
        0
    };

    Ok((mean, std, outliers))
}

pub(super) fn split_feature_columns(df: &DataFrame, target: &str) -> (Vec<String>, Vec<String>) {
    let mut numeric = Vec::new();
    let mut categorical = Vec::new();
    for series in df.get_columns() {
        let name = series.name();
        if name == target {
            continue;
        }
        if is_numeric_dtype(series.dtype()) {
            numeric.push(name.to_string());
        } else {
            categorical.push(name.to_string());
        }
    }
    (numeric, categorical)
}

pub(super) fn select_target_column(df: &DataFrame) -> Result<(String, Series)> {
    if let Ok(col) = df.column(TARGET_COLUMN) {
        return Ok((
            TARGET_COLUMN.to_string(),
            col.as_materialized_series().clone(),
        ));
    }

    let mut numeric = df
        .get_columns()
        .iter()
        .filter(|series| is_numeric_dtype(series.dtype()))
        .cloned()
        .collect::<Vec<_>>();

    let fallback = numeric
        .pop()
        .ok_or_else(|| anyhow!("no numeric columns available for target"))?;
    let series = fallback.as_materialized_series().clone();
    Ok((series.name().to_string(), series))
}

pub(super) fn get_numeric_series(df: &DataFrame, name: &str) -> Result<Series> {
    let series = df
        .column(name)
        .map_err(|_| anyhow!("missing target column {}", name))?
        .as_materialized_series();
    let casted = series.cast(&DataType::Float64)?;
    Ok(casted)
}

pub(super) fn mean_of_series(series: &Series) -> Result<f64> {
    let casted = series.cast(&DataType::Float64)?;
    let values = casted
        .f64()
        .context("target column not f64")?
        .into_no_null_iter();
    let mut total = 0.0;
    let mut count = 0usize;
    for value in values {
        total += value;
        count += 1;
    }
    if count == 0 {
        return Err(anyhow!("no values to compute mean"));
    }
    Ok(total / count as f64)
}

pub(super) fn mean_abs_error(target: &Series, mean: f64) -> Result<f64> {
    let casted = target.cast(&DataType::Float64)?;
    let values = casted
        .f64()
        .context("target column not f64")?
        .into_no_null_iter();
    let mut total = 0.0;
    let mut count = 0usize;
    for value in values {
        total += (value - mean).abs();
        count += 1;
    }
    if count == 0 {
        return Err(anyhow!("no values to evaluate"));
    }
    Ok(total / count as f64)
}

pub(super) fn mean_abs_value(target: &Series) -> Result<f64> {
    let casted = target.cast(&DataType::Float64)?;
    let values = casted
        .f64()
        .context("target column not f64")?
        .into_no_null_iter();
    let mut total = 0.0;
    let mut count = 0usize;
    for value in values {
        total += value.abs();
        count += 1;
    }
    if count == 0 {
        return Err(anyhow!("no values to evaluate"));
    }
    Ok(total / count as f64)
}

pub(super) fn compute_mean_std(values: &ChunkedArray<Float64Type>) -> Result<(f64, f64)> {
    let vals: Vec<f64> = values.into_no_null_iter().collect();
    if vals.is_empty() {
        return Err(anyhow!("no values for mean/std computation"));
    }

    let mean = vals.iter().sum::<f64>() / vals.len() as f64;
    let variance = vals.iter().map(|v| (v - mean).powi(2)).sum::<f64>() / vals.len() as f64;
    let std = variance.sqrt();

    Ok((mean, std))
}
