// Copyright 2024-2026 Reflective Labs
// SPDX-License-Identifier: MIT

//! File ingest: read CSV, Parquet, and Excel into Polars DataFrames.
//!
//! This is the entry point for structured data import. It detects the format
//! from the file extension, reads the data, and returns a `DataFrame`.
//!
//! # Supported formats
//!
//! | Format  | Extension       | Feature gate |
//! |---------|-----------------|-------------|
//! | CSV     | `.csv`, `.tsv`  | always      |
//! | Parquet | `.parquet`      | always      |
//! | Excel   | `.xlsx`, `.xls` | `excel`     |

use std::path::Path;

use anyhow::{Result, anyhow};
use polars::prelude::*;

/// Detected file format for ingest.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum IngestFormat {
    Csv,
    Tsv,
    Parquet,
    Excel,
}

impl IngestFormat {
    /// Detect format from file extension.
    pub fn from_path(path: &Path) -> Result<Self> {
        let ext = path
            .extension()
            .and_then(|e| e.to_str())
            .map(str::to_ascii_lowercase)
            .unwrap_or_default();

        match ext.as_str() {
            "csv" => Ok(Self::Csv),
            "tsv" => Ok(Self::Tsv),
            "parquet" | "pq" => Ok(Self::Parquet),
            "xlsx" | "xls" => Ok(Self::Excel),
            _ => Err(anyhow!(
                "unsupported file format: .{ext} (expected csv, tsv, parquet, xlsx, or xls)"
            )),
        }
    }
}

/// Read a file into a Polars DataFrame. Format is detected from the extension.
///
/// For Excel files, reads the first sheet by default. Use [`read_excel_sheet`]
/// to specify a sheet name.
pub fn read_file(path: &Path) -> Result<DataFrame> {
    let format = IngestFormat::from_path(path)?;

    match format {
        IngestFormat::Csv => read_csv(path),
        IngestFormat::Tsv => read_tsv(path),
        IngestFormat::Parquet => read_parquet(path),
        IngestFormat::Excel => {
            #[cfg(feature = "excel")]
            {
                read_excel(path, None)
            }
            #[cfg(not(feature = "excel"))]
            {
                Err(anyhow!(
                    "Excel support requires the 'excel' feature. Rebuild with: cargo build --features excel"
                ))
            }
        }
    }
}

/// Read a CSV file into a DataFrame.
pub fn read_csv(path: &Path) -> Result<DataFrame> {
    let df = CsvReadOptions::default()
        .with_has_header(true)
        .with_infer_schema_length(Some(1000))
        .try_into_reader_with_file_path(Some(path.into()))?
        .finish()?;

    tracing::info!(
        path = %path.display(),
        rows = df.height(),
        cols = df.width(),
        "ingested CSV"
    );

    Ok(df)
}

/// Read a TSV (tab-separated) file into a DataFrame.
pub fn read_tsv(path: &Path) -> Result<DataFrame> {
    let df = CsvReadOptions::default()
        .with_has_header(true)
        .with_parse_options(CsvParseOptions::default().with_separator(b'\t'))
        .with_infer_schema_length(Some(1000))
        .try_into_reader_with_file_path(Some(path.into()))?
        .finish()?;

    tracing::info!(
        path = %path.display(),
        rows = df.height(),
        cols = df.width(),
        "ingested TSV"
    );

    Ok(df)
}

/// Read a Parquet file into a DataFrame.
pub fn read_parquet(path: &Path) -> Result<DataFrame> {
    let path_str = path
        .to_str()
        .ok_or_else(|| anyhow!("path is not valid UTF-8: {}", path.display()))?;

    let df = LazyFrame::scan_parquet(PlPath::from_str(path_str), Default::default())?.collect()?;

    tracing::info!(
        path = %path.display(),
        rows = df.height(),
        cols = df.width(),
        "ingested Parquet"
    );

    Ok(df)
}

/// Read an Excel file into a DataFrame.
///
/// If `sheet` is None, reads the first sheet.
#[cfg(feature = "excel")]
pub fn read_excel(path: &Path, sheet: Option<&str>) -> Result<DataFrame> {
    use calamine::{Reader, open_workbook_auto};

    let mut workbook =
        open_workbook_auto(path).map_err(|e| anyhow!("failed to open Excel file: {e}"))?;

    let sheet_name = match sheet {
        Some(name) => name.to_string(),
        None => {
            let names = workbook.sheet_names();
            names
                .first()
                .ok_or_else(|| anyhow!("Excel file has no sheets"))?
                .clone()
        }
    };

    let range = workbook
        .worksheet_range(&sheet_name)
        .map_err(|e| anyhow!("failed to read sheet '{sheet_name}': {e}"))?;

    let (rows, cols) = range.get_size();
    if rows < 2 || cols == 0 {
        return Err(anyhow!(
            "sheet '{sheet_name}' has no data (rows={rows}, cols={cols})"
        ));
    }

    // First row is headers
    let headers: Vec<String> = range
        .rows()
        .next()
        .map(|row| {
            row.iter()
                .enumerate()
                .map(|(i, cell)| {
                    let val = cell.to_string();
                    if val.is_empty() {
                        format!("column_{i}")
                    } else {
                        val
                    }
                })
                .collect()
        })
        .unwrap_or_default();

    // Build columns
    let mut columns: Vec<Vec<String>> = vec![Vec::with_capacity(rows - 1); cols];
    for row in range.rows().skip(1) {
        for (col_idx, cell) in row.iter().enumerate() {
            if col_idx < cols {
                columns[col_idx].push(cell.to_string());
            }
        }
    }

    // Try to infer numeric columns, otherwise keep as string
    let series: Vec<Column> = headers
        .iter()
        .zip(columns.iter())
        .map(|(name, values)| {
            // Try f64 first
            let floats: Option<Vec<Option<f64>>> = values
                .iter()
                .map(|v| {
                    if v.is_empty() {
                        Some(None)
                    } else {
                        v.parse::<f64>().ok().map(Some)
                    }
                })
                .collect();

            if let Some(floats) = floats {
                Column::new(
                    name.as_str().into(),
                    floats.into_iter().collect::<Float64Chunked>(),
                )
            } else {
                Column::new(
                    name.as_str().into(),
                    values.iter().map(String::as_str).collect::<StringChunked>(),
                )
            }
        })
        .collect();

    let df = DataFrame::new(series)?;

    tracing::info!(
        path = %path.display(),
        sheet = sheet_name,
        rows = df.height(),
        cols = df.width(),
        "ingested Excel"
    );

    Ok(df)
}

/// List sheet names in an Excel file.
#[cfg(feature = "excel")]
pub fn list_excel_sheets(path: &Path) -> Result<Vec<String>> {
    use calamine::{Reader, open_workbook_auto};

    let workbook =
        open_workbook_auto(path).map_err(|e| anyhow!("failed to open Excel file: {e}"))?;

    Ok(workbook.sheet_names().to_vec())
}

/// Summary of an ingested file.
#[derive(Debug, Clone, serde::Serialize)]
pub struct IngestSummary {
    pub format: String,
    pub rows: usize,
    pub columns: Vec<ColumnSummary>,
}

/// Summary of a single column.
#[derive(Debug, Clone, serde::Serialize)]
pub struct ColumnSummary {
    pub name: String,
    pub dtype: String,
    pub null_count: usize,
    pub sample_values: Vec<String>,
}

/// Produce a summary of a DataFrame for inspection.
pub fn summarize(df: &DataFrame) -> IngestSummary {
    let columns = df
        .get_columns()
        .iter()
        .map(|col| {
            let sample_values: Vec<String> = (0..col.len().min(3))
                .map(|i| format!("{}", col.get(i).unwrap_or(AnyValue::Null)))
                .collect();

            ColumnSummary {
                name: col.name().to_string(),
                dtype: format!("{}", col.dtype()),
                null_count: col.null_count(),
                sample_values,
            }
        })
        .collect();

    IngestSummary {
        format: "dataframe".to_string(),
        rows: df.height(),
        columns,
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use std::io::Write;

    #[test]
    fn read_csv_basic() {
        let dir = tempfile::tempdir().unwrap();
        let path = dir.path().join("test.csv");
        let mut f = std::fs::File::create(&path).unwrap();
        writeln!(f, "name,arr,growth").unwrap();
        writeln!(f, "Acme,5000000,0.25").unwrap();
        writeln!(f, "Beta,12000000,0.15").unwrap();

        let df = read_csv(&path).unwrap();
        assert_eq!(df.height(), 2);
        assert_eq!(df.width(), 3);
    }

    #[test]
    fn read_tsv_basic() {
        let dir = tempfile::tempdir().unwrap();
        let path = dir.path().join("test.tsv");
        let mut f = std::fs::File::create(&path).unwrap();
        writeln!(f, "name\tarr\tgrowth").unwrap();
        writeln!(f, "Acme\t5000000\t0.25").unwrap();

        let df = read_tsv(&path).unwrap();
        assert_eq!(df.height(), 1);
        assert_eq!(df.width(), 3);
    }

    #[test]
    fn read_parquet_roundtrip() {
        let dir = tempfile::tempdir().unwrap();
        let path = dir.path().join("test.parquet");

        let mut df = df! {
            "company" => &["Acme", "Beta"],
            "arr" => &[5_000_000i64, 12_000_000],
        }
        .unwrap();

        let file = std::fs::File::create(&path).unwrap();
        ParquetWriter::new(file).finish(&mut df).unwrap();

        let loaded = read_parquet(&path).unwrap();
        assert_eq!(loaded.height(), 2);
    }

    #[test]
    fn format_detection() {
        assert_eq!(
            IngestFormat::from_path(Path::new("data.csv")).unwrap(),
            IngestFormat::Csv
        );
        assert_eq!(
            IngestFormat::from_path(Path::new("data.XLSX")).unwrap(),
            IngestFormat::Excel
        );
        assert_eq!(
            IngestFormat::from_path(Path::new("data.parquet")).unwrap(),
            IngestFormat::Parquet
        );
        assert!(IngestFormat::from_path(Path::new("data.txt")).is_err());
    }

    #[test]
    fn summarize_df() {
        let df = df! {
            "name" => &["Acme", "Beta", "Gamma"],
            "arr" => &[5_000_000i64, 12_000_000, 3_000_000],
        }
        .unwrap();

        let summary = summarize(&df);
        assert_eq!(summary.rows, 3);
        assert_eq!(summary.columns.len(), 2);
        assert_eq!(summary.columns[0].name, "name");
    }

    #[test]
    fn read_file_dispatches_csv() {
        // Intent: `read_file` is the public dispatcher — if it stops
        // routing .csv to read_csv (or any other branch), callers see
        // an UnsupportedFormat error even though support exists.
        let dir = tempfile::tempdir().unwrap();
        let path = dir.path().join("dispatch.csv");
        let mut f = std::fs::File::create(&path).unwrap();
        writeln!(f, "a,b").unwrap();
        writeln!(f, "1,2").unwrap();
        let df = read_file(&path).unwrap();
        assert_eq!(df.height(), 1);
        assert_eq!(df.width(), 2);
    }

    #[test]
    fn read_file_dispatches_tsv() {
        let dir = tempfile::tempdir().unwrap();
        let path = dir.path().join("dispatch.tsv");
        let mut f = std::fs::File::create(&path).unwrap();
        writeln!(f, "a\tb").unwrap();
        writeln!(f, "1\t2").unwrap();
        let df = read_file(&path).unwrap();
        assert_eq!(df.height(), 1);
    }

    #[test]
    fn read_file_dispatches_parquet() {
        let dir = tempfile::tempdir().unwrap();
        let path = dir.path().join("dispatch.parquet");
        let mut df = df! { "a" => &[1i64, 2] }.unwrap();
        let file = std::fs::File::create(&path).unwrap();
        ParquetWriter::new(file).finish(&mut df).unwrap();
        let loaded = read_file(&path).unwrap();
        assert_eq!(loaded.height(), 2);
    }

    #[test]
    #[cfg(not(feature = "excel"))]
    fn read_file_excel_errors_without_feature() {
        // Intent: silently returning an empty DataFrame for an Excel
        // path when the feature is off would corrupt the next stage —
        // it must error so the caller knows to rebuild with the
        // feature flag.
        let dir = tempfile::tempdir().unwrap();
        let path = dir.path().join("dispatch.xlsx");
        std::fs::write(&path, b"not real").unwrap();
        let err = read_file(&path).unwrap_err();
        assert!(err.to_string().contains("excel"));
    }

    #[test]
    fn read_file_errors_on_unsupported_extension() {
        // Intent: a typo'd extension must not silently fall through to
        // a CSV reader; the explicit unsupported-format error is the
        // only signal a caller has to fix their pipeline.
        let dir = tempfile::tempdir().unwrap();
        let path = dir.path().join("dispatch.weirdext");
        std::fs::write(&path, b"a,b\n1,2").unwrap();
        let err = read_file(&path).unwrap_err();
        assert!(err.to_string().contains("unsupported file format"));
    }

    #[test]
    fn ingest_format_pq_extension_maps_to_parquet() {
        // Intent: callers who use the conventional `.pq` short
        // extension expect Parquet handling; losing this alias would
        // break any pipeline that writes `.pq` artifacts.
        assert_eq!(
            IngestFormat::from_path(Path::new("data.pq")).unwrap(),
            IngestFormat::Parquet
        );
    }

    #[test]
    fn ingest_format_unknown_extension_returns_helpful_error() {
        // Intent: the error message must enumerate supported formats so
        // the caller can fix their pipeline without grepping source.
        let err = IngestFormat::from_path(Path::new("data.bin")).unwrap_err();
        let msg = err.to_string();
        assert!(msg.contains("csv"));
        assert!(msg.contains("parquet"));
        assert!(msg.contains(".bin"));
    }

    #[test]
    fn read_parquet_errors_on_nonexistent_file() {
        // Intent: missing file must surface as an error, not a hidden
        // empty DataFrame.
        let err = read_parquet(Path::new("/nonexistent/dir/no.parquet"));
        assert!(err.is_err());
    }
}
