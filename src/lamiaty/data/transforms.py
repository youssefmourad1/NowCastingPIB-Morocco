"""
Generic, reusable time-series transformations.

All functions are pure (Series → Series) and free of side effects.
They are applied after data corrections and before DFM panel assembly.

Key design decisions documented here:
  - assign_quarterly_to_month_end() implements the CORRECT DFM treatment for
    quarterly series — see §3.3 of the Morocco BTP Nowcasting Implementation Plan.
    The upsample_masi.py script's repetition approach (same value for all 3 months
    of a quarter) is explicitly NOT migrated — it is the anti-pattern that the
    DFM's EM-Kalman algorithm is designed to replace.
"""

from __future__ import annotations

import logging

import numpy as np
import pandas as pd

logger = logging.getLogger(__name__)

# Quarter-end months (last month of each quarter)
QUARTER_END_MONTHS = [3, 6, 9, 12]


# ---------------------------------------------------------------------------
# Stationarity transforms
# ---------------------------------------------------------------------------


def yoy_log_diff(series: pd.Series) -> pd.Series:
    """Year-over-year log difference as percentage.

    Formula: 100 × (ln(y_t) − ln(y_{t-12}))
    Equivalent to approximately the annual growth rate in percent.

    The first 12 observations will be NaN (no year-ago value available).

    Args:
        series: Numeric series with a DatetimeIndex (monthly frequency assumed).

    Returns:
        Series of yoy growth rates (percent). Name is preserved.
    """
    result = np.log(series).diff(12) * 100
    result.name = series.name
    return result


def standardize(series: pd.Series) -> pd.Series:
    """Subtract mean and divide by standard deviation (z-score).

    Ignores NaN values when computing mean and std.

    Args:
        series: Numeric series.

    Returns:
        Standardized series with mean ≈ 0 and std ≈ 1 over non-NaN values.
        Name is preserved.
    """
    mu = series.mean()
    sigma = series.std()
    if sigma == 0 or np.isnan(sigma):
        logger.warning("standardize: std is 0 or NaN for series '%s' — returning zeros", series.name)
        return pd.Series(0.0, index=series.index, name=series.name)
    result = (series - mu) / sigma
    result.name = series.name
    return result


# ---------------------------------------------------------------------------
# Mixed-frequency treatment for quarterly series
# ---------------------------------------------------------------------------


def assign_quarterly_to_month_end(
    series: pd.Series,
    quarter_months: list[int] = QUARTER_END_MONTHS,
) -> pd.Series:
    """Assign quarterly values to the last month of each quarter; set NaN elsewhere.

    This is the CORRECT DFM treatment for series like VA CONSTRUCTION, L'IPAI,
    and Creation nette d'emploi.

    Context (§3.3 of Implementation Plan):
      The raw Excel file repeats the quarterly value across all three months of
      the quarter (e.g., Jan=Feb=Mar=16,284). The DFM with EM-Kalman filter
      handles mixed-frequency data natively via temporal aggregation constraints:
          VA^q_t = VA^m_t + VA^m_{t-1} + VA^m_{t-2}
      To enable this, the quarterly observation must appear only in the last month
      of the quarter (March, June, September, December), with NaN in the first two
      months. The Kalman smoother then interpolates the monthly path.

    This function replaces the repetition logic from upsample_masi.py, which
    was the anti-pattern explicitly rejected in §3.3.

    Args:
        series: Quarterly series with monthly DatetimeIndex. Values may be
                repeated (raw format) or may already be at quarter-end only.
        quarter_months: Months considered as quarter-end (default: [3, 6, 9, 12]).

    Returns:
        Series with the same DatetimeIndex: quarter-end months retain their value,
        all other months are set to NaN.
    """
    result = series.copy()
    non_quarter_mask = ~result.index.month.isin(quarter_months)
    result.loc[non_quarter_mask] = np.nan
    n_retained = (~non_quarter_mask).sum()
    n_zeroed = non_quarter_mask.sum()
    logger.debug(
        "assign_quarterly_to_month_end '%s': %d quarter-end values retained, %d set to NaN",
        series.name,
        n_retained,
        n_zeroed,
    )
    return result


# ---------------------------------------------------------------------------
# Daily → monthly aggregation (for MASI / shares data)
# ---------------------------------------------------------------------------


def resample_daily_to_monthly_first(
    df: pd.DataFrame,
    date_col: str = "date",
    value_col: str | None = None,
) -> pd.DataFrame:
    """Aggregate daily financial data to monthly first-trading-day observations.

    Groups by (year, month) and takes the first record in each group.
    Migrates the core aggregation logic from extract_masi.py::extract_masi_data().

    Args:
        df: DataFrame with a date column (daily records).
        date_col: Name of the date column.
        value_col: If provided, only this column (plus date) is retained in output.
                   If None, all columns are retained.

    Returns:
        DataFrame with one row per month (first available trading day).
        DatetimeIndex set to the first day of each month (MS frequency).
    """
    df = df.copy()
    df[date_col] = pd.to_datetime(df[date_col])
    df = df.sort_values(date_col)

    # Group by year-month, take first record
    df["_ym"] = df[date_col].dt.to_period("M")
    monthly = df.groupby("_ym").first().reset_index()
    monthly[date_col] = monthly["_ym"].dt.to_timestamp()
    monthly = monthly.drop(columns=["_ym"])

    if value_col is not None:
        monthly = monthly[[date_col, value_col]]

    monthly = monthly.set_index(date_col)
    monthly.index.name = "date"
    logger.debug("Resampled %d daily → %d monthly records", len(df), len(monthly))
    return monthly


# ---------------------------------------------------------------------------
# Multi-file deduplication (migrated from extract_masi.py::merge_excel_files)
# ---------------------------------------------------------------------------


def merge_and_deduplicate(
    dfs: list[pd.DataFrame],
    date_col: str = "date",
) -> pd.DataFrame:
    """Combine multiple DataFrames with the same schema, sort by date, drop duplicates.

    Migrates the logic from extract_masi.py::merge_excel_files().
    When two records share the same date, the first occurrence is kept (order matters
    — put higher-priority frames first).

    Args:
        dfs: List of DataFrames with a common date column or DatetimeIndex.
        date_col: Name of date column, or None if using DatetimeIndex.

    Returns:
        Deduplicated, date-sorted DataFrame.
    """
    combined = pd.concat(dfs, ignore_index=True)
    if date_col in combined.columns:
        combined[date_col] = pd.to_datetime(combined[date_col])
        combined = combined.sort_values(date_col).drop_duplicates(subset=[date_col]).reset_index(drop=True)
    else:
        combined = combined.sort_index()
        combined = combined[~combined.index.duplicated(keep="first")]
    logger.debug("merge_and_deduplicate: %d → %d records after dedup", sum(len(d) for d in dfs), len(combined))
    return combined
