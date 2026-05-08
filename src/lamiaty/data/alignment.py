"""
Mixed-frequency panel assembly.

Brings monthly and quarterly series onto a single monthly DatetimeIndex
without pre-interpolation. Quarterly series will have NaN in the first
two months of each quarter — this is the correct input format for the
DFM's EM-Kalman filter (§4.4 of the Implementation Plan).

Panel convention:
  Q1 observation (Jan–Mar) → appears in the March row
  Q2 observation (Apr–Jun) → appears in the June row
  Q3 observation (Jul–Sep) → appears in the September row
  Q4 observation (Oct–Dec) → appears in the December row
  All other rows for quarterly series → NaN
"""

from __future__ import annotations

import logging

import pandas as pd

logger = logging.getLogger(__name__)


def build_monthly_index(start: str, end: str) -> pd.DatetimeIndex:
    """Generate a complete monthly DatetimeIndex from start to end.

    Args:
        start: Start period, e.g., "2007-01".
        end: End period, e.g., "2026-02".

    Returns:
        DatetimeIndex with monthly frequency (MS = month start), covering
        every month from start through end inclusive.
    """
    return pd.date_range(start=start, end=end, freq="MS")


def align_series_to_monthly_index(
    series: pd.Series,
    target_index: pd.DatetimeIndex,
) -> pd.Series:
    """Reindex a series to the target monthly DatetimeIndex.

    Performs NO fill or interpolation — missing observations remain NaN.
    The series index is normalised to month-start timestamps before reindexing.

    Args:
        series: Series with a DatetimeIndex (any sub-monthly or monthly freq).
        target_index: Target monthly DatetimeIndex (from build_monthly_index).

    Returns:
        Series reindexed to target_index. NaN where data is unavailable.
    """
    # Normalise to month-start (first day of month)
    s = series.copy()
    s.index = s.index.to_period("M").to_timestamp()
    s = s[~s.index.duplicated(keep="last")]
    return s.reindex(target_index)


def build_mixed_frequency_panel(
    series_dict: dict[str, pd.Series],
    target_index: pd.DatetimeIndex,
) -> pd.DataFrame:
    """Assemble the mixed-frequency panel for DFM input.

    Each series in series_dict is aligned to target_index via
    align_series_to_monthly_index(). No interpolation is applied.

    Quarterly series (VA CONSTRUCTION, IPAI, employment) will have NaN
    in the first two months of each quarter — the EM-Kalman algorithm
    in the DFM handles this natively (§3.4 and §4.4 of the Plan).

    Args:
        series_dict: Mapping of column name → Series (already transformed,
                     i.e., yoy log-differenced and standardized for monthly
                     series; quarter-end assigned for quarterly series).
        target_index: Monthly DatetimeIndex (from build_monthly_index).

    Returns:
        DataFrame with target_index as index and one column per series.
        Shape: (len(target_index), len(series_dict)).
    """
    aligned = {}
    for name, series in series_dict.items():
        aligned[name] = align_series_to_monthly_index(series, target_index)
        n_missing = aligned[name].isna().sum()
        logger.debug("Aligned '%s': %d NaN out of %d rows", name, n_missing, len(target_index))

    panel = pd.DataFrame(aligned, index=target_index)
    panel.index.name = "date"
    logger.info(
        "Built mixed-frequency panel: %d rows × %d columns",
        len(panel),
        len(panel.columns),
    )
    return panel
