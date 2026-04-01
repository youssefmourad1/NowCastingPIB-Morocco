"""
Pseudo-vintage builder — reconstructs the information set as it would have
been available on any historical date, using publication lags from
configs/publication_calendar.yaml.

Protocol (§5.3.1 of Implementation Plan):
  For each forecast origin (7th and 21st of each month, Jan 2015 – Dec 2024):
    1. Apply publication lags to determine available observations per series
    2. Reconstruct information set Ω_v — future/unavailable obs set to NaN
    3. Pass to DFM for estimation + nowcast generation

Key constraint: ~40 quarterly evaluation points for VA CONSTRUCTION.
"""

from __future__ import annotations

import logging

import numpy as np
import pandas as pd

logger = logging.getLogger(__name__)


def build_vintage(
    panel: pd.DataFrame,
    forecast_origin: pd.Timestamp,
    publication_calendar: dict,
) -> pd.DataFrame:
    """Reconstruct the dataset as-of forecast_origin.

    For each series in publication_calendar, computes the availability cutoff:
      availability_cutoff = forecast_origin − lag_days
    All observations whose month-start index is after the cutoff are set to NaN,
    simulating the information set that would have been available on that date.

    Args:
        panel: Full model panel (all data, monthly DatetimeIndex).
        forecast_origin: The date at which knowledge is simulated.
        publication_calendar: Dict loaded from configs/publication_calendar.yaml.
                              Expected structure: {'series': {'name': {'lag_days': N}}}

    Returns:
        Copy of panel with future/unavailable observations censored to NaN.
    """
    vintage = panel.copy()
    # publication_calendar is already the inner series dict (settings.py extracts
    # the "series" key before storing it).  Accept both forms for robustness.
    if "series" in publication_calendar and isinstance(publication_calendar["series"], dict):
        series_calendar = publication_calendar["series"]
    else:
        series_calendar = publication_calendar

    for series_name, cal_info in series_calendar.items():
        if series_name not in vintage.columns:
            continue

        lag_days = int(cal_info.get("lag_days", 35))
        availability_cutoff = forecast_origin - pd.Timedelta(days=lag_days)

        # For monthly series: last available month-start is the latest month
        # whose data would have been published by forecast_origin.
        # Panel index is month-start (MS), so we censor rows > availability_cutoff.
        mask_future = vintage.index > availability_cutoff
        vintage.loc[mask_future, series_name] = np.nan

        n_censored = mask_future.sum()
        logger.debug(
            "Vintage %s: censored %d rows of '%s' (lag=%d days, cutoff=%s)",
            forecast_origin.date(), n_censored, series_name, lag_days,
            availability_cutoff.date(),
        )

    # Also censor any columns NOT in the publication calendar after forecast_origin
    # (conservative: no information beyond the origin date itself)
    unlisted = [c for c in vintage.columns if c not in series_calendar]
    if unlisted:
        mask_origin = vintage.index > forecast_origin
        for col in unlisted:
            vintage.loc[mask_origin, col] = np.nan
        logger.debug(
            "Vintage %s: censored %d unlisted columns after origin",
            forecast_origin.date(), len(unlisted),
        )

    logger.info(
        "Built vintage as-of %s: %d rows, NaN per col: %s",
        forecast_origin.date(),
        len(vintage),
        vintage.isnull().sum().to_dict(),
    )
    return vintage
