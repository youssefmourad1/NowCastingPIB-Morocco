"""
Backtest runner — iterates over forecast origins, builds vintages,
fits the DFM, and generates nowcasts for pseudo-real-time evaluation.

Protocol (§5.3.1):
  For each origin in [2015-01-07, …, 2024-12-21] (bi-monthly, ~240 origins):
    1. build_vintage()  — reconstruct information set as-of origin
    2. DFM.fit()        — estimate on rolling 12-year window up to origin
    3. DFM.nowcast()    — predict next unreleased VA CONSTRUCTION quarter
    4. Store results    — origin, target_quarter, nowcast, ci_lower, ci_upper
  Align with realized VA CONSTRUCTION → compute errors
"""

from __future__ import annotations

import logging
from itertools import product
from pathlib import Path

import numpy as np
import pandas as pd

from lamiaty.backtest.vintage_builder import build_vintage
from lamiaty.config.settings import Settings
from lamiaty.model.dfm import DynamicFactorModel
from lamiaty.utils.logging import log_stage

logger = logging.getLogger(__name__)

_RESULTS_CACHE = "data/vintages/backtest_results.parquet"


def run_backtest(
    settings: Settings,
    panel: pd.DataFrame,
    save_vintages: bool = False,
) -> pd.DataFrame:
    """Run pseudo-real-time backtest over all configured forecast origins.

    Args:
        settings:      Loaded Settings (includes backtest config + publication_calendar).
        panel:         Full model panel from run_pipeline().
        save_vintages: If True, save each vintage to data/vintages/{date}.parquet.

    Returns:
        DataFrame with columns:
          origin, target_quarter, nowcast, ci_lower, ci_upper, realized, error
        Rows are sorted by origin.
    """
    bt = settings.model.backtest
    origin_start = pd.Timestamp(bt.origin_start)
    origin_end   = pd.Timestamp(bt.origin_end)
    update_days  = bt.update_days
    rolling_yrs  = bt.rolling_window_years
    full_start   = pd.Timestamp(settings.model.sample.full_sample_start)

    # Generate all forecast origins
    origins = _generate_origins(origin_start, origin_end, update_days)
    logger.info("Backtest: %d forecast origins (%s → %s)", len(origins),
                origins[0].date(), origins[-1].date())

    results = []

    for i, origin in enumerate(origins):
        with log_stage(f"Origin {i+1}/{len(origins)}: {origin.date()}", "lamiaty.backtest"):
            try:
                record = _run_single_origin(
                    origin=origin,
                    panel=panel,
                    settings=settings,
                    rolling_yrs=rolling_yrs,
                    full_start=full_start,
                    save_vintages=save_vintages,
                )
                results.append(record)
            except Exception as exc:
                logger.error("Origin %s failed: %s — skipping", origin.date(), exc)
                results.append({
                    "origin":         origin,
                    "target_quarter": pd.NaT,
                    "nowcast":        float("nan"),
                    "ci_lower":       float("nan"),
                    "ci_upper":       float("nan"),
                    "realized":       float("nan"),
                    "error":          float("nan"),
                    "status":         f"FAILED: {exc}",
                })

    df = pd.DataFrame(results).sort_values("origin").reset_index(drop=True)

    # Save consolidated results
    _save_results(df, settings.paths.vintages_dir)

    logger.info(
        "Backtest complete: %d origins, %d successful, %d failed",
        len(df),
        df["status"].eq("OK").sum() if "status" in df.columns else len(df),
        df["status"].ne("OK").sum() if "status" in df.columns else 0,
    )
    return df


def load_backtest_results(vintages_dir: Path) -> pd.DataFrame | None:
    """Load previously saved backtest results, if available.

    Returns None if no cached results exist.
    """
    cache_path = Path(vintages_dir) / "backtest_results.parquet"
    if cache_path.exists():
        logger.info("Loading cached backtest results from %s", cache_path)
        return pd.read_parquet(cache_path)
    return None


# ---------------------------------------------------------------------------
# Private helpers
# ---------------------------------------------------------------------------


def _run_single_origin(
    origin: pd.Timestamp,
    panel: pd.DataFrame,
    settings: Settings,
    rolling_yrs: int,
    full_start: pd.Timestamp,
    save_vintages: bool,
) -> dict:
    """Fit DFM and nowcast at a single forecast origin."""

    # 1. Build vintage
    vintage = build_vintage(panel, origin, settings.publication_calendar)

    if save_vintages:
        _save_vintage(vintage, origin, settings.paths.vintages_dir)

    # 2. Rolling estimation window
    window_start = max(
        origin - pd.DateOffset(years=rolling_yrs),
        full_start,
    )
    train = vintage.loc[window_start:origin]

    if len(train.dropna(how="all")) < 24:
        raise ValueError(f"Insufficient training data: only {len(train)} rows")

    # 3. Fit DFM on rolling window
    dfm = DynamicFactorModel(
        n_factors=settings.model.n_factors,
        n_lags=settings.model.n_lags,
        settings=settings.model,
    )
    dfm.fit(train)

    # 4. Identify target quarter first so we can extend the panel to cover it
    target_q = _next_unreleased_quarter(origin, vintage)

    # 5. Extend panel to include target quarter (needed for out-of-sample prediction)
    nowcast_panel = vintage.loc[:origin]
    if target_q is not None and len(nowcast_panel) > 0 and target_q > nowcast_panel.index[-1]:
        future_idx = pd.date_range(
            nowcast_panel.index[-1] + pd.offsets.MonthBegin(1),
            target_q,
            freq="MS",
        )
        future_df = pd.DataFrame(np.nan, index=future_idx, columns=vintage.columns)
        nowcast_panel = pd.concat([nowcast_panel, future_df])

    nc = dfm.nowcast(nowcast_panel)

    if len(nc) == 0:
        raise ValueError("Nowcast returned empty series")

    # 6. Extract nowcast value at the target quarter (or fallback to last)
    nc_val   = _get_nowcast_at_target(nc, target_q)
    ci_lo_s  = nc.attrs.get("ci_lower", pd.Series(dtype=float))
    ci_hi_s  = nc.attrs.get("ci_upper", pd.Series(dtype=float))
    ci_lower = _get_nowcast_at_target(ci_lo_s, target_q) if len(ci_lo_s) > 0 else float("nan")
    ci_upper = _get_nowcast_at_target(ci_hi_s, target_q) if len(ci_hi_s) > 0 else float("nan")

    # 7. Realized value (from full, un-censored panel)
    if target_q is not None and target_q in panel.index:
        realized = float(panel.loc[target_q, "va_construction"])
    else:
        realized = float("nan")

    error = nc_val - realized if not np.isnan(realized) else float("nan")

    return {
        "origin":         origin,
        "target_quarter": target_q,
        "nowcast":        nc_val,
        "ci_lower":       ci_lower,
        "ci_upper":       ci_upper,
        "realized":       realized,
        "error":          error,
        "status":         "OK",
        "log_likelihood": dfm.results_.log_likelihood if dfm.results_ else float("nan"),
    }


def _generate_origins(
    start: pd.Timestamp, end: pd.Timestamp, update_days: list[int]
) -> list[pd.Timestamp]:
    """Generate all forecast origin timestamps."""
    origins = []
    current = start.replace(day=1)
    while current <= end:
        for day in sorted(update_days):
            try:
                ts = current.replace(day=day)
                if ts <= end:
                    origins.append(ts)
            except ValueError:
                pass  # invalid day for this month (e.g., day=31 in April)
        current = (current + pd.offsets.MonthBegin(1))
    return sorted(origins)


def _get_nowcast_at_target(nc: pd.Series, target_q: pd.Timestamp | None) -> float:
    """Extract nowcast value at target_q, matching by year-month regardless of day."""
    if len(nc) == 0:
        return float("nan")
    if target_q is None:
        return float(nc.iloc[-1])
    for idx_val in nc.index:
        if idx_val.year == target_q.year and idx_val.month == target_q.month:
            return float(nc.loc[idx_val])
    return float(nc.iloc[-1])


def _next_unreleased_quarter(
    origin: pd.Timestamp, vintage: pd.DataFrame
) -> pd.Timestamp | None:
    """Find the next quarter-end for which va_construction is NaN in the vintage."""
    if "va_construction" not in vintage.columns:
        return None

    va = vintage["va_construction"]
    # Quarter-end months: 3, 6, 9, 12
    quarter_ends = va[va.index.month.isin([3, 6, 9, 12])]
    unreleased   = quarter_ends[quarter_ends.isna() & (quarter_ends.index >= origin)]

    return unreleased.index[0] if len(unreleased) > 0 else None


def _save_vintage(
    vintage: pd.DataFrame, origin: pd.Timestamp, vintages_dir: Path
) -> None:
    """Save a single vintage to parquet."""
    vintages_dir = Path(vintages_dir)
    vintages_dir.mkdir(parents=True, exist_ok=True)
    path = vintages_dir / f"{origin.strftime('%Y-%m-%d')}.parquet"
    vintage.to_parquet(path, engine="pyarrow")
    logger.debug("Saved vintage %s", path)


def _save_results(df: pd.DataFrame, vintages_dir: Path) -> None:
    """Save consolidated backtest results."""
    vintages_dir = Path(vintages_dir)
    vintages_dir.mkdir(parents=True, exist_ok=True)
    path = vintages_dir / "backtest_results.parquet"
    df.to_parquet(path, engine="pyarrow")
    logger.info("Saved backtest results to %s (%d rows)", path, len(df))
