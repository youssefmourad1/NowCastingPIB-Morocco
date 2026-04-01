"""
News decomposition — Phase 3 implementation.

Implements Equation (11) of Danov et al. (2026):
  E[VA^q_t | Ω_{new}] - E[VA^q_t | Ω_{prev}]
      = Σ_j  weight_j × (y_{j,t_j} - E[y_{j,t_j} | Ω_{prev}])
            ───────────   ────────────────────────────────────
             Kalman gain         News (data surprise)

Uses statsmodels DynamicFactorMQ.news() for the computation.
"""

from __future__ import annotations

import logging
import warnings
from typing import TYPE_CHECKING

import pandas as pd

if TYPE_CHECKING:
    from lamiaty.model.dfm import DynamicFactorModel

logger = logging.getLogger(__name__)


def compute_news_decomposition(
    panel_prev: pd.DataFrame,
    panel_new: pd.DataFrame,
    dfm: "DynamicFactorModel",
    impact_date: str | pd.Timestamp | None = None,
) -> pd.DataFrame:
    """Decompose the nowcast revision into per-variable news contributions.

    Args:
        panel_prev: Panel representing the previous information set Ω_prev.
        panel_new:  Panel representing the new information set Ω_new.
        dfm:        Fitted DynamicFactorModel (must have called fit()).
        impact_date: Quarter-end date for which to attribute the revision.
                     If None, uses the latest available quarter-end in panel_new.

    Returns:
        DataFrame sorted by |contribution| descending, with columns:
          series, update_date, news, weight, contribution
        Plus a summary 'TOTAL' row at the bottom.

    Raises:
        RuntimeError: If dfm is not fitted.
        ValueError:   If no quarterly observations found for va_construction.
    """
    if not dfm._is_fitted:
        raise RuntimeError("DFM must be fitted before computing news decomposition.")

    sm_result = dfm._sm_result

    # ── Determine impact date first ──────────────────────────────────────────
    # Impact date = next unreleased quarter for va_construction.
    # If panel_new has all quarters observed (historical panel), project forward
    # to the first quarter-end beyond panel_new's last date.
    if impact_date is None:
        va_col = "va_construction"
        va_new = panel_new[va_col] if va_col in panel_new.columns else pd.Series(dtype=float)
        q_months = va_new[va_new.index.month.isin([3, 6, 9, 12])]
        unreleased = q_months[q_months.isna()]
        if len(unreleased) > 0:
            impact_target = unreleased.index[0]
        else:
            # All quarters are observed → target the first quarter beyond the panel
            last_date = panel_new.index[-1]
            # Advance to next quarter-end month (3/6/9/12) after last_date
            next_qe = last_date + pd.offsets.MonthBegin(1)
            while next_qe.month not in (3, 6, 9, 12):
                next_qe += pd.offsets.MonthBegin(1)
            impact_target = next_qe
        impact_ts = impact_target.strftime("%Y-%m")
    else:
        impact_target = pd.Timestamp(impact_date)
        # Normalise to month-start
        impact_target = impact_target.replace(day=1)
        impact_ts = impact_target.strftime("%Y-%m")

    logger.info("Computing news decomposition for impact_date=%s", impact_ts)

    # ── Extend both panels to cover impact_target (needed for out-of-sample) ─
    def _extend_to(panel: pd.DataFrame, target: pd.Timestamp) -> pd.DataFrame:
        if len(panel) == 0 or panel.index[-1] >= target:
            return panel
        future_idx = pd.date_range(
            panel.index[-1] + pd.offsets.MonthBegin(1), target, freq="MS"
        )
        return pd.concat([panel, pd.DataFrame(float("nan"), index=future_idx, columns=panel.columns)])

    panel_prev_ext = _extend_to(panel_prev, impact_target)
    panel_new_ext  = _extend_to(panel_new,  impact_target)

    # panel_new must cover at least as many dates as panel_prev
    combined_idx   = panel_prev_ext.index.union(panel_new_ext.index)
    panel_prev_ext = panel_prev_ext.reindex(combined_idx)
    panel_new_ext  = panel_new_ext.reindex(combined_idx)

    # ── Apply fitted model to both information sets ──────────────────────────
    def _apply(panel: pd.DataFrame):
        endog_m = panel[dfm._monthly_cols] if dfm._monthly_cols else panel
        endog_q = dfm._prepare_quarterly(panel, dfm._quarterly_cols)
        kwargs: dict = {"endog": endog_m}
        if len(endog_q) > 0:
            kwargs["endog_quarterly"] = endog_q
        with warnings.catch_warnings(record=True):
            warnings.simplefilter("always")
            return sm_result.apply(**kwargs)

    prev_applied = _apply(panel_prev_ext)
    new_applied  = _apply(panel_new_ext)

    logger.info("Computing news decomposition for impact_date=%s", impact_ts)

    # ── Compute news via statsmodels ─────────────────────────────────────────
    # news() is called on the NEW applied result; prev_applied is the comparison.
    with warnings.catch_warnings(record=True) as caught:
        warnings.simplefilter("always")
        news_result = new_applied.news(
            prev_applied,
            impact_date=impact_ts,
            comparison_type="previous",
        )
    for w in caught:
        logger.warning("statsmodels news: %s", str(w.message))

    # ── Extract details for va_construction ─────────────────────────────────
    try:
        details = news_result.details_by_impact
        details_va = details.xs("va_construction", level="impacted variable")
    except KeyError:
        logger.warning("va_construction not found in news details — returning empty")
        return _empty_news_df()

    records = []
    for idx, row in details_va.iterrows():
        # After xs("va_construction", level="impacted variable"),
        # remaining MultiIndex levels: (impact_date, update_date, updated_variable)
        if isinstance(idx, tuple) and len(idx) >= 3:
            update_date = idx[1]   # 'update date'
            series      = idx[2]   # 'updated variable'
        elif isinstance(idx, tuple) and len(idx) == 2:
            update_date = idx[0]
            series      = idx[1]
        else:
            update_date = idx
            series      = str(idx)

        news_val = float(
            row.get("news", 0.0)
            if "news" in row.index
            else (row.get("observed", 0.0) - row.get("forecast (prev)", 0.0))
        )
        weight_val       = float(row.get("weights", row.get("weight", float("nan"))))
        contribution_val = float(row.get("impact", float("nan")))

        records.append({
            "series":       str(series),
            "update_date":  update_date.to_timestamp() if isinstance(update_date, pd.Period) else pd.Timestamp(update_date),
            "news":         news_val,
            "weight":       weight_val,
            "contribution": contribution_val,
        })

    if not records:
        return _empty_news_df()

    df = pd.DataFrame(records)
    df = df.sort_values("contribution", key=lambda s: s.abs(), ascending=False)

    total_contribution = df["contribution"].sum()
    total_row = pd.DataFrame([{
        "series":       "TOTAL",
        "update_date":  pd.NaT,
        "news":         float("nan"),
        "weight":       float("nan"),
        "contribution": total_contribution,
    }])
    df = pd.concat([df, total_row], ignore_index=True)

    logger.info(
        "News decomposition: %d updates, total revision=%.4f",
        len(df) - 1,
        total_contribution,
    )
    return df


def _empty_news_df() -> pd.DataFrame:
    return pd.DataFrame(
        columns=["series", "update_date", "news", "weight", "contribution"]
    )
