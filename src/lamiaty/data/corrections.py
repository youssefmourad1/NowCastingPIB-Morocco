"""
Data corrections — critical fixes required before DFM estimation.

Each correction is an isolated, testable function. Parameters are driven by
configs/corrections.yaml via CorrectionSettings.

Corrections implemented:
  1. fix_cement_unit_break     — structural break in April 2022 (×759 factor)
  2. fix_investissement_etat   — convert likely YTD cumulative series into monthly flow
  3. fix_lafarge_strings       — string → float
  4. apply_all_corrections     — orchestrates all corrections
"""

from __future__ import annotations

import logging
import warnings
from typing import Literal

import numpy as np
import pandas as pd

from lamiaty.config.settings import CorrectionSettings
from lamiaty.data.loader import (
    COL_CEMENT,
    COL_INVESTISSEMENT,
    COL_LAFARGE,
)

logger = logging.getLogger(__name__)


# ---------------------------------------------------------------------------
# 1. Cement unit break correction
# ---------------------------------------------------------------------------


def fix_cement_unit_break(
    series: pd.Series,
    break_date: str,
    factor: float,
    confirmed_by: str | None = None,
) -> pd.Series:
    """
    Correct the structural unit break in consommation_ciment at break_date.

    Multiplies all values BEFORE break_date by factor to bring them into the
    same unit as the post-break values.
    """
    if not isinstance(series, pd.Series):
        raise TypeError("series must be a pandas Series")

    if not isinstance(factor, (int, float)) or factor <= 0:
        raise TypeError(f"correction_factor must be a positive number, got {factor!r}")

    if not isinstance(series.index, pd.DatetimeIndex):
        raise TypeError("fix_cement_unit_break requires a DatetimeIndex")

    if confirmed_by is None:
        warnings.warn(
            f"Cement break correction factor ({factor}) has not been confirmed with APC. "
            "Set corrections.yaml cement_break.confirmed_by before treating output as "
            "production-ready. Pipeline continues with the placeholder value.",
            UserWarning,
            stacklevel=2,
        )

    corrected = pd.to_numeric(series.copy(), errors="coerce")
    break_ts = pd.Timestamp(break_date)

    pre_break_mask = corrected.index < break_ts
    n_corrected = int(pre_break_mask.sum())

    corrected.loc[pre_break_mask] = corrected.loc[pre_break_mask] * factor
    corrected.name = series.name

    logger.info(
        "Cement break correction: ×%.3f applied to %d values before %s",
        factor,
        n_corrected,
        break_date,
    )
    return corrected


# ---------------------------------------------------------------------------
# 2. Investissement_Etat correction
# ---------------------------------------------------------------------------


def _compute_monthly_flow_from_ytd(series: pd.Series) -> pd.Series:
    """
    Convert a likely cumulative YTD series into monthly flow.

    Logic:
      - For January: monthly_flow = January level
      - For Feb..Dec: monthly_flow = YTD_t - YTD_{t-1} within the same year

    This avoids the artificial large January jumps caused by differencing across
    December of year t-1 and January of year t.
    """
    if not isinstance(series.index, pd.DatetimeIndex):
        raise TypeError("_compute_monthly_flow_from_ytd requires a DatetimeIndex")

    s = pd.to_numeric(series.copy(), errors="coerce").sort_index()
    out = pd.Series(index=s.index, dtype="float64", name=s.name)

    for year, group in s.groupby(s.index.year):
        g = group.sort_index()

        if g.empty:
            continue

        # January = first observed cumulative value of the year
        out.loc[g.index[0]] = g.iloc[0]

        # Remaining months = within-year differences
        if len(g) > 1:
            out.loc[g.index[1:]] = g.diff().iloc[1:]

        logger.debug(
            "Investissement_Etat YTD→monthly_flow | year=%s | obs=%d",
            year,
            len(g),
        )

    return out


def _diagnose_january_pattern(series: pd.Series) -> dict[str, float | int]:
    """
    Simple diagnostics for January anomaly detection.
    """
    s = pd.to_numeric(series, errors="coerce").dropna()
    if not isinstance(s.index, pd.DatetimeIndex):
        raise TypeError("_diagnose_january_pattern requires a DatetimeIndex")

    jan = s[s.index.month == 1]
    non_jan = s[s.index.month != 1]

    jan_negative_share = float((jan < 0).mean()) if len(jan) else np.nan
    jan_mean = float(jan.mean()) if len(jan) else np.nan
    non_jan_mean = float(non_jan.mean()) if len(non_jan) else np.nan

    return {
        "n_january": int(len(jan)),
        "jan_negative_share": jan_negative_share,
        "jan_mean": jan_mean,
        "non_jan_mean": non_jan_mean,
    }


def fix_investissement_etat(
    series: pd.Series,
    method: Literal["monthly_diff", "keep_as_is", "ytd_to_monthly_flow"] = "ytd_to_monthly_flow",
    confirmed_by: str | None = None,
) -> pd.Series:
    """
    Transform Investissement_Etat to address atypical January values.

    Methods:
      - ytd_to_monthly_flow:
          Recommended. Treat the input as a cumulative year-to-date series and
          convert it to monthly flows year by year.
      - monthly_diff:
          Plain first difference over the full series. Kept for comparison and
          backward compatibility, but less appropriate if the series resets each January.
      - keep_as_is:
          No correction.

    Returns:
        Corrected/transformed series.
    """
    if not isinstance(series, pd.Series):
        raise TypeError("series must be a pandas Series")

    if not isinstance(series.index, pd.DatetimeIndex):
        raise TypeError("fix_investissement_etat requires a DatetimeIndex")

    s = pd.to_numeric(series.copy(), errors="coerce").sort_index()
    s.name = series.name

    if confirmed_by is None:
        warnings.warn(
            f"Investissement_Etat treatment ('{method}') has not been confirmed with TGR/MEF. "
            "Proceeding with a statistical correction, but external confirmation remains recommended.",
            UserWarning,
            stacklevel=2,
        )

    diag_before = _diagnose_january_pattern(s)
    logger.info(
        "Investissement_Etat raw diagnostics | n_january=%d | jan_negative_share=%.2f | "
        "jan_mean=%.3f | non_jan_mean=%.3f",
        diag_before["n_january"],
        diag_before["jan_negative_share"],
        diag_before["jan_mean"],
        diag_before["non_jan_mean"],
    )

    if method == "ytd_to_monthly_flow":
        result = _compute_monthly_flow_from_ytd(s)
        logger.info(
            "Investissement_Etat: applied ytd_to_monthly_flow "
            "(January kept as level, Feb-Dec as within-year differences)"
        )

    elif method == "monthly_diff":
        result = s.diff(1)
        logger.info("Investissement_Etat: applied monthly_diff (plain first difference)")

    elif method == "keep_as_is":
        warnings.warn(
            "Investissement_Etat kept as-is. Any January reset pattern will propagate "
            "to downstream analysis and may hurt model performance.",
            UserWarning,
            stacklevel=2,
        )
        result = s.copy()
        logger.info("Investissement_Etat: kept raw values unchanged")

    else:
        raise ValueError(
            f"Unknown method: {method!r}. Expected "
            "'ytd_to_monthly_flow', 'monthly_diff', or 'keep_as_is'."
        )

    diag_after = _diagnose_january_pattern(result)
    logger.info(
        "Investissement_Etat corrected diagnostics | n_january=%d | jan_negative_share=%.2f | "
        "jan_mean=%.3f | non_jan_mean=%.3f",
        diag_after["n_january"],
        diag_after["jan_negative_share"],
        diag_after["jan_mean"],
        diag_after["non_jan_mean"],
    )

    result.name = series.name
    return result


# ---------------------------------------------------------------------------
# 3. LafargeHolcim string → float
# ---------------------------------------------------------------------------


def fix_lafarge_strings(series: pd.Series) -> pd.Series:
    """
    Convert the LafargeHolcim index from comma-separated strings to float.
    """
    if not isinstance(series, pd.Series):
        raise TypeError("series must be a pandas Series")

    if pd.api.types.is_float_dtype(series):
        logger.debug("Lafarge series already float — skipping conversion")
        return series.copy()

    cleaned = series.copy().astype(str)
    cleaned = cleaned.str.replace(",", "", regex=False).str.strip()
    cleaned = cleaned.replace({"nan": np.nan, "None": np.nan, "": np.nan})

    result = pd.to_numeric(cleaned, errors="coerce")

    original_non_null = series.notna()
    newly_nan = result.isna() & original_non_null

    if newly_nan.any():
        bad_values = series[newly_nan].tolist()
        raise ValueError(
            f"fix_lafarge_strings: could not parse {len(bad_values)} values after comma removal: "
            f"{bad_values[:5]}"
        )

    result.name = series.name
    logger.info("Lafarge string fix: converted %d non-null values to float", int(original_non_null.sum()))
    return result


# ---------------------------------------------------------------------------
# 4. Orchestrator
# ---------------------------------------------------------------------------


def apply_all_corrections(df: pd.DataFrame, settings: CorrectionSettings) -> pd.DataFrame:
    """
    Apply all corrections to the raw BTP DataFrame.

    Order:
      1. fix_lafarge_strings
      2. fix_cement_unit_break
      3. fix_investissement_etat
    """
    if not isinstance(df, pd.DataFrame):
        raise TypeError("df must be a pandas DataFrame")

    out = df.copy()

    # 1. Lafarge strings → float
    if COL_LAFARGE in out.columns:
        out[COL_LAFARGE] = fix_lafarge_strings(out[COL_LAFARGE])

    # 2. Cement break
    if COL_CEMENT in out.columns:
        cb = settings.cement_break
        out[COL_CEMENT] = fix_cement_unit_break(
            out[COL_CEMENT],
            break_date=cb.break_date,
            factor=cb.correction_factor,
            confirmed_by=cb.confirmed_by,
        )

    # 3. Investissement_Etat
    if COL_INVESTISSEMENT in out.columns:
        inv = settings.investissement_etat
        out[COL_INVESTISSEMENT] = fix_investissement_etat(
            out[COL_INVESTISSEMENT],
            method=inv.treatment,
            confirmed_by=inv.confirmed_by,
        )

    logger.info("All corrections applied successfully")
    return out