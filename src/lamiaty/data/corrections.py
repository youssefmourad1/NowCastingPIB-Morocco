"""
Data corrections — three critical fixes required before DFM estimation.

Each correction is an isolated, testable function. Parameters are driven by
configs/corrections.yaml via CorrectionSettings — no magic numbers here.

Corrections implemented (see §3 of Morocco BTP Nowcasting Implementation Plan):
  1. fix_cement_unit_break     — §3.1 CRITIQUE: structural break in April 2022 (×759 factor)
  2. fix_investissement_etat   — §3.2 IMPORTANT: atypical negative January values
  3. fix_lafarge_strings       — pre-requisite for fix #1/model: string → float
  4. apply_all_corrections     — orchestrates all three
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
    """Correct the structural unit break in Consommation_ciment at break_date.

    Multiplies all values BEFORE break_date by factor to bring them into the
    same unit as the post-break values.

    Context (§3.1): The cement series shows a ~759× jump in April 2022, most
    likely due to a change of unit (thousands of tonnes → tonnes) or a scope
    change at APC. The correction_factor of 759 is a PLACEHOLDER — it must be
    confirmed with the Association Professionnelle des Cimentiers (APC) before
    the model output can be treated as production-ready.

    Args:
        series: Raw cement consumption series with DatetimeIndex.
        break_date: First date of the new unit (values from this date onward are
                    kept as-is). Format: "YYYY-MM-DD".
        factor: Multiplicative correction applied to pre-break values.
        confirmed_by: Name of the source/analyst who confirmed the factor.
                      If None, a UserWarning is emitted.

    Returns:
        Corrected Series with the same index and name.
    """
    if not isinstance(factor, (int, float)) or factor <= 0:
        raise TypeError(f"correction_factor must be a positive number, got {factor!r}")

    if confirmed_by is None:
        warnings.warn(
            f"Cement break correction factor ({factor}) has not been confirmed with APC. "
            "Set corrections.yaml cement_break.confirmed_by before treating output as "
            "production-ready. Pipeline continues with the placeholder value.",
            UserWarning,
            stacklevel=2,
        )

    break_ts = pd.Timestamp(break_date)
    corrected = series.copy()
    pre_break_mask = corrected.index < break_ts
    n_corrected = pre_break_mask.sum()
    corrected.loc[pre_break_mask] = corrected.loc[pre_break_mask] * factor

    logger.info(
        "Cement break correction: ×%.1f applied to %d pre-%s values",
        factor,
        n_corrected,
        break_date,
    )
    return corrected


# ---------------------------------------------------------------------------
# 2. Investissement_Etat correction
# ---------------------------------------------------------------------------


def fix_investissement_etat(
    series: pd.Series,
    method: Literal["monthly_diff", "keep_as_is"] = "monthly_diff",
    confirmed_by: str | None = None,
) -> pd.Series:
    """Transform Investissement_Etat to address the atypical January values.

    Context (§3.2): The series shows very large negative values every January
    (e.g., -53 272 MDH in Jan 2019), consistent with a YTD cumulative series
    that resets at year-end. If the series is indeed a YTD cumulative, applying
    first-differences converts it to monthly flows.

    IMPORTANT: This series is excluded from the DFM (include_in_model: false)
    until its definition is confirmed with TGR/MEF.

    Args:
        series: Raw Investissement_Etat series with DatetimeIndex.
        method: Transformation to apply.
            "monthly_diff" — first difference (.diff(1)), converts YTD → monthly flow.
            "keep_as_is"   — no transformation (triggers a UserWarning).
        confirmed_by: Name of the source/analyst who confirmed the treatment.

    Returns:
        Transformed Series (first row will be NaN if method="monthly_diff").
    """
    if confirmed_by is None:
        warnings.warn(
            f"Investissement_Etat treatment ('{method}') has not been confirmed with TGR/MEF. "
            "The series is excluded from the DFM until confirmed. "
            "Set corrections.yaml investissement_etat.confirmed_by to suppress this warning.",
            UserWarning,
            stacklevel=2,
        )

    if method == "monthly_diff":
        result = series.diff(1)
        logger.info("Investissement_Etat: applied monthly_diff (first difference)")
        return result
    elif method == "keep_as_is":
        warnings.warn(
            "Investissement_Etat: method='keep_as_is' — raw values retained without transformation. "
            "Negative January values will propagate to downstream analysis.",
            UserWarning,
            stacklevel=2,
        )
        return series.copy()
    else:
        raise ValueError(f"Unknown method: {method!r}. Expected 'monthly_diff' or 'keep_as_is'.")


# ---------------------------------------------------------------------------
# 3. LafargeHolcim string → float
# ---------------------------------------------------------------------------


def fix_lafarge_strings(series: pd.Series) -> pd.Series:
    """Convert the LafargeHolcim index from comma-separated strings to float.

    The raw Excel column contains values like "1,612" or "2,445" (thousands
    separator). This function strips commas and casts to float.

    Args:
        series: Raw LafargeHolcim series (dtype object).

    Returns:
        Series with float dtype.

    Raises:
        ValueError: If any non-null value cannot be parsed after comma removal.
    """
    if pd.api.types.is_float_dtype(series):
        logger.debug("Lafarge series is already float — skipping string fix")
        return series.copy()

    cleaned = series.copy().astype(str)
    cleaned = cleaned.str.replace(",", "", regex=False).str.strip()
    # Replace "nan"/"None" strings back to NaN
    cleaned = cleaned.replace({"nan": np.nan, "None": np.nan, "": np.nan})

    result = pd.to_numeric(cleaned, errors="coerce")

    # Check for values that failed to parse (were not already NaN in the original)
    original_non_null = series.notna()
    newly_nan = result.isna() & original_non_null
    if newly_nan.any():
        bad_values = series[newly_nan].tolist()
        raise ValueError(
            f"fix_lafarge_strings: could not parse {len(bad_values)} values after comma removal: "
            f"{bad_values[:5]}"
        )

    logger.info("Lafarge string fix: converted %d values to float", original_non_null.sum())
    return result


# ---------------------------------------------------------------------------
# 4. Orchestrator
# ---------------------------------------------------------------------------


def apply_all_corrections(df: pd.DataFrame, settings: CorrectionSettings) -> pd.DataFrame:
    """Apply all three data corrections to the raw BTP DataFrame.

    Corrections applied in order:
      1. fix_lafarge_strings     (must run before yoy transforms)
      2. fix_cement_unit_break
      3. fix_investissement_etat

    Args:
        df: Raw DataFrame from loader.load_base_btp().
        settings: CorrectionSettings loaded from corrections.yaml.

    Returns:
        Corrected DataFrame with the same index and column structure.
    """
    df = df.copy()

    # 1. Lafarge strings → float
    if COL_LAFARGE in df.columns:
        df[COL_LAFARGE] = fix_lafarge_strings(df[COL_LAFARGE])

    # 2. Cement break
    if COL_CEMENT in df.columns:
        cb = settings.cement_break
        df[COL_CEMENT] = fix_cement_unit_break(
            df[COL_CEMENT],
            break_date=cb.break_date,
            factor=cb.correction_factor,
            confirmed_by=cb.confirmed_by,
        )

    # 3. Investissement_Etat
    if COL_INVESTISSEMENT in df.columns:
        inv = settings.investissement_etat
        df[COL_INVESTISSEMENT] = fix_investissement_etat(
            df[COL_INVESTISSEMENT],
            method=inv.treatment,
            confirmed_by=inv.confirmed_by,
        )

    logger.info("All corrections applied successfully")
    return df
