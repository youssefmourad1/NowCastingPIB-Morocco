"""
Unit root diagnostics — run before DFM estimation to verify stationarity.

This module:
1. Applies optional transformations to raw series
   (level, diff, log, log_diff, pct_change)
2. Runs ADF and KPSS tests
3. Produces a stationarity summary for each series

Useful before DFM estimation because the DFM assumes
approximately covariance-stationary inputs.
"""

from __future__ import annotations

import logging
import warnings
from typing import Any

import numpy as np
import pandas as pd
from statsmodels.tsa.stattools import adfuller, kpss

logger = logging.getLogger(__name__)

# Significance level
ALPHA = 0.05

# -------------------------------------------------------------------
# 1) CONFIGURATION
# -------------------------------------------------------------------

# Series that usually exhibit deterministic trend in levels
TREND_SERIES = {
    "va_construction",
    "credits_immobilier",
    "credits_equipement",
    "consommation_ciment",
    "lafarge_index",
    "investissement_etat",
}

# Recommended default transformation by variable
# Adjust names if your actual column names differ
TRANSFORM_RULES = {
    "va_construction": "log_diff",
    "credits_immobilier": "log_diff",
    "credits_equipement": "log_diff",
    "consommation_ciment": "log_diff",
    "lafarge_index": "diff",
    "ipai": "diff",
    "creation_emploi": "diff",
    "investissement_etat": "diff",
}

# Minimum sample size warning threshold
MIN_OBS_WARNING = 20


# -------------------------------------------------------------------
# 2) HELPER: PREPARE SERIES
# -------------------------------------------------------------------

def _coerce_numeric(series: pd.Series) -> pd.Series:
    """Ensure numeric dtype and preserve series name."""
    out = pd.to_numeric(series, errors="coerce")
    out.name = series.name
    return out


def _safe_log(series: pd.Series) -> pd.Series:
    """
    Safe log transform.
    Replaces non-positive values with NaN before taking logs.
    """
    s = _coerce_numeric(series).copy()
    s[s <= 0] = np.nan
    out = np.log(s)
    out.name = series.name
    return out


def make_stationary_transform(series: pd.Series, method: str = "level") -> pd.Series:
    """
    Apply a transformation intended to improve stationarity.

    Supported methods:
      - level
      - diff
      - log
      - log_diff
      - pct_change
    """
    s = _coerce_numeric(series).copy()

    if method == "level":
        out = s

    elif method == "diff":
        out = s.diff()

    elif method == "log":
        out = _safe_log(s)

    elif method == "log_diff":
        out = _safe_log(s).diff()

    elif method == "pct_change":
        out = s.pct_change()

    else:
        raise ValueError(f"Unknown transformation method: {method}")

    out.name = series.name
    return out


def infer_regression_type(series_name: str, method: str = "level") -> str:
    """
    Choose deterministic component for ADF/KPSS.
    'ct' = constant + trend
    'c'  = constant only

    If we have already differenced the series, we usually only need a constant.
    """
    if method in ["diff", "log_diff", "pct_change"]:
        return "c"
    return "ct" if series_name in TREND_SERIES else "c"


# -------------------------------------------------------------------
# 3) ADF / KPSS TESTS
# -------------------------------------------------------------------

def run_adf_test(
    series: pd.Series,
    max_lags: int = 6,
    regression: str = "c",
) -> dict[str, Any]:
    """
    Augmented Dickey-Fuller test.

    H0: Series has a unit root (non-stationary)
    Reject H0 (p < alpha) => stationary
    """
    clean = _coerce_numeric(series).dropna()

    if len(clean) < MIN_OBS_WARNING:
        logger.warning(
            "ADF test on '%s': only %d non-NaN values — result may be unreliable",
            series.name,
            len(clean),
        )

    if len(clean) < 10:
        raise ValueError(f"ADF test requires more observations for series '{series.name}'")

    result = adfuller(
        clean,
        maxlag=max_lags,
        autolag="AIC",
        regression=regression,
    )

    return {
        "statistic": result[0],
        "pvalue": result[1],
        "n_lags_used": result[2],
        "n_obs": result[3],
        "critical_values": result[4],
        "is_stationary": result[1] < ALPHA,
    }


def run_kpss_test(
    series: pd.Series,
    regression: str = "c",
) -> dict[str, Any]:
    """
    KPSS test.

    H0: Series is stationary
    Reject H0 (p < alpha) => non-stationary
    """
    clean = _coerce_numeric(series).dropna()

    if len(clean) < MIN_OBS_WARNING:
        logger.warning(
            "KPSS test on '%s': only %d non-NaN values — result may be unreliable",
            series.name,
            len(clean),
        )

    if len(clean) < 10:
        raise ValueError(f"KPSS test requires more observations for series '{series.name}'")

    with warnings.catch_warnings():
        warnings.simplefilter("ignore")
        result = kpss(clean, regression=regression, nlags="auto")

    return {
        "statistic": result[0],
        "pvalue": result[1],
        "n_lags_used": result[2],
        "critical_values": result[3],
        "is_stationary": result[1] > ALPHA,  # fail to reject H0
    }


# -------------------------------------------------------------------
# 4) VERDICT LOGIC
# -------------------------------------------------------------------

def combine_adf_kpss_verdict(adf_stationary: bool, kpss_stationary: bool) -> str:
    """
    Combine ADF and KPSS conclusions into a final verdict.
    """
    if adf_stationary and kpss_stationary:
        return "STATIONARY"

    if (not adf_stationary) and (not kpss_stationary):
        return "UNIT_ROOT"

    if adf_stationary and (not kpss_stationary):
        return "AMBIGUOUS (ADF stationary, KPSS non-stationary)"

    return "AMBIGUOUS (ADF non-stationary, KPSS stationary)"


# -------------------------------------------------------------------
# 5) MAIN BATTERY
# -------------------------------------------------------------------

def run_stationarity_battery(
    panel: pd.DataFrame,
    transform_rules: dict[str, str] | None = None,
    max_lags_adf: int = 6,
) -> pd.DataFrame:
    """
    Run ADF + KPSS on each column of a DataFrame.

    Parameters
    ----------
    panel : pd.DataFrame
        Input panel containing raw series.
    transform_rules : dict[str, str] | None
        Mapping column -> transformation method.
        If None, uses TRANSFORM_RULES and defaults to 'level'.
    max_lags_adf : int
        Maximum lag length passed to ADF.

    Returns
    -------
    pd.DataFrame
        Summary table with one row per series.
    """
    if transform_rules is None:
        transform_rules = TRANSFORM_RULES.copy()

    rows: list[dict[str, Any]] = []

    for col in panel.columns:
        raw_series = panel[col]
        method = transform_rules.get(col, "level")
        regression_type = infer_regression_type(col)

        try:
            transformed = make_stationary_transform(raw_series, method).dropna()
            n_obs = len(transformed)

            adf_res = run_adf_test(
                transformed,
                max_lags=max_lags_adf,
                regression=regression_type,
            )

            kpss_res = run_kpss_test(
                transformed,
                regression=regression_type,
            )

            verdict = combine_adf_kpss_verdict(
                adf_stationary=adf_res["is_stationary"],
                kpss_stationary=kpss_res["is_stationary"],
            )

            rows.append({
                "series": col,
                "transformation": method,
                "regression": regression_type,
                "n_obs": n_obs,
                "ADF_stat": round(adf_res["statistic"], 4),
                "ADF_pvalue": round(adf_res["pvalue"], 4),
                "ADF_stationary": adf_res["is_stationary"],
                "KPSS_stat": round(kpss_res["statistic"], 4),
                "KPSS_pvalue": round(kpss_res["pvalue"], 4),
                "KPSS_stationary": kpss_res["is_stationary"],
                "verdict": verdict,
            })

            logger.info(
                "Series '%s' | transform=%s | regression=%s | verdict=%s",
                col, method, regression_type, verdict
            )

        except Exception as exc:
            logger.exception("Stationarity test failed for '%s': %s", col, exc)
            rows.append({
                "series": col,
                "transformation": method,
                "regression": regression_type,
                "n_obs": np.nan,
                "ADF_stat": np.nan,
                "ADF_pvalue": np.nan,
                "ADF_stationary": np.nan,
                "KPSS_stat": np.nan,
                "KPSS_pvalue": np.nan,
                "KPSS_stationary": np.nan,
                "verdict": f"ERROR: {exc}",
            })

    result = pd.DataFrame(rows).set_index("series")

    n_stationary = (result["verdict"] == "STATIONARY").sum()
    logger.info(
        "Stationarity battery completed: %d/%d series classified as STATIONARY",
        n_stationary,
        len(result),
    )

    return result


# -------------------------------------------------------------------
# 6) OPTIONAL: COMPARE MULTIPLE TRANSFORMATIONS FOR ONE SERIES
# -------------------------------------------------------------------

def compare_transformations_for_series(
    series: pd.Series,
    methods: list[str] | None = None,
    max_lags_adf: int = 6,
) -> pd.DataFrame:
    """
    Useful to decide which transformation works best for one variable.
    """
    if methods is None:
        methods = ["level", "diff", "log", "log_diff", "pct_change"]

    rows = []
    regression_type = infer_regression_type(series.name or "")

    for method in methods:
        try:
            transformed = make_stationary_transform(series, method).dropna()

            adf_res = run_adf_test(
                transformed,
                max_lags=max_lags_adf,
                regression=regression_type,
            )
            kpss_res = run_kpss_test(
                transformed,
                regression=regression_type,
            )

            verdict = combine_adf_kpss_verdict(
                adf_stationary=adf_res["is_stationary"],
                kpss_stationary=kpss_res["is_stationary"],
            )

            rows.append({
                "method": method,
                "n_obs": len(transformed),
                "ADF_pvalue": round(adf_res["pvalue"], 4),
                "KPSS_pvalue": round(kpss_res["pvalue"], 4),
                "verdict": verdict,
            })

        except Exception as exc:
            rows.append({
                "method": method,
                "n_obs": np.nan,
                "ADF_pvalue": np.nan,
                "KPSS_pvalue": np.nan,
                "verdict": f"ERROR: {exc}",
            })

    return pd.DataFrame(rows)


# -------------------------------------------------------------------
# 7) OPTIONAL: QUICK PRESET FOR YOUR BTP PROJECT
# -------------------------------------------------------------------

def get_btp_transform_rules() -> dict[str, str]:
    """
    Convenient helper for your BTP nowcasting project.
    Adjust column names if needed.
    """
    return {
        "va_construction": "log_diff",
        "credits_immobilier": "log_diff",
        "credits_equipement": "log_diff",
        "consommation_ciment": "log_diff",
        "lafarge_index": "diff",
        "ipai": "diff",
        "creation_emploi": "diff",
        "investissement_etat": "diff",
    }