"""
Unit root diagnostics — run before DFM estimation to verify stationarity.

Wraps statsmodels ADF and KPSS tests and runs them across all series in the panel.
Required before Phase 2 estimation: the DFM assumes covariance-stationary inputs.
"""

from __future__ import annotations

import logging
import warnings
from typing import Any

import pandas as pd
from statsmodels.tsa.stattools import adfuller, kpss

logger = logging.getLogger(__name__)

# Significance level for test decisions
ALPHA = 0.05


def run_adf_test(series: pd.Series, max_lags: int = 12) -> dict[str, Any]:
    """Augmented Dickey-Fuller test for a unit root.

    H0: Series has a unit root (non-stationary).
    Reject H0 (p < alpha) → series is stationary.

    Args:
        series: Numeric series. NaNs are dropped before testing.
        max_lags: Maximum number of lags to consider (autolag='AIC' selects optimally).

    Returns:
        Dict with keys: statistic, pvalue, n_lags_used, critical_values, is_stationary.
    """
    clean = series.dropna()
    if len(clean) < 20:
        logger.warning("ADF test on '%s': only %d non-NaN values — result unreliable", series.name, len(clean))
    result = adfuller(clean, maxlag=max_lags, autolag="AIC")
    return {
        "statistic": result[0],
        "pvalue": result[1],
        "n_lags_used": result[2],
        "n_obs": result[3],
        "critical_values": result[4],
        "is_stationary": result[1] < ALPHA,
    }


def run_kpss_test(series: pd.Series) -> dict[str, Any]:
    """KPSS test for stationarity.

    H0: Series is stationary (level or trend).
    Reject H0 (p < alpha) → series has a unit root.

    Note: KPSS has the opposite null hypothesis from ADF. Use both together:
      ADF: fail to reject + KPSS: fail to reject → stationary (agreed)
      ADF: reject + KPSS: fail to reject → stationary (agreed)
      ADF: fail to reject + KPSS: reject → unit root (agreed)
      ADF: reject + KPSS: reject → ambiguous (possible structural break or regime change)

    Args:
        series: Numeric series. NaNs are dropped before testing.

    Returns:
        Dict with keys: statistic, pvalue, n_lags_used, critical_values, is_stationary.
    """
    clean = series.dropna()
    if len(clean) < 20:
        logger.warning("KPSS test on '%s': only %d non-NaN values — result unreliable", series.name, len(clean))
    with warnings.catch_warnings():
        warnings.simplefilter("ignore")  # suppress lags truncation warning
        result = kpss(clean, regression="c", nlags="auto")
    return {
        "statistic": result[0],
        "pvalue": result[1],
        "n_lags_used": result[2],
        "critical_values": result[3],
        "is_stationary": result[1] > ALPHA,  # fail to reject H0 → stationary
    }


def run_stationarity_battery(panel: pd.DataFrame) -> pd.DataFrame:
    """Run ADF and KPSS tests on every column of the panel.

    Args:
        panel: Mixed-frequency panel (output of build_mixed_frequency_panel).

    Returns:
        DataFrame with one row per series and columns:
          series, n_obs, ADF_stat, ADF_pvalue, ADF_stationary,
          KPSS_stat, KPSS_pvalue, KPSS_stationary, verdict
    """
    rows = []
    for col in panel.columns:
        series = panel[col].dropna()
        n_obs = len(series)

        adf = run_adf_test(series)
        kpss_res = run_kpss_test(series)

        # Determine consensus verdict
        if adf["is_stationary"] and kpss_res["is_stationary"]:
            verdict = "STATIONARY"
        elif not adf["is_stationary"] and not kpss_res["is_stationary"]:
            verdict = "UNIT_ROOT"
        elif adf["is_stationary"] and not kpss_res["is_stationary"]:
            verdict = "AMBIGUOUS (ADF: stationary, KPSS: unit root)"
        else:
            verdict = "AMBIGUOUS (ADF: unit root, KPSS: stationary)"

        rows.append({
            "series": col,
            "n_obs": n_obs,
            "ADF_stat": round(adf["statistic"], 4),
            "ADF_pvalue": round(adf["pvalue"], 4),
            "ADF_stationary": adf["is_stationary"],
            "KPSS_stat": round(kpss_res["statistic"], 4),
            "KPSS_pvalue": round(kpss_res["pvalue"], 4),
            "KPSS_stationary": kpss_res["is_stationary"],
            "verdict": verdict,
        })
        logger.debug("Stationarity: '%s' → %s", col, verdict)

    result = pd.DataFrame(rows).set_index("series")
    n_stationary = (result["verdict"] == "STATIONARY").sum()
    logger.info(
        "Stationarity battery: %d/%d series are stationary",
        n_stationary,
        len(result),
    )
    return result
