"""
Forecast evaluation metrics for backtesting the DFM nowcast.

Metrics (§5.3.2 of Implementation Plan):
  - RMSFE: Root Mean Squared Forecast Error
  - MAFE:  Mean Absolute Forecast Error
  - Theil U: RMSFE(model) / RMSFE(benchmark)
  - Diebold-Mariano: Harvey, Leybourne & Newbold (1997) small-sample correction

Benchmarks:
  - random_walk: VA_{t} = VA_{t-4}  (naïve yoy stable)
  - ar1:         AR(1) on quarterly VA CONSTRUCTION
  - bridge_ols:  OLS of VA_BTP on ciment + lafarge_index
"""

from __future__ import annotations

import logging

import numpy as np
import pandas as pd

logger = logging.getLogger(__name__)


# ---------------------------------------------------------------------------
# Core metrics
# ---------------------------------------------------------------------------


def compute_rmsfe(nowcasts: pd.Series, realized: pd.Series) -> float:
    """Root Mean Squared Forecast Error.

    Args:
        nowcasts: Model predictions indexed by quarter-end date.
        realized: Realized values at the same dates.

    Returns:
        RMSFE scalar.
    """
    common = nowcasts.index.intersection(realized.index)
    errors = nowcasts.loc[common] - realized.loc[common]
    valid  = errors.dropna()
    if len(valid) == 0:
        return float("nan")
    return float(np.sqrt((valid ** 2).mean()))


def compute_mafe(nowcasts: pd.Series, realized: pd.Series) -> float:
    """Mean Absolute Forecast Error.

    Args:
        nowcasts: Model predictions indexed by quarter-end date.
        realized: Realized values at the same dates.

    Returns:
        MAFE scalar.
    """
    common = nowcasts.index.intersection(realized.index)
    errors = nowcasts.loc[common] - realized.loc[common]
    valid  = errors.dropna()
    if len(valid) == 0:
        return float("nan")
    return float(valid.abs().mean())


def compute_theil_u(
    nowcasts: pd.Series,
    realized: pd.Series,
    naive_forecasts: pd.Series,
) -> float:
    """Theil U statistic: RMSFE(model) / RMSFE(naive).

    U < 1 means the model beats the naïve benchmark.

    Args:
        nowcasts:        Model predictions.
        realized:        Realized values.
        naive_forecasts: Benchmark predictions (e.g., random walk).

    Returns:
        Theil U scalar (lower is better; 1.0 = same as naive).
    """
    rmsfe_model = compute_rmsfe(nowcasts, realized)
    rmsfe_naive = compute_rmsfe(naive_forecasts, realized)
    if np.isnan(rmsfe_naive) or rmsfe_naive == 0.0:
        return float("nan")
    return float(rmsfe_model / rmsfe_naive)


def diebold_mariano(
    errors_model: pd.Series,
    errors_benchmark: pd.Series,
    h: int = 1,
) -> dict[str, float]:
    """Diebold-Mariano test for equal predictive accuracy.

    Uses the Harvey, Leybourne & Newbold (1997) small-sample correction
    for nowcasting horizons (h=1 quarter ahead).

    H0: equal MSE — two-sided test.
    H1: model MSE ≠ benchmark MSE.

    Args:
        errors_model:     Forecast errors from the DFM.
        errors_benchmark: Forecast errors from the benchmark.
        h:                Forecast horizon in quarters (default 1).

    Returns:
        Dict with keys:
          statistic  — HLN-corrected DM statistic
          pvalue     — two-sided p-value (t_{T-1} distribution)
          n_obs      — number of evaluation periods
    """
    from scipy import stats

    common = errors_model.index.intersection(errors_benchmark.index)
    e1 = errors_model.loc[common].dropna()
    e2 = errors_benchmark.loc[common].dropna()

    # Align on common non-NaN index
    idx = e1.index.intersection(e2.index)
    e1, e2 = e1.loc[idx], e2.loc[idx]
    T = len(idx)

    if T < 4:
        logger.warning("DM test: only %d obs — result unreliable", T)
        return {"statistic": float("nan"), "pvalue": float("nan"), "n_obs": T}

    d     = e1 ** 2 - e2 ** 2
    d_bar = d.mean()

    # HAC variance estimate with bandwidth h-1
    gamma_0 = float(((d - d_bar) ** 2).mean())
    gamma_k = sum(
        float(((d.iloc[k:] - d_bar) * (d.iloc[:-k] - d_bar)).mean())
        for k in range(1, h)
    )
    v_d = (gamma_0 + 2 * gamma_k) / T
    if v_d <= 0:
        return {"statistic": float("nan"), "pvalue": float("nan"), "n_obs": T}

    dm_stat = d_bar / np.sqrt(v_d)

    # Harvey-Leybourne-Newbold small-sample correction
    hln = np.sqrt((T + 1 - 2 * h + h * (h - 1) / T) / T)
    dm_corrected = dm_stat * hln

    pvalue = float(2 * stats.t.sf(abs(dm_corrected), df=T - 1))

    logger.debug(
        "DM test: T=%d, DM_stat=%.3f (HLN-corrected), p=%.4f",
        T, dm_corrected, pvalue,
    )
    return {
        "statistic": float(dm_corrected),
        "pvalue":    pvalue,
        "n_obs":     T,
    }


# ---------------------------------------------------------------------------
# Benchmark forecasts
# ---------------------------------------------------------------------------


def benchmark_random_walk(va_series: pd.Series) -> pd.Series:
    """Naïve yoy-stable benchmark: forecast = same quarter one year ago.

    Args:
        va_series: Quarterly VA CONSTRUCTION series (quarter-end index).

    Returns:
        Series of forecasts aligned to the same index.
    """
    # Shift 4 quarters back (yoy)
    rw = va_series.shift(4)
    rw.name = "random_walk"
    return rw


def benchmark_ar1(va_series: pd.Series) -> pd.Series:
    """AR(1) benchmark estimated recursively on available data.

    At each quarter t, fits AR(1) on all available history up to t-1,
    then forecasts t.

    Args:
        va_series: Quarterly VA CONSTRUCTION series (quarter-end index).

    Returns:
        Series of out-of-sample AR(1) forecasts.
    """
    from statsmodels.tsa.ar_model import AutoReg

    forecasts = pd.Series(index=va_series.index, dtype=float)
    valid = va_series.dropna()

    for i in range(5, len(valid)):
        train = valid.iloc[:i]
        try:
            model = AutoReg(train, lags=1, old_names=False)
            result = model.fit()
            forecasts.loc[valid.index[i]] = float(result.forecast(steps=1).iloc[0])
        except Exception:
            pass

    forecasts.name = "ar1"
    return forecasts


def compute_bridge_ols(
    panel: pd.DataFrame,
    target_col: str = "va_construction",
    predictor_cols: list[str] | None = None,
) -> pd.Series:
    """Bridge equation benchmark: OLS of VA_BTP on monthly indicators.

    Each quarter's forecast uses the average of monthly indicators available
    within that quarter, regressed on the target.  Fit is recursive
    (expanding window, minimum 16 quarters).

    Args:
        panel:          Full model panel (monthly DatetimeIndex).
        target_col:     Name of target column.
        predictor_cols: Monthly indicator columns to use.  Defaults to
                        ['consommation_ciment', 'lafarge_index'].

    Returns:
        Series of bridge-equation forecasts indexed by quarter-end dates.
    """
    import statsmodels.api as sm

    if predictor_cols is None:
        predictor_cols = ["consommation_ciment", "lafarge_index"]

    # Aggregate monthly predictors to quarterly averages
    q_panel = panel[[target_col] + [c for c in predictor_cols if c in panel.columns]].copy()
    q_panel_q = q_panel.resample("QE").mean()
    q_panel_q = q_panel_q.dropna()

    if len(q_panel_q) < 20:
        logger.warning("Bridge OLS: only %d quarterly obs — skipping", len(q_panel_q))
        return pd.Series(dtype=float, name="bridge_ols")

    y = q_panel_q[target_col]
    X = q_panel_q[[c for c in predictor_cols if c in q_panel_q.columns]]
    X = sm.add_constant(X)

    forecasts = pd.Series(index=y.index, dtype=float)
    min_obs = 16

    for i in range(min_obs, len(y)):
        y_train = y.iloc[:i]
        X_train = X.iloc[:i]
        try:
            res  = sm.OLS(y_train, X_train).fit()
            pred = res.predict(X.iloc[[i]])
            forecasts.iloc[i] = float(pred.iloc[0])
        except Exception:
            pass

    forecasts.name = "bridge_ols"
    return forecasts


# ---------------------------------------------------------------------------
# Summary table
# ---------------------------------------------------------------------------


def compute_evaluation_table(
    nowcasts: pd.Series,
    realized: pd.Series,
    benchmarks: dict[str, pd.Series],
) -> pd.DataFrame:
    """Compute full evaluation table comparing DFM to all benchmarks.

    Args:
        nowcasts:   DFM nowcast series.
        realized:   Realized VA CONSTRUCTION values.
        benchmarks: Dict of benchmark_name → forecast series.

    Returns:
        DataFrame with columns [RMSFE, MAFE, TheilU, DM_stat, DM_pvalue]
        and one row per model/benchmark.
    """
    errors_model = nowcasts - realized

    rows = []

    # DFM row
    rows.append({
        "model":    "DFM",
        "RMSFE":    compute_rmsfe(nowcasts, realized),
        "MAFE":     compute_mafe(nowcasts, realized),
        "TheilU":   1.0,   # baseline
        "DM_stat":  float("nan"),
        "DM_pvalue": float("nan"),
    })

    # Benchmark rows
    for name, bench in benchmarks.items():
        errors_bench = bench - realized
        dm = diebold_mariano(errors_model, errors_bench)
        rw_for_theil = benchmarks.get("random_walk", bench)
        rows.append({
            "model":    name,
            "RMSFE":    compute_rmsfe(bench, realized),
            "MAFE":     compute_mafe(bench, realized),
            "TheilU":   compute_theil_u(nowcasts, realized, rw_for_theil),
            "DM_stat":  dm["statistic"],
            "DM_pvalue": dm["pvalue"],
        })

    return pd.DataFrame(rows).set_index("model")
