"""
Shared fixtures for DFM model tests.

Provides a small synthetic mixed-frequency panel (60 months, 2015-01 to 2019-12)
generated from a known 1-factor DGP — fast enough for CI and test runs.
"""

from __future__ import annotations

import numpy as np
import pandas as pd
import pytest


@pytest.fixture(scope="module")
def synthetic_panel():
    """60-row mixed-frequency panel for DFM tests.

    DGP: 1 latent factor F_t = 0.8 F_{t-1} + u_t
      monthly_a = 0.7 F_t + ε1     (monthly)
      monthly_b = 0.5 F_t + ε2     (monthly)
      va_q      = 0.9 F_t + ε3     (quarterly, NaN in non-quarter-end months)

    Returns DataFrame with DatetimeIndex (MS), columns:
      monthly_a, monthly_b, va_construction
    """
    np.random.seed(99)
    T = 60
    idx = pd.date_range("2015-01", periods=T, freq="MS")

    # Latent factor
    F = np.zeros(T)
    for t in range(1, T):
        F[t] = 0.8 * F[t - 1] + np.random.randn()

    # Monthly series (yoy-diff already applied so no transforms needed)
    monthly_a = 0.7 * F + 0.5 * np.random.randn(T)
    monthly_b = 0.5 * F + 0.5 * np.random.randn(T)
    va_raw    = 0.9 * F + 0.3 * np.random.randn(T)

    # Quarterly: keep only quarter-end months (Mar, Jun, Sep, Dec), NaN elsewhere
    va_q = np.full(T, np.nan)
    for i, ts in enumerate(idx):
        if ts.month in (3, 6, 9, 12):
            va_q[i] = va_raw[i]

    panel = pd.DataFrame(
        {"monthly_a": monthly_a, "monthly_b": monthly_b, "va_construction": va_q},
        index=idx,
    )
    return panel


@pytest.fixture(scope="module")
def fitted_dfm(synthetic_panel):
    """DynamicFactorModel fitted on the synthetic panel."""
    from lamiaty.model.dfm import DynamicFactorModel
    from lamiaty.config.settings import ModelSettings, MixedFrequencySettings, SampleSettings, BacktestSettings

    ms = ModelSettings(
        n_factors=1,
        n_lags=1,
        max_iterations=200,
        tolerance=1e-4,
        mixed_frequency=MixedFrequencySettings(
            quarterly_series=["va_construction"],
            aggregation="sum",
        ),
        sample=SampleSettings(
            in_sample_start="2015-01",
            in_sample_end="2019-12",
            full_sample_start="2015-01",
            full_sample_end="2019-12",
        ),
        backtest=BacktestSettings(
            origin_start="2018-01",
            origin_end="2019-12",
            update_days=[7, 21],
            rolling_window_years=3,
        ),
    )

    dfm = DynamicFactorModel(n_factors=1, n_lags=1, settings=ms)
    dfm.fit(synthetic_panel)
    return dfm
