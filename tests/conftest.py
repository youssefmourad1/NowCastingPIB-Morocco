"""
Shared pytest fixtures for the lamiaty test suite.

The mini_base.xlsx fixture (tests/fixtures/mini_base.xlsx) is a synthetic
30-row Excel file designed to exercise all edge cases:
  - Pre/post cement unit break (April 2022)
  - LafargeHolcim values formatted as comma-separated strings
  - Negative January values in Investissement_Etat
  - Quarterly series (VA CONSTRUCTION, IPAI, emploi) with values repeated
    across all 3 months of each quarter (raw format from HCP)
"""

from __future__ import annotations

from pathlib import Path

import numpy as np
import pandas as pd
import pytest

FIXTURES_DIR = Path(__file__).parent / "fixtures"
MINI_BASE_PATH = FIXTURES_DIR / "mini_base.xlsx"


@pytest.fixture(scope="session")
def mini_base_path() -> Path:
    """Path to the synthetic mini_base.xlsx fixture."""
    return MINI_BASE_PATH


@pytest.fixture(scope="session")
def raw_mini_df() -> pd.DataFrame:
    """Load mini_base.xlsx as a raw DataFrame (before any corrections).

    The LafargeHolcim column remains as object dtype (comma-formatted strings).
    """
    df = pd.read_excel(MINI_BASE_PATH, engine="openpyxl")
    return df


@pytest.fixture
def cement_series_with_break() -> pd.Series:
    """Synthetic cement series with a structural break in April 2022.

    Pre-break values: ~1000 (old unit)
    Post-break values: ~759000 (new unit — approximately ×759)
    """
    dates = pd.date_range("2020-01", periods=30, freq="MS")
    values = np.ones(30) * 1000.0
    # Post-break: multiply by 759
    break_idx = dates.get_loc(pd.Timestamp("2022-04-01"))
    values[break_idx:] = 1000.0 * 759
    return pd.Series(values, index=dates, name="consommation_ciment")


@pytest.fixture
def investissement_series_with_negative_january() -> pd.Series:
    """Synthetic Investissement_Etat series with negative January values.

    Simulates YTD cumulative behaviour: negative at year start, growing positive.
    """
    dates = pd.date_range("2020-01", periods=30, freq="MS")
    values = []
    for i, d in enumerate(dates):
        if d.month == 1:
            values.append(-50000.0)
        else:
            values.append(float(i * 500))
    return pd.Series(values, index=dates, name="investissement_etat")


@pytest.fixture
def quarterly_va_series() -> pd.Series:
    """Synthetic VA CONSTRUCTION series — quarterly value repeated 3× per quarter.

    This is the raw format from HCP before DFM correction.
    Q1 (Jan-Mar): 16000, Q2 (Apr-Jun): 17000, ...
    """
    dates = pd.date_range("2020-01", periods=30, freq="MS")
    quarterly_values = {
        1: 16000.0, 2: 16000.0, 3: 16000.0,   # Q1 2020
        4: 17000.0, 5: 17000.0, 6: 17000.0,   # Q2 2020
        7: 15000.0, 8: 15000.0, 9: 15000.0,   # Q3 2020 (COVID dip)
        10: 16500.0, 11: 16500.0, 12: 16500.0, # Q4 2020
        13: 17500.0, 14: 17500.0, 15: 17500.0, # Q1 2021
        16: 18000.0, 17: 18000.0, 18: 18000.0, # Q2 2021
        19: 18500.0, 20: 18500.0, 21: 18500.0, # Q3 2021
        22: 19000.0, 23: 19000.0, 24: 19000.0, # Q4 2021
        25: 19500.0, 26: 19500.0, 27: 19500.0, # Q1 2022
        28: 20000.0, 29: 20000.0, 30: 20000.0, # Q2 2022
    }
    values = [quarterly_values[i + 1] for i in range(30)]
    return pd.Series(values, index=dates, name="va_construction")
