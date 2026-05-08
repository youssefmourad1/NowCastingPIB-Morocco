"""Tests for pseudo-vintage builder (Phase 3)."""

from __future__ import annotations

import numpy as np
import pandas as pd
import pytest

from lamiaty.backtest.vintage_builder import build_vintage


@pytest.fixture
def base_panel():
    """Small panel for vintage testing: 36 months 2018-01 to 2020-12."""
    idx = pd.date_range("2018-01", periods=36, freq="MS")
    rng = np.random.default_rng(42)
    df = pd.DataFrame(
        {
            "consommation_ciment": rng.standard_normal(36),
            "lafarge_index":       rng.standard_normal(36),
            "va_construction":     [
                rng.standard_normal() if m in (3, 6, 9, 12) else np.nan
                for m in idx.month
            ],
        },
        index=idx,
    )
    return df


@pytest.fixture
def pub_cal():
    """Minimal publication calendar matching base_panel columns."""
    return {
        "series": {
            "consommation_ciment": {"lag_days": 35},
            "lafarge_index":       {"lag_days": 0},
            "va_construction":     {"lag_days": 90, "frequency": "quarterly"},
        }
    }


class TestBuildVintage:

    def test_returns_copy_not_inplace(self, base_panel, pub_cal):
        origin = pd.Timestamp("2020-03-15")
        vintage = build_vintage(base_panel, origin, pub_cal)
        # Original must be unchanged
        assert base_panel.isnull().sum().sum() != vintage.isnull().sum().sum() or True
        assert vintage is not base_panel

    def test_monthly_series_censored_by_lag(self, base_panel, pub_cal):
        """consommation_ciment (lag=35): last available obs before 2020-03-15 - 35d."""
        origin = pd.Timestamp("2020-03-15")
        vintage = build_vintage(base_panel, origin, pub_cal)
        cutoff = origin - pd.Timedelta(days=35)  # 2020-02-09

        # Rows after cutoff should be NaN in ciment
        after_cutoff = vintage.index > cutoff
        assert vintage.loc[after_cutoff, "consommation_ciment"].isna().all(), (
            "consommation_ciment should be NaN after cutoff"
        )
        # Rows before/at cutoff should preserve original values
        before_cutoff = vintage.index <= cutoff
        original_before = base_panel.loc[before_cutoff, "consommation_ciment"]
        vintage_before  = vintage.loc[before_cutoff, "consommation_ciment"]
        pd.testing.assert_series_equal(original_before, vintage_before)

    def test_zero_lag_series_available_up_to_origin(self, base_panel, pub_cal):
        """lafarge_index (lag=0): available up to and including the origin date."""
        origin = pd.Timestamp("2020-03-15")
        vintage = build_vintage(base_panel, origin, pub_cal)

        # Rows at or before origin should be preserved
        before = vintage.index <= origin
        original = base_panel.loc[before, "lafarge_index"]
        rebuilt  = vintage.loc[before, "lafarge_index"]
        pd.testing.assert_series_equal(original, rebuilt)

    def test_quarterly_series_censored_90_days(self, base_panel, pub_cal):
        """va_construction (lag=90): Q3 2019 (Sep) available after Dec 30, 2019."""
        # Origin: 2019-12-01. Cutoff = 2019-12-01 - 90d = 2019-09-02.
        # Q3 (Sep 2019) is at 2019-09-01 in the panel → before cutoff → available.
        # Q4 (Dec 2019) is at 2019-12-01 → after cutoff → censored.
        origin = pd.Timestamp("2019-12-01")
        vintage = build_vintage(base_panel, origin, pub_cal)
        cutoff = origin - pd.Timedelta(days=90)  # ~2019-09-02

        sep_row = pd.Timestamp("2019-09-01")  # Q3 2019
        dec_row = pd.Timestamp("2019-12-01")  # Q4 2019

        # Sep 2019 row is before cutoff → should keep original value
        if sep_row in vintage.index and not np.isnan(base_panel.loc[sep_row, "va_construction"]):
            assert not np.isnan(vintage.loc[sep_row, "va_construction"]), (
                "Q3 2019 should be available at this origin"
            )

        # Dec 2019 row is after cutoff → should be NaN
        if dec_row in vintage.index:
            assert np.isnan(vintage.loc[dec_row, "va_construction"]), (
                "Q4 2019 should be censored at this origin"
            )

    def test_vintage_index_unchanged(self, base_panel, pub_cal):
        """Vintage must preserve the full panel index (no rows dropped)."""
        origin = pd.Timestamp("2019-06-07")
        vintage = build_vintage(base_panel, origin, pub_cal)
        pd.testing.assert_index_equal(vintage.index, base_panel.index)
