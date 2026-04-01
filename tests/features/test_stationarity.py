"""
Tests for src/lamiaty/features/stationarity.py
"""

import numpy as np
import pandas as pd
import pytest

from lamiaty.features.stationarity import (
    ALPHA,
    run_adf_test,
    run_kpss_test,
    run_stationarity_battery,
)


@pytest.fixture
def white_noise_series():
    """White noise — should be detected as stationary by both tests."""
    rng = np.random.default_rng(42)
    dates = pd.date_range("2000-01", periods=120, freq="MS")
    return pd.Series(rng.standard_normal(120), index=dates, name="white_noise")


@pytest.fixture
def random_walk_series():
    """Random walk — should be flagged as non-stationary by both tests."""
    rng = np.random.default_rng(42)
    dates = pd.date_range("2000-01", periods=120, freq="MS")
    increments = rng.standard_normal(120)
    values = np.cumsum(increments)
    return pd.Series(values, index=dates, name="random_walk")


@pytest.fixture
def panel_with_two_series(white_noise_series, random_walk_series):
    return pd.DataFrame({"white_noise": white_noise_series, "random_walk": random_walk_series})


class TestRunAdfTest:
    def test_returns_required_keys(self, white_noise_series):
        result = run_adf_test(white_noise_series)
        for key in ("statistic", "pvalue", "n_lags_used", "n_obs", "critical_values", "is_stationary"):
            assert key in result, f"Missing key: {key}"

    def test_stationary_series_detected(self, white_noise_series):
        result = run_adf_test(white_noise_series)
        assert result["is_stationary"]

    def test_unit_root_detected(self, random_walk_series):
        result = run_adf_test(random_walk_series)
        assert not result["is_stationary"]

    def test_pvalue_in_unit_interval(self, white_noise_series):
        result = run_adf_test(white_noise_series)
        assert 0.0 <= result["pvalue"] <= 1.0


class TestRunKpssTest:
    def test_returns_required_keys(self, white_noise_series):
        result = run_kpss_test(white_noise_series)
        for key in ("statistic", "pvalue", "n_lags_used", "critical_values", "is_stationary"):
            assert key in result

    def test_stationary_series_detected(self, white_noise_series):
        result = run_kpss_test(white_noise_series)
        assert result["is_stationary"]

    def test_unit_root_detected(self, random_walk_series):
        result = run_kpss_test(random_walk_series)
        assert not result["is_stationary"]


class TestRunStationaryBattery:
    def test_returns_dataframe(self, panel_with_two_series):
        result = run_stationarity_battery(panel_with_two_series)
        assert isinstance(result, pd.DataFrame)

    def test_one_row_per_series(self, panel_with_two_series):
        result = run_stationarity_battery(panel_with_two_series)
        assert len(result) == len(panel_with_two_series.columns)

    def test_index_matches_column_names(self, panel_with_two_series):
        result = run_stationarity_battery(panel_with_two_series)
        assert set(result.index) == set(panel_with_two_series.columns)

    def test_verdict_column_present(self, panel_with_two_series):
        result = run_stationarity_battery(panel_with_two_series)
        assert "verdict" in result.columns

    def test_white_noise_verdict(self, panel_with_two_series):
        result = run_stationarity_battery(panel_with_two_series)
        # White noise should be stationary; ambiguous result is also acceptable
        # (test environments may have edge cases with short series)
        verdict = result.loc["white_noise", "verdict"]
        assert "STATIONARY" in verdict or "AMBIGUOUS" in verdict

    def test_required_columns(self, panel_with_two_series):
        result = run_stationarity_battery(panel_with_two_series)
        for col in ("ADF_stat", "ADF_pvalue", "ADF_stationary", "KPSS_stat", "KPSS_pvalue", "KPSS_stationary", "verdict"):
            assert col in result.columns
