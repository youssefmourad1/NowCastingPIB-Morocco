"""
Tests for src/lamiaty/data/transforms.py
"""

import numpy as np
import pandas as pd
import pytest

from lamiaty.data.transforms import (
    QUARTER_END_MONTHS,
    assign_quarterly_to_month_end,
    merge_and_deduplicate,
    resample_daily_to_monthly_first,
    standardize,
    yoy_log_diff,
)


# ---------------------------------------------------------------------------
# yoy_log_diff
# ---------------------------------------------------------------------------


class TestYoyLogDiff:
    def test_produces_12_leading_nans(self):
        dates = pd.date_range("2010-01", periods=36, freq="MS")
        s = pd.Series(np.ones(36) * 100.0, index=dates)
        result = yoy_log_diff(s)
        assert result.iloc[:12].isna().all()

    def test_constant_series_zero_after_lag(self):
        """A constant series should produce 0 yoy growth everywhere (after first 12)."""
        dates = pd.date_range("2010-01", periods=36, freq="MS")
        s = pd.Series(np.ones(36) * 100.0, index=dates)
        result = yoy_log_diff(s)
        non_nan = result.dropna()
        assert (non_nan.abs() < 1e-10).all()

    def test_name_preserved(self):
        dates = pd.date_range("2010-01", periods=36, freq="MS")
        s = pd.Series(np.ones(36), index=dates, name="test_series")
        result = yoy_log_diff(s)
        assert result.name == "test_series"

    def test_10pct_growth_approx(self):
        """A series growing 10% per year should yield ~10 pp yoy log-diff."""
        dates = pd.date_range("2010-01", periods=25, freq="MS")
        monthly_growth = (1.10 ** (1 / 12))
        values = np.cumprod(np.full(25, monthly_growth)) * 100
        s = pd.Series(values, index=dates)
        result = yoy_log_diff(s).dropna()
        assert (result.abs() - 10.0).abs().max() < 1.0


# ---------------------------------------------------------------------------
# standardize
# ---------------------------------------------------------------------------


class TestStandardize:
    def test_mean_near_zero(self):
        s = pd.Series([1.0, 2.0, 3.0, 4.0, 5.0])
        result = standardize(s)
        assert abs(result.mean()) < 1e-10

    def test_std_near_one(self):
        s = pd.Series([1.0, 2.0, 3.0, 4.0, 5.0])
        result = standardize(s)
        assert abs(result.std() - 1.0) < 1e-6

    def test_nan_ignored(self):
        s = pd.Series([1.0, np.nan, 3.0, 4.0, 5.0])
        result = standardize(s)
        non_nan = result.dropna()
        assert abs(non_nan.mean()) < 1e-10


# ---------------------------------------------------------------------------
# assign_quarterly_to_month_end
# ---------------------------------------------------------------------------


class TestAssignQuarterlyToMonthEnd:
    def test_non_quarter_end_months_are_nan(self, quarterly_va_series):
        result = assign_quarterly_to_month_end(quarterly_va_series)
        non_q_end_mask = ~result.index.month.isin(QUARTER_END_MONTHS)
        assert result[non_q_end_mask].isna().all(), (
            "Non-quarter-end months must be NaN for correct DFM treatment (§3.3)"
        )

    def test_quarter_end_months_retain_values(self, quarterly_va_series):
        result = assign_quarterly_to_month_end(quarterly_va_series)
        q_end_mask = result.index.month.isin(QUARTER_END_MONTHS)
        # Quarter-end values should not be NaN where original had values
        original_q_end = quarterly_va_series[q_end_mask]
        assert result[q_end_mask].notna().all()
        assert (result[q_end_mask] == original_q_end).all()

    def test_correct_month_convention(self, quarterly_va_series):
        """Q1 value appears in March row, Q2 in June, Q3 in September, Q4 in December."""
        result = assign_quarterly_to_month_end(quarterly_va_series)
        # Q1 2020: Jan=NaN, Feb=NaN, Mar=16000
        assert pd.isna(result.loc[pd.Timestamp("2020-01-01")])
        assert pd.isna(result.loc[pd.Timestamp("2020-02-01")])
        assert result.loc[pd.Timestamp("2020-03-01")] == pytest.approx(16000.0)
        # Q2 2020: Apr=NaN, May=NaN, Jun=17000
        assert pd.isna(result.loc[pd.Timestamp("2020-04-01")])
        assert result.loc[pd.Timestamp("2020-06-01")] == pytest.approx(17000.0)

    def test_result_length_unchanged(self, quarterly_va_series):
        result = assign_quarterly_to_month_end(quarterly_va_series)
        assert len(result) == len(quarterly_va_series)

    def test_only_one_third_non_nan(self, quarterly_va_series):
        """Exactly 1/3 of rows should be non-NaN for a perfectly quarterly series."""
        result = assign_quarterly_to_month_end(quarterly_va_series)
        # With 30 months (Jan 2020 – Jun 2022), quarter-end months: Mar,Jun,Sep,Dec,Mar,Jun,Sep,Dec,Mar,Jun = 10
        n_non_nan = result.notna().sum()
        assert n_non_nan == 10


# ---------------------------------------------------------------------------
# resample_daily_to_monthly_first
# ---------------------------------------------------------------------------


class TestResampleDailyToMonthlyFirst:
    def test_produces_one_row_per_month(self):
        dates = pd.date_range("2020-01-01", periods=90, freq="D")
        df = pd.DataFrame({"date": dates, "value": np.random.randn(90)})
        monthly = resample_daily_to_monthly_first(df, date_col="date", value_col="value")
        # 90 days spans ~3 months
        assert len(monthly) == 3

    def test_takes_first_day(self):
        # First record of January 2020 should be 2020-01-01
        dates = pd.date_range("2020-01-01", periods=62, freq="D")
        values = np.arange(62, dtype=float)
        df = pd.DataFrame({"date": dates, "value": values})
        monthly = resample_daily_to_monthly_first(df)
        assert monthly.index[0] == pd.Timestamp("2020-01-01")


# ---------------------------------------------------------------------------
# merge_and_deduplicate
# ---------------------------------------------------------------------------


class TestMergeAndDeduplicate:
    def test_deduplicates_overlapping_dates(self):
        dates1 = pd.date_range("2020-01-01", periods=3, freq="D")
        dates2 = pd.date_range("2020-01-02", periods=3, freq="D")  # overlap on Jan 2-3
        df1 = pd.DataFrame({"date": dates1, "value": [1.0, 2.0, 3.0]})
        df2 = pd.DataFrame({"date": dates2, "value": [20.0, 30.0, 40.0]})
        result = merge_and_deduplicate([df1, df2], date_col="date")
        # Should have 4 unique dates: Jan 1, 2, 3, 4
        assert len(result) == 4

    def test_first_occurrence_wins(self):
        """When dates overlap, the first DataFrame's value is kept."""
        date = pd.Timestamp("2020-01-15")
        df1 = pd.DataFrame({"date": [date], "value": [100.0]})
        df2 = pd.DataFrame({"date": [date], "value": [999.0]})
        result = merge_and_deduplicate([df1, df2], date_col="date")
        assert result.loc[result["date"] == date, "value"].iloc[0] == pytest.approx(100.0)

    def test_sorted_ascending(self):
        dates1 = pd.date_range("2020-03-01", periods=3, freq="D")
        dates2 = pd.date_range("2020-01-01", periods=3, freq="D")
        df1 = pd.DataFrame({"date": dates1, "value": np.ones(3)})
        df2 = pd.DataFrame({"date": dates2, "value": np.ones(3)})
        result = merge_and_deduplicate([df1, df2], date_col="date")
        assert result["date"].is_monotonic_increasing
